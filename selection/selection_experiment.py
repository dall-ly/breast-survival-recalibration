"""Frozen calibration-only two-stage OOB comparison; no base-model training.

The decision file is persisted before any evaluation labels or archived Brier
scores are used. Run --mode pilot for implementation QA, then --mode full.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq

B = Path(__file__).resolve().parent
DEFAULT_INPUT = B.parent / "sensitivity"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def predict(p, a):
    if a == 0.:
        return p.copy()
    if np.isneginf(a):
        return np.zeros_like(p)
    if np.isposinf(a):
        return np.ones_like(p)
    h0 = -np.log1p(-np.clip(p, 1e-10, 1. - 1e-10))
    return -np.expm1(-np.exp(a) * h0)


def intercepts(p, y_all, w_all, kappa):
    active = w_all > 0
    if not active.any():
        return 0., 0., "no_observable_outcomes", 0.
    y, w = y_all[active], w_all[active]
    h0 = -np.log1p(-np.clip(p[active], 1e-10, 1. - 1e-10))
    event = y == 1
    event_h0, event_w = h0[event], w[event]
    survivor_hazard = float(w[~event] @ h0[~event])

    def score(a, penalty):
        scale = np.exp(a)
        h = scale * event_h0
        # Stable equivalent of -h/(exp(h)-1), avoiding exp(h) overflow.
        event_derivative = h * np.exp(-h) / (-np.expm1(-h))
        return scale * survivor_hazard - float(event_w @ event_derivative) + penalty * a

    if not event.any():
        ordinary, status = -np.inf, "no_events"
    elif event.all():
        ordinary, status = np.inf, "all_events"
    else:
        ordinary = float(brentq(lambda a: score(a, 0.), -30., 10., xtol=1e-11))
        status = "interior"
    shrunk = float(brentq(lambda a: score(a, kappa), -30., 10., xtol=1e-11))
    residual = abs(score(shrunk, kappa))
    if np.isfinite(ordinary):
        residual = max(residual, abs(score(ordinary, 0.)))
    assert residual < 1e-6, residual
    return shrunk, ordinary, status, residual


def choose(scores, alpha):
    med = np.median(scores, axis=0)
    best = int(np.argmin(med))
    intervals = np.empty((3, 3, 2), dtype=float)
    for first in range(3):
        for second in range(3):
            intervals[first, second] = np.quantile(
                scores[:, first] - scores[:, second], [alpha / 2, 1 - alpha / 2]
            )
    simple = best
    for candidate in range(best):
        lo, hi = intervals[candidate, best]
        if lo <= 0 <= hi:
            simple = candidate
            break
    return best, simple, med, intervals


def independently_check_decisions(saved, candidate_order):
    # Does not call choose(); checks the independently persisted summaries.
    count = 0
    for row in saved.to_dict("records"):
        reference = min(range(3), key=lambda k: (row[f"median_{k}"], k))
        assert candidate_order[reference] == row["OOB_min"]
        decision = reference
        for earlier in range(reference):
            if row[f"lo_{earlier}_{reference}"] <= 0 and row[f"hi_{earlier}_{reference}"] >= 0:
                decision = earlier
                break
        assert candidate_order[decision] == row["OOB_simple"]
        count += 1
    return count


def paired_summaries(results):
    keys = ["evaluation", "cohort", "budget", "model", "repeat", "method"]
    individual = results.pivot(index=keys[:-1], columns="method", values="brier").reset_index()
    average = results.groupby(["evaluation", "cohort", "budget", "repeat", "method"]).brier.mean().unstack().reset_index()
    average["model"] = "Average"
    both = pd.concat([individual, average], ignore_index=True)
    records = []
    for keys, group in both.groupby(["evaluation", "cohort", "budget", "model"]):
        for method in ["shrink_intercept", "intercept_ML", "OOB_min", "OOB_simple"]:
            for reference in ["unchanged", "shrink_intercept", "intercept_ML"]:
                if method == reference:
                    continue
                delta = group[method] - group[reference]
                records.append(dict(zip(["evaluation", "cohort", "budget", "model"], keys),
                    method=method, reference=reference, n_allocations=len(group),
                    mean_delta=float(delta.mean()), median_delta=float(delta.median()),
                    q90_delta=float(delta.quantile(.9)), fraction_better=float((delta < 0).mean()),
                    fraction_harmed=float((delta > 0).mean()), fraction_equal=float((delta == 0).mean())))
    return both, pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "full"], required=True)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    started = time.time()
    protocol_path = B / "selection_protocol.json"
    proto = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
    assert proto["candidate_order"] == ["unchanged", "shrink_intercept", "intercept_ML"]
    assert proto["B1"] == 100 and proto["B2"] == 1000 and proto["kappa"] == 5 and proto["alpha"] == .05
    repeats = 10 if args.mode == "pilot" else proto["outer_repetitions"]
    output = args.output or B / ("selection_pilot" if args.mode == "pilot" else "selection_results")
    output.mkdir(parents=True, exist_ok=True)
    source = args.inputs / "inputs"
    sys.path.insert(0, str(source))
    from calibration import weights, objective
    prediction_file = source / "fixed_predictions.csv"
    archive_file = source / "archived_results" / "primary_results.csv"
    common_file = args.inputs / "output" / "split_results.csv"
    input_paths = [protocol_path, prediction_file, source / "calibration.py", archive_file, common_file]
    input_hashes = {str(p): sha(p) for p in input_paths}
    data = pd.read_csv(prediction_file, float_precision="round_trip")
    candidates, models = proto["candidate_order"], proto["models"]
    decisions, coverage, full_coefficients = [], [], []
    boundaries = {"interior": 0, "no_events": 0, "all_events": 0, "no_observable_outcomes": 0}
    max_residual, max_objective_disagreement = 0., 0.
    fit_path = output / "inner_coefficients.csv"
    fields = ["cohort", "repeat", "budget", "bootstrap", "model", "shrink_intercept", "ordinary_intercept", "boundary", "score_residual", "inbag_events", "inbag_observable", "n_oob"]
    with fit_path.open("w", newline="", encoding="utf-8") as fit_file:
        fit_writer = csv.DictWriter(fit_file, fieldnames=fields)
        fit_writer.writeheader()
        for ci, cohort in enumerate(proto["cohorts"]):
            d = data[data.cohort == cohort].sort_values("patient_id").reset_index(drop=True)
            budget = proto["main_budgets"][cohort]
            for repeat in range(repeats):
                order = np.random.default_rng(proto["outer_seed"] + 100000 * ci + repeat).permutation(len(d))
                train = order[:budget]
                cal = d.iloc[train]
                t, event = cal.os_days.to_numpy(), cal.os_event.to_numpy()
                cal_p = cal[["p_" + m for m in models]].to_numpy()
                cal_y, cal_w = weights(t, event)
                assert np.isfinite(cal_w).all()
                final_params = []
                for mi, model in enumerate(models):
                    shrunk, ordinary, status, resid = intercepts(cal_p[:, mi], cal_y, cal_w, proto["kappa"])
                    final_params.append([0., shrunk, ordinary])
                    full_coefficients.append({"cohort": cohort, "repeat": repeat, "budget": budget, "model": model,
                        "shrink_intercept": shrunk, "ordinary_intercept": ordinary, "boundary": status, "score_residual": resid})
                    # Verify optimized scalar derivative against archived objective API.
                    active = cal_w > 0
                    x = np.log(-np.log1p(-np.clip(cal_p[active, mi], 1e-10, 1 - 1e-10)))
                    g = objective([shrunk, 1.], x, cal_y[active], cal_w[active])[1][0] * cal_w[active].sum() + 5 * shrunk
                    max_objective_disagreement = max(max_objective_disagreement, abs(g))
                rng = np.random.default_rng(np.random.SeedSequence([proto["inner_seed"], ci, repeat]))
                pooled_losses, pooled_ids, pooled_probs, inbag_hash_parts = [], [], [], []
                oob_counts = np.zeros(budget, dtype=int)
                for bootstrap in range(proto["B1"]):
                    inbag = rng.integers(0, budget, size=budget)
                    oob = np.flatnonzero(np.bincount(inbag, minlength=budget) == 0)
                    inbag_hash_parts.append(inbag)
                    oob_counts[oob] += 1
                    y, w = weights(t[inbag], event[inbag])
                    probabilities = np.empty((len(oob), 3, 3), dtype=float)
                    for mi, model in enumerate(models):
                        shrunk, ordinary, status, resid = intercepts(cal_p[inbag, mi], y, w, proto["kappa"])
                        boundaries[status] += 1
                        max_residual = max(max_residual, resid)
                        probabilities[:, mi, 0] = cal_p[oob, mi]
                        probabilities[:, mi, 1] = predict(cal_p[oob, mi], shrunk)
                        probabilities[:, mi, 2] = predict(cal_p[oob, mi], ordinary)
                        fit_writer.writerow({"cohort": cohort, "repeat": repeat, "budget": budget, "bootstrap": bootstrap,
                            "model": model, "shrink_intercept": shrunk, "ordinary_intercept": ordinary,
                            "boundary": status, "score_residual": resid, "inbag_events": int(y.sum()),
                            "inbag_observable": int((w > 0).sum()), "n_oob": len(oob)})
                    assert np.isfinite(probabilities).all() and (probabilities >= 0).all() and (probabilities <= 1).all()
                    pooled_losses.append(cal_w[oob, None, None] * (cal_y[oob, None, None] - probabilities) ** 2)
                    pooled_ids.append(oob)
                    if ci == 0 and repeat == 0:
                        pooled_probs.append(probabilities)
                losses = np.concatenate(pooled_losses, axis=0)
                pool_ids = np.concatenate(pooled_ids)
                assert len(losses) > 0 and np.all(oob_counts > 0)
                second_indices = rng.integers(0, len(losses), size=(proto["B2"], budget))
                scores = losses[second_indices].mean(axis=1)
                assert scores.shape == (proto["B2"], 3, 3) and np.isfinite(scores).all()
                for mi, model in enumerate(models):
                    best, simple, med, intervals = choose(scores[:, mi, :], proto["alpha"])
                    row = {"cohort": cohort, "repeat": repeat, "budget": budget, "model": model,
                        "OOB_min": candidates[best], "OOB_simple": candidates[simple], "calibration_events": int(cal_y.sum()),
                        "OOB_min_intercept": final_params[mi][best], "OOB_simple_intercept": final_params[mi][simple],
                        "pool_rows": len(losses), "calibration_id_hash": fingerprint(cal.patient_id.to_numpy(dtype="U")),
                        "bootstrap_indices_hash": fingerprint(np.array(inbag_hash_parts)), "stage2_indices_hash": fingerprint(second_indices)}
                    row.update({f"median_{k}": float(med[k]) for k in range(3)})
                    for first in range(3):
                        for second in range(3):
                            row[f"lo_{first}_{second}"] = intervals[first, second, 0]
                            row[f"hi_{first}_{second}"] = intervals[first, second, 1]
                    decisions.append(row)
                coverage.append({"cohort": cohort, "repeat": repeat, "budget": budget, "pool_rows": len(losses),
                    "oob_min": int(oob_counts.min()), "oob_max": int(oob_counts.max()), "oob_mean": float(oob_counts.mean()),
                    "oob_count_by_calibration_position": ";".join(map(str, oob_counts))})
                if ci == 0 and repeat == 0:
                    np.savez_compressed(output / "first_allocation_trace.npz", calibration_p=cal_p, calibration_t=t,
                        calibration_event=event, calibration_y=cal_y, calibration_w=cal_w,
                        inbag_indices=np.array(inbag_hash_parts), pool_patient_positions=pool_ids,
                        pool_probabilities=np.concatenate(pooled_probs), pool_losses=losses,
                        second_stage_indices=second_indices, second_stage_scores=scores)
                if (repeat + 1) % 10 == 0:
                    print(f"Selection {cohort}: {repeat + 1}/{repeats}; elapsed {time.time() - started:.1f}s", flush=True)

    # Persist all calibration-only decisions BEFORE reading any evaluation outcomes.
    decision_path = output / "selection_decisions.csv"
    pd.DataFrame(decisions).to_csv(decision_path, index=False)
    pd.DataFrame(full_coefficients).to_csv(output / "full_calibration_coefficients.csv", index=False)
    pd.DataFrame(coverage).to_csv(output / "oob_coverage.csv", index=False)
    decision_hash = sha(decision_path)
    saved = pd.read_csv(decision_path, float_precision="round_trip")
    checked = independently_check_decisions(saved, candidates)
    assert checked == repeats * 9
    for _, g in saved.groupby(["cohort", "repeat"]):
        assert g.bootstrap_indices_hash.nunique() == 1 and g.stage2_indices_hash.nunique() == 1
    freeze = {"phase": "calibration_only_selection_finished_before_evaluation", "decisions": checked,
        "decision_sha256": decision_hash, "seconds_to_decision_freeze": time.time() - started}
    (output / "selection_freeze.json").write_text(json.dumps(freeze, indent=2), encoding="utf-8")

    # Evaluation starts here; original archived baseline coefficients are now read.
    archive = pd.read_csv(archive_file, float_precision="round_trip")
    archive = archive.set_index(["cohort", "repeat", "budget", "model", "method"], verify_integrity=True)
    archived_common = pd.read_csv(common_file, float_precision="round_trip")
    archived_common = archived_common[archived_common.method.isin(candidates)].set_index(
        ["evaluation", "cohort", "repeat", "budget", "model", "method"], verify_integrity=True)
    full = pd.DataFrame(full_coefficients).set_index(["cohort", "repeat", "budget", "model"])
    selected = saved.set_index(["cohort", "repeat", "budget", "model"])
    evaluations, max_brier_error, max_parameter_error = [], 0., 0.
    for ci, cohort in enumerate(proto["cohorts"]):
        d = data[data.cohort == cohort].sort_values("patient_id").reset_index(drop=True)
        budget = proto["main_budgets"][cohort]
        t, event = d.os_days.to_numpy(), d.os_event.to_numpy()
        for repeat in range(repeats):
            order = np.random.default_rng(proto["outer_seed"] + 100000 * ci + repeat).permutation(len(d))
            train = order[:budget]
            for design, ix in [("remaining", order[budget:]), ("common", order[proto["common_max_budgets"][cohort]:])]:
                assert not np.intersect1d(train, ix).size
                y, w = weights(t[ix], event[ix])
                for model in models:
                    key = (cohort, repeat, budget, model)
                    p = d["p_" + model].to_numpy()[ix]
                    params = {method: float(archive.loc[key + (method,), "intercept"]) for method in candidates}
                    for method, column in [("shrink_intercept", "shrink_intercept"), ("intercept_ML", "ordinary_intercept")]:
                        expected, found = params[method], float(full.loc[key, column])
                        if np.isfinite(expected) and np.isfinite(found):
                            max_parameter_error = max(max_parameter_error, abs(expected - found))
                        else:
                            assert expected == found
                    for method in candidates + ["OOB_min", "OOB_simple"]:
                        chosen = method if method in candidates else selected.loc[key, method]
                        q = p if chosen == "unchanged" else predict(p, params[chosen])
                        value = float(w @ ((y - q) ** 2) / len(ix))
                        assert np.isfinite(value)
                        expected = float(archived_common.loc[(design,) + key + (chosen,), "brier"])
                        max_brier_error = max(max_brier_error, abs(value - expected))
                        if method in candidates and design == "remaining":
                            max_brier_error = max(max_brier_error, abs(value - float(archive.loc[key + (method,), "brier"])))
                        evaluations.append({"evaluation": design, "cohort": cohort, "repeat": repeat, "budget": budget,
                            "model": model, "method": method, "selected_method": chosen,
                            "intercept": params[chosen], "slope": 1., "n_test": len(ix), "brier": value})
    assert max_brier_error <= 1e-10, max_brier_error
    assert max_parameter_error <= 1e-8, max_parameter_error
    assert max_objective_disagreement < 1e-6, max_objective_disagreement
    results = pd.DataFrame(evaluations)
    assert len(results) == repeats * 3 * 3 * 5 * 2
    results.to_csv(output / "evaluation_results.csv", index=False)
    results.groupby(["evaluation", "cohort", "budget", "model", "method"]).brier.mean().unstack().reset_index().to_csv(output / "model_means.csv", index=False)
    results.groupby(["evaluation", "cohort", "budget", "method"]).brier.mean().unstack().reset_index().to_csv(output / "aggregate_means.csv", index=False)
    both, paired = paired_summaries(results)
    both.to_csv(output / "paired_allocation_scores.csv", index=False)
    paired.to_csv(output / "paired_comparisons.csv", index=False)
    frequencies = saved.melt(id_vars=["cohort", "repeat", "budget", "model"], value_vars=["OOB_min", "OOB_simple"], var_name="selector", value_name="selected_method")
    freq = frequencies.groupby(["cohort", "budget", "model", "selector", "selected_method"]).size().rename("count").reset_index()
    freq["fraction"] = freq["count"] / repeats
    freq.to_csv(output / "selection_frequencies.csv", index=False)
    assert decision_hash == sha(decision_path)
    assert input_hashes == {str(p): sha(p) for p in input_paths}
    report = {"passed": True, "mode": args.mode, "repetitions_per_cohort": repeats, "decisions": checked,
        "evaluation_rows": len(results), "inner_fit_rows": repeats * 3 * 3 * proto["B1"],
        "boundary_counts_predictor_bootstrap": boundaries, "max_root_residual": max_residual,
        "max_archived_objective_gradient_residual": max_objective_disagreement,
        "max_archived_parameter_difference": max_parameter_error, "max_archived_Brier_difference": max_brier_error,
        "shared_inner_resampling_passed": True, "independent_selection_summary_recheck_passed": True,
        "all_oob_patient_coverage_positive": True, "decision_hash_unchanged_after_evaluation": True,
        "original_inputs_unchanged": True, "input_hashes": input_hashes, "seconds": time.time() - started}
    (output / "numerical_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "input_hashes"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
