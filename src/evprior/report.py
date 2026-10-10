"""Stage: render figures and REPORT.md from results/*.json. No result number is typed by hand."""
from __future__ import annotations

import json
import logging
import platform
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from .config import Config  # noqa: E402

log = logging.getLogger(__name__)

LABEL = {"static": "Pre-trip prior (no update)", "rate": "Rate extrapolation", "blend_rate": "Credibility blend (rate)",
         "blend_ratio": "Credibility blend (ratio)", "gbm_dyn": "Learned in-trip GBM", "kf_mult": "Kalman, multiplier",
         "kf_add": "Kalman, offset", "kf_full": "Kalman, multiplier + offset", "kf_carry": "Kalman + vehicle carry-over", "oracle_mult": "Oracle ceiling (hindsight)"}
PRIOR_LABEL = {"mean_intensity": "Mean kWh/km", "physics_ridge": "Physics regression", "gbm": "Gradient boosting (segments)",
               "physics_gbm": "Physics + boosting residual", "gbm_trip": "Gradient boosting (trip level)"}
REGIME_LABEL = {"full": "full prior (route history + weather)", "no_history": "no route-speed history", "naive": "naive prior (mean kWh/km)"}
MAIN_SCHEMES = ["group_cv", "forward_chain", "leave_vehicle_out", "season_out"]
SCHEME_LABEL = {"group_cv": "grouped 5-fold CV", "forward_chain": "forward chain by month", "leave_vehicle_out": "leave one vehicle out",
                "season_out": "season out (winter / non-winter)",
                "ved_transfer": "external: train on KU Leuven Teslas, test on VED Leafs", "ved_within": "external: grouped CV inside VED"}
COL = {"static": "#6b7280", "rate": "#d1d5db", "blend_rate": "#9ca3af", "blend_ratio": "#a78bfa", "gbm_dyn": "#f59e0b",
       "kf_mult": "#2563eb", "kf_add": "#60a5fa", "kf_full": "#1e40af", "kf_carry": "#dc2626", "oracle_mult": "#10b981"}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130, "axes.grid": True,
                     "grid.alpha": 0.25, "legend.frameon": False})


def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}%"


def rel(a: float, b: float) -> str:
    """Relative change of a versus b, signed (negative = a is better when metric is an error)."""
    return f"{100 * (a - b) / b:+.1f}%"


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def load(res: Path, name: str) -> dict:
    return json.loads((res / name).read_text(encoding="utf-8"))


def fig_prior(pre: dict, fig_dir: Path) -> None:
    reg = pre["full"]
    schemes = [s for s in MAIN_SCHEMES if s in reg]
    kinds = [k for k in PRIOR_LABEL if all(k in reg[s] for s in schemes)]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    w = 0.8 / len(kinds)
    for i, k in enumerate(kinds):
        v = [reg[s][k]["wape"] * 100 for s in schemes]
        lo = [(reg[s][k]["wape"] - reg[s][k]["wape_ci95"][0]) * 100 for s in schemes]
        hi = [(reg[s][k]["wape_ci95"][1] - reg[s][k]["wape"]) * 100 for s in schemes]
        ax.bar(np.arange(len(schemes)) + i * w, v, w, yerr=[lo, hi], label=PRIOR_LABEL[k], capsize=2, color=plt.cm.viridis(i / max(len(kinds) - 1, 1) * 0.85))
    ax.set_xticks(np.arange(len(schemes)) + w * (len(kinds) - 1) / 2)
    ax.set_xticklabels([SCHEME_LABEL[s].replace(" (", "\n(") for s in schemes])
    ax.set_ylabel("trip energy error, WAPE (%)")
    ax.set_title("Pre-trip energy prediction (95% trip-bootstrap intervals)")
    ax.legend(ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_prior_models.png")
    plt.close(fig)


def fig_handover(ho: dict, fig_dir: Path, scheme: str = "group_cv") -> None:
    regs = [r for r in REGIME_LABEL if r in ho]
    fig, axes = plt.subplots(1, len(regs), figsize=(4.2 * len(regs), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, regs):
        cur = ho[r][scheme]["curves"]
        for m in ("rate", "static", "blend_ratio", "gbm_dyn", "kf_mult", "oracle_mult"):
            xs = sorted(int(k) for k in cur[m])
            ax.plot(xs, [100 * cur[m][str(x)]["wape"] for x in xs], label=LABEL[m], color=COL[m], lw=1.6 if m in ("kf_mult", "static") else 1.1,
                    ls="--" if m == "oracle_mult" else "-")
        ax.set_yscale("log")
        ax.set_title(REGIME_LABEL[r], fontsize=9)
        ax.set_xlabel("distance driven (km)")
    axes[0].set_ylabel("remaining-energy error, WAPE (%)")
    axes[-1].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Handover curves, {SCHEME_LABEL[scheme]}", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_handover.png")
    plt.close(fig)


def fig_shift(dyn: dict, fig_dir: Path, cp: str = "km_10") -> None:
    methods = ["blend_ratio", "gbm_dyn", "kf_mult", "oracle_mult"]
    cols = [(r, s) for r in REGIME_LABEL for s in ("group_cv", "leave_vehicle_out", "season_out") if r in dyn and s in dyn[r] and cp in dyn[r][s]]
    M = np.array([[100 * (dyn[r][s][cp][m]["wape"] - dyn[r][s][cp]["static"]["wape"]) / dyn[r][s][cp]["static"]["wape"] for r, s in cols] for m in methods])
    fig, ax = plt.subplots(figsize=(9, 2.6))
    lim = 30.0   # colour scale saturates at +-30%; the printed numbers are exact
    ax.imshow(np.clip(M, -lim, lim), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{r}\n{s.replace('_', ' ')}" for r, s in cols], fontsize=6.5)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([LABEL[m] for m in methods])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:+.0f}%", ha="center", va="center", fontsize=7, color="white" if abs(M[i, j]) > 18 else "black")
    ax.grid(False)
    ax.set_title(f"Change in remaining-energy error vs the pre-trip prior at {cp.split('_')[1]} km driven (blue = better)", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_adaptation_gain.png")
    plt.close(fig)


def fig_calibration(dyn: dict, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    for ax, s in zip(axes, ("group_cv", "season_out")):
        if "full" not in dyn or s not in dyn["full"] or "km_5" not in dyn["full"][s]:
            continue
        for m in ("static", "kf_mult", "blend_ratio", "gbm_dyn"):
            d = dyn["full"][s]["km_5"].get(m, {}).get("picp_by_quantile")
            if d:
                xs = sorted(float(k) for k in d)
                ax.plot(xs, [d[str(x)] for x in xs], marker="o", ms=3, label=LABEL[m], color=COL[m])
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.set_title(SCHEME_LABEL[s], fontsize=9)
        ax.set_xlabel("nominal quantile level")
    axes[0].set_ylabel("share of outcomes below the quantile")
    axes[0].legend(fontsize=7)
    fig.suptitle("Quantile reliability of remaining-energy forecasts at 5 km driven (full prior)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_calibration.png")
    plt.close(fig)


def fig_robust(rob: dict, fig_dir: Path) -> None:
    if "full" not in rob:
        return
    schemes = [s for s in ("group_cv", "season_out") if s in rob["full"]]
    kinds = [k for k in ("dropout", "noise", "delay") if k in rob["full"][schemes[0]]]
    fig, axes = plt.subplots(len(schemes), len(kinds), figsize=(3.6 * len(kinds), 2.9 * len(schemes)), sharey="row", squeeze=False)
    for i, s in enumerate(schemes):
        r = rob["full"][s]
        for j, k in enumerate(kinds):
            ax = axes[i][j]
            lv = sorted(r[k], key=float)
            for m in ("static", "blend_ratio", "kf_mult"):   # rate extrapolation (30-36%) is in the table; it would flatten this scale
                ys = [100 * r[k][l]["km_10"][m]["wape"] if m in r[k][l]["km_10"] else np.nan for l in lv]
                ax.plot([float(l) for l in lv], ys, marker="o", ms=3, label=LABEL[m], color=COL[m], ls="--" if m == "static" else "-")
            if i == len(schemes) - 1:
                ax.set_xlabel({"dropout": "share of segments not observed", "noise": "relative sensor noise", "delay": "observation delay (segments)"}[k])
        axes[i][0].set_ylabel(f"{SCHEME_LABEL[s]}\nWAPE at 10 km (%)", fontsize=8)
    axes[0][-1].legend(fontsize=6.5)
    fig.suptitle("Robustness of online methods to degraded observations (full prior)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig5_robustness.png")
    plt.close(fig)


def fig_importance(imp: dict, abl: dict, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    items = list(imp["importance"].items())[:12][::-1]
    axes[0].barh([k.replace("pre_", "") for k, _ in items], [v for _, v in items], color="#2563eb")
    axes[0].set_xlabel("mean |SHAP| (Wh/km)")
    axes[0].set_title("What the boosted prior uses", fontsize=9)
    names = [n for n in abl if n.startswith("drop_") and n != "drop_speed_weather_elevation"]
    vals = [100 * abl[n]["delta_wape_vs_all"] for n in names]
    order = np.argsort(vals)
    axes[1].barh([names[i].replace("drop_", "") for i in order], [vals[i] for i in order], color="#f59e0b")
    axes[1].set_xlabel("increase in trip WAPE when the group is removed (percentage points)")
    axes[1].set_title("Drop-one feature-group ablation", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig6_importance.png")
    plt.close(fig)


def fig_ved(dyn: dict, fig_dir: Path) -> None:
    """Transfer to the Leafs: remaining-energy error at 2, 5 and 10 km driven, with the in-domain VED reference."""
    reg = "no_history"
    if reg not in dyn or "ved_transfer" not in dyn[reg]:
        return
    methods = ["static", "rate", "blend_ratio", "gbm_dyn", "kf_mult", "kf_carry"]
    cps = [c for c in ("km_2", "km_5", "km_10") if c in dyn[reg]["ved_transfer"]]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        v = [100 * dyn[reg]["ved_transfer"][c][m]["wape"] if m in dyn[reg]["ved_transfer"][c] else np.nan for c in cps]
        ax.bar(np.arange(len(cps)) + i * w, v, w, label=LABEL[m], color=COL[m])
    ref = dyn.get(reg, {}).get("ved_within", {})
    for j, c in enumerate(cps):
        if c in ref and "static" in ref[c]:
            ax.hlines(100 * ref[c]["static"]["wape"], j - 0.05, j + 0.85, color="black", ls=":", lw=1.2,
                      label="prior trained inside VED (reference)" if j == 0 else None)
    ax.set_xticks(np.arange(len(cps)) + w * (len(methods) - 1) / 2)
    ax.set_xticklabels([c.split("_")[1] + " km driven" for c in cps])
    ax.set_ylabel("remaining-energy error, WAPE (%)")
    ax.set_title("Transfer to a different car and region (Tesla-trained, tested on Nissan Leafs)", fontsize=9.5)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig8_ved_transfer.png")
    plt.close(fig)


def fig_examples(cfg: Config, fig_dir: Path) -> list[str]:
    """Two illustrative trips chosen by a neutral rule: per vehicle, the trip closest to the median length among trips >= 30 km."""
    d = cfg.path("interim_dir") / "folds"
    files = sorted(d.glob("full__group_cv__*.long.parquet"))
    long = pl.concat([pl.read_parquet(f) for f in files])
    cand = long.filter(pl.col("method") == "static").group_by("trip_id", "vehicle").agg(pl.col("n_seg").first())
    cand = cand.filter(pl.col("n_seg") >= 30)
    picks = []
    for v in sorted(cand["vehicle"].unique().to_list()):
        c = cand.filter(pl.col("vehicle") == v)
        med = c["n_seg"].median()
        picks.append(c.with_columns((pl.col("n_seg") - med).abs().alias("dist")).sort("dist", "trip_id")["trip_id"][0])
    fig, axes = plt.subplots(1, len(picks), figsize=(4.4 * len(picks), 3.2), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, tid in zip(axes, picks):
        t = long.filter(pl.col("trip_id") == tid)
        s = t.filter(pl.col("method") == "static").sort("km_driven")
        k = t.filter(pl.col("method") == "kf_mult").sort("km_driven")
        ax.plot(s["km_driven"], s["y"], color="black", lw=1.6, label="actual remaining")
        ax.plot(s["km_driven"], s["pred"], color=COL["static"], lw=1.3, label=LABEL["static"])
        ax.plot(k["km_driven"], k["pred"], color=COL["kf_mult"], lw=1.3, label=LABEL["kf_mult"])
        ax.fill_between(k["km_driven"], k["c05"], k["c95"], color=COL["kf_mult"], alpha=0.15, label="90% interval (Kalman)")
        ax.set_title(tid.replace("_drive", " drive "), fontsize=8.5)
        ax.set_xlabel("distance driven (km)")
    axes[0].set_ylabel("remaining energy (kWh)")
    axes[0].legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig7_example_trips.png")
    plt.close(fig)
    return picks


def build_text(cfg: Config, res: Path, picks: list[str]) -> str:
    q = load(res, "quality_report.json")
    ved = load(res, "ved_quality.json") if (res / "ved_quality.json").exists() else None
    pre = load(res, "pretrip_summary.json")
    dyn = load(res, "dynamic_summary.json")
    ho = load(res, "handover.json")
    sub = load(res, "subgroups.json")
    rob = load(res, "robustness.json")
    abl = load(res, "ablation_prior.json")
    imp = load(res, "feature_importance.json")
    phy = load(res, "physics_coefficients.json")
    info = load(res, "fold_info.json")
    L: list[str] = []
    A = L.append

    A("# Results report\n")
    A(f"Generated by `python -m evprior report` from `results/*.json` (config hash `{cfg.run_hash}`). Every number below is read from those files.\n")

    A("## 1. Data\n")
    A(f"Source: KU Leuven BEV energy-dynamics dataset (CC BY 4.0, DOI 10.48804/8KPDTW), 2 Tesla Model Y, 1 Hz CAN-derived signals. "
      f"{q['n_files']} driving-session files; {q['n_kept']} kept after quality filters ({q['km_total_kept']:,.0f} km, {q['kwh_total_kept']:,.0f} kWh, "
      f"median {q['wh_per_km_median']:.0f} Wh/km). Kept per vehicle: " + ", ".join(f"{k} {v}" for k, v in sorted(q["kept_per_vehicle"].items())) + ".\n")
    A("Dropped: " + ", ".join(f"{k} ({v})" for k, v in q["drop_reasons"].items()) + ".\n")
    A(f"Energy label = integral of pack voltage x current. Against the car's own energy counters it has a median ratio of {q['vi_vs_counter_within_5pct'] and q['vi_vs_counter_ratio_median']:.4f} "
      f"and agrees within 5% on {pct(q['vi_vs_counter_within_5pct'])} of trips (trips with |counter energy| > 0.5 kWh). "
      f"{q['counter_negative_energy_trips_all']} sessions show negative counter energy (counter glitches), which is why the counters are used as a check, not as the label. {q['odometer_note']}.\n")

    A("## 2. Pre-trip prediction (the prior)\n")
    A("Trip energy error as weighted absolute percentage error (WAPE = sum|error| / sum actual), with 95% intervals from a trip-level bootstrap. "
      "Features are pre-trip only (route geometry, SRTM elevation, archive weather, start state, calendar, vehicle, and a leakage-safe route-speed history); "
      "a unit test fails the build if any in-trip column enters the prior.\n")
    rows = []
    for k in PRIOR_LABEL:
        row = [PRIOR_LABEL[k]]
        for s in MAIN_SCHEMES:
            e = pre["full"].get(s, {}).get(k)
            row.append(f"{pct(e['wape'])} [{pct(e['wape_ci95'][0])}, {pct(e['wape_ci95'][1])}]" if e else "n/a")
        rows.append(row)
    A(table(["model"] + [SCHEME_LABEL[s] for s in MAIN_SCHEMES], rows))
    A("![prior models](fig1_prior_models.png)\n")
    mk = cfg.eval.regimes[0]["kind"]
    rows = []
    for s in MAIN_SCHEMES:
        e = pre["full"].get(s, {}).get(mk)
        if e and "picp90" in e:
            rows.append([SCHEME_LABEL[s], pct(e["picp80"]), pct(e["picp90"]), pct(e["width90_rel"], 0)])
    A(f"Calibrated trip-level intervals for the main prior (`{mk}`; conformal on out-of-fold residuals). Nominal 80% and 90%:\n")
    A(table(["scheme", "coverage of 80% interval", "coverage of 90% interval", "mean 90% width relative to forecast"], rows))

    A("## 3. Dynamic remaining-energy prediction\n")
    A("At a checkpoint after d km, every method forecasts the energy still to be used on the planned route. Error is WAPE of that remainder. "
      "The three prior-quality regimes show how much online adaptation is worth as the starting estimate gets worse. "
      "`Oracle ceiling` scales the nominal remainder by the trip's true overall consumption ratio. That ratio includes the very segments being forecast, so the oracle is close to tautological: "
      "it shows that most of the error is a single per-trip scale, not that an online method could reach it from the first few kilometres. It is not deployable and is not a target.\n")
    for scheme in ("group_cv", "leave_vehicle_out", "season_out", "forward_chain"):  # external VED schemes have their own section
        A(f"### {SCHEME_LABEL[scheme]}\n")
        for r in REGIME_LABEL:
            if r not in dyn or scheme not in dyn[r]:
                continue
            cps = [c for c in ("km_2", "km_5", "km_10", "km_20") if c in dyn[r][scheme]]
            A(f"**Prior regime: {REGIME_LABEL[r]}**\n")
            rows = []
            for m in LABEL:
                row = [LABEL[m]]
                for c in cps:
                    e = dyn[r][scheme][c].get(m)
                    row.append(f"{pct(e['wape'])} [{pct(e['wape_ci95'][0])}, {pct(e['wape_ci95'][1])}]" if e else "n/a")
                rows.append(row)
            A(table(["method"] + [f"{c.split('_')[1]} km driven (n={dyn[r][scheme][c]['static']['n_rows']})" for c in cps], rows))
    A("![handover](fig2_handover.png)\n")
    if ved is not None:
        A("### External validation: a different car in a different region\n")
        A(f"The Vehicle Energy Dataset (Ann Arbor, Michigan, Apache-2.0) contains three Nissan Leaf BEVs: {ved['n_kept']} usable trips ({ved['km_total_kept']:,.0f} km, median {ved['wh_per_km_median']:.0f} Wh/km, "
          f"versus {q['wh_per_km_median']:.0f} Wh/km for the Teslas). Trip energy is the integral of -V x I (VED reports discharge as negative current) and correlates {ved['corr_energy_vs_soc_drop']:.2f} with the trip's SOC drop; "
          f"the implied pack capacity is {ved['implied_pack_capacity_kwh_median']:.0f} kWh (IQR {ved['implied_pack_capacity_kwh_iqr'][0]:.0f} to {ved['implied_pack_capacity_kwh_iqr'][1]:.0f}), the same order as the 24 kWh nominal pack of a used 2013 Leaf. "
          "That is a plausibility check, not a calibration. VED has no battery temperature (ambient stands in) and its GPS is privacy-processed (distance comes from the speed integral).\n")
        A("**Transfer** trains everything (prior, filter parameters, calibration, learned in-trip model) on the KU Leuven trips only and tests on every Leaf trip; vehicle identity is dropped from the features. "
          "**Kalman + vehicle carry-over** starts each trip from the energy-weighted consumption ratio of the same vehicle's earlier, already-finished trips (30-day half-life, shrunk towards 1), which uses no future information. "
          "'Within VED' is the in-domain reference (grouped CV inside VED).\n")
        for sch in ("ved_transfer", "ved_within"):
            for r in ("no_history", "naive"):
                if r not in dyn or sch not in dyn[r]:
                    continue
                cps = [c for c in ("km_1", "km_2", "km_5", "km_10") if c in dyn[r][sch]]
                A(f"**{SCHEME_LABEL[sch]}, prior regime: {REGIME_LABEL[r]}**\n")
                rows = []
                for m in LABEL:
                    row = [LABEL[m]]
                    for c in cps:
                        e = dyn[r][sch][c].get(m)
                        row.append(f"{pct(e['wape'])} (bias {100 * e['bias_kwh'] / e['mean_y_kwh']:+.0f}%)" if e else "n/a")
                    rows.append(row)
                A(table(["method"] + [f"{c.split('_')[1]} km driven (n={dyn[r][sch][c]['static']['n_rows']})" for c in cps], rows))
        A("![ved transfer](fig8_ved_transfer.png)\n")
    A("![adaptation gain](fig3_adaptation_gain.png)\n")

    A("### Handover points\n")
    A("First distance at which a method's remaining-energy error is lower than the static pre-trip prior with the paired trip-bootstrap 95% interval entirely below zero (\"never\" = not within 40 km).\n")
    rows = []
    for r in REGIME_LABEL:
        for s in ("group_cv", "leave_vehicle_out", "season_out"):
            if r in ho and s in ho[r]:
                h = ho[r][s]["handover_vs_static"]
                rows.append([REGIME_LABEL[r], SCHEME_LABEL[s]] + [(f"{h[m]['first_km_significantly_better']} km" if h[m]["first_km_significantly_better"] else "never") for m in ("kf_mult", "kf_full", "blend_ratio", "gbm_dyn")])
    A(table(["prior regime", "scheme", "Kalman multiplier", "Kalman mult.+offset", "credibility blend (ratio)", "learned in-trip GBM"], rows))

    A("## 4. Uncertainty and decisions\n")
    A("Interval coverage (nominal 90%) and sharpness at 5 km driven, full prior. 'Conformal' uses held-out standardised residuals from training trips; 'Gaussian' uses the model's own variance.\n")
    for s in ("group_cv", "season_out"):
        if "full" not in dyn or s not in dyn["full"]:
            continue
        rows = []
        for m in ("static", "kf_mult", "kf_full", "blend_ratio", "gbm_dyn"):
            e = dyn["full"][s]["km_5"].get(m)
            if e:
                rows.append([LABEL[m], pct(e["picp90_conformal"]), f"{e['width90_conformal_kwh']:.2f}", pct(e["picp90_gaussian"]), f"{e['crps_kwh']:.3f}",
                             f"{e['reserve95_kwh']:.2f}", pct(e["violation_fixed_reserve"])])
        A(f"**{SCHEME_LABEL[s]}**\n")
        A(table(["method", "90% coverage (conformal)", "mean 90% width (kWh)", "90% coverage (Gaussian)", "CRPS (kWh)", "reserve for 95% assurance (kWh)",
                 f"arrive-below-reserve rate at fixed {cfg.eval.reserve_kwh:g} kWh reserve"], rows))
    A("![calibration](fig4_calibration.png)\n")

    A("## 5. Where the error comes from\n")
    A("WAPE at 5 km driven by condition (grouped CV, full prior).\n")
    if "full" in sub and "km_5" in sub["full"]:
        rows = []
        for col, groups in sub["full"]["km_5"].items():
            for g, ms in groups.items():
                if "static" in ms and "kf_mult" in ms:
                    rows.append([col, g, ms["static"]["n"], pct(ms["static"]["wape"]), pct(ms["blend_ratio"]["wape"]) if "blend_ratio" in ms else "n/a", pct(ms["kf_mult"]["wape"]), pct(ms["kf_mult"]["picp90"])])
        A(table(["factor", "level", "n checkpoints", "prior", "credibility blend", "Kalman multiplier", "Kalman 90% coverage"], rows))

    A("## 6. Robustness to degraded observations\n")
    A("Full prior, WAPE at 10 km driven. Dropout hides random segments from the online methods, noise multiplies observed segment energy by 1 + N(0, level), delay means only segments up to k - level are available at checkpoint k. "
      "Grouped CV is the in-distribution case where the online update changes little; season-out is the shifted case where it matters, so it is the more informative test of robustness. "
      "The static prior does not use observations, so its row is the reference every online method should be compared to.\n")
    if "full" in rob:
        for s in ("group_cv", "season_out"):
            if s not in rob["full"]:
                continue
            rows = []
            for kind in ("dropout", "noise", "delay"):
                for lvl in sorted(rob["full"][s].get(kind, {}), key=float):
                    e = rob["full"][s][kind][lvl]["km_10"]
                    rows.append([kind, lvl] + [pct(e[m]["wape"]) if m in e else "n/a" for m in ("static", "rate", "blend_ratio", "kf_mult", "kf_full")] + [pct(e["kf_mult"]["picp90_conformal"]) if "kf_mult" in e else "n/a"])
            A(f"**{SCHEME_LABEL[s]}**\n")
            A(table(["perturbation", "level", "prior (no update)", "rate extrapolation", "credibility blend (ratio)", "Kalman multiplier", "Kalman mult.+offset", "Kalman 90% coverage"], rows))
    A("![robustness](fig5_robustness.png)\n")

    A("## 7. Interpretability\n")
    A("Drop-one ablation of the pre-trip prior (grouped CV; trip WAPE and change in percentage points):\n")
    rows = [[n, ", ".join(v["dropped"]) or "none", pct(v["wape"]), f"{100 * v['delta_wape_vs_all']:+.2f}"] for n, v in abl.items()]
    A(table(["variant", "removed", "trip WAPE", "change (pp)"], rows))
    A("![importance](fig6_importance.png)\n")
    A("Road-load coefficients (physics regression, trip-bootstrap 95% intervals):\n")
    rows = [[k, f"{v['coef']:.4f}", f"[{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]"] for k, v in phy.items() if not k.startswith("_")]
    A(table(["term", "coefficient (kWh per unit)", "95% interval"], rows))
    s = phy["_sanity"]
    A(f"Plausibility check: the climb coefficient implies a vehicle mass of about {s['implied_mass_kg_at_eff_0.9']:.0f} kg at 0.9 drivetrain efficiency, and the descent coefficient recovers "
      f"{pct(s['regen_fraction_of_climb_energy'], 0)} of the climb energy. A Model Y weighs roughly 2,000 kg and regeneration cannot exceed 100%, so the elevation terms are attenuated by the coarse SRTM resolution "
      "(and are confounded with route type). Treat the elevation coefficients as predictive, not as physical constants.\n")

    A("## 8. Example trips\n")
    A(f"Chosen by a fixed rule (per vehicle, the trip closest to the median length among trips of at least 30 segments), not by outcome: {', '.join(picks)}.\n")
    A("![examples](fig7_example_trips.png)\n")

    A("## 9. Limitations\n")
    for t in (
        "Only two vehicles of one model in one region (Belgium and neighbours); leave-one-vehicle-out has two folds and no power to separate vehicle effects from chemistry (LFP versus NCA).",
        "The dataset has no planned route. The realised GPS track is used as the planned route, which is optimistic for the pre-trip stage.",
        "Weather comes from the Open-Meteo archive (reanalysis), not from forecasts, which is optimistic for the pre-trip stage.",
        "The route-speed history proxy uses other training trips in the same ~500 m grid cell (leave-own-trip-out). It encodes habitual routes; it is absent for new routes, which the 'no route-speed history' regime approximates.",
        "SRTM 30 m elevation sampled every 200 m attenuates grade; the fitted elevation coefficients are not physically calibrated (section 7).",
        "Auxiliary load (HVAC) is not observed directly; it is inferred through temperature, and the additive-offset state only partly captures it.",
        "Conformal coverage is marginal under exchangeability; coverage under shift is measured and reported, not guaranteed.",
        "No internal fleet data was used; conclusions about scale or other vehicle platforms are not supported by this study.",
    ):
        A(f"- {t}")
    A("")
    A("## 10. Reproducibility\n")
    A(f"Python {platform.python_version()}, config hash `{cfg.run_hash}`, {len(info)} fold fits (3 prior regimes x 18 splits). Re-run: `python scripts/get_data.py && python -m evprior all`.")
    return "\n".join(L)


def headline(res: Path) -> str:
    dyn, pre = load(res, "dynamic_summary.json"), load(res, "pretrip_summary.json")
    best = min((e["wape"], PRIOR_LABEL[k]) for k, e in pre["full"]["group_cv"].items())
    out = [f"Pre-trip trip-energy error (grouped CV, full prior): best model **{best[1]}** at **{pct(best[0])}** WAPE; "
           f"mean kWh/km reference {pct(pre['full']['group_cv']['mean_intensity']['wape'])}.\n",
           "Remaining-energy error at 10 km driven, pre-trip prior alone versus the Kalman multiplier "
           f"(relative change in brackets; negative = online update helps):\n"]
    rows = []
    for r in REGIME_LABEL:
        for sc in ("group_cv", "leave_vehicle_out", "season_out"):
            if r in dyn and sc in dyn[r] and "km_10" in dyn[r][sc]:
                a, b = dyn[r][sc]["km_10"]["static"]["wape"], dyn[r][sc]["km_10"]["kf_mult"]["wape"]
                g = dyn[r][sc]["km_10"]["gbm_dyn"]["wape"]
                o = dyn[r][sc]["km_10"]["oracle_mult"]["wape"]
                rows.append([REGIME_LABEL[r], SCHEME_LABEL[sc], pct(a), f"{pct(b)} ({rel(b, a)})", f"{pct(g)} ({rel(g, a)})", pct(o)])
    out.append(table(["prior regime", "evaluation", "prior alone", "Kalman multiplier", "learned in-trip GBM", "oracle ceiling"], rows))
    if "no_history" in dyn and "ved_transfer" in dyn["no_history"] and "km_5" in dyn["no_history"]["ved_transfer"]:
        e = dyn["no_history"]["ved_transfer"]["km_5"]
        out.append(f"External check on a different car and region (train on the Teslas, test on Nissan Leafs, 5 km driven): prior alone {pct(e['static']['wape'])}, "
                   f"Kalman multiplier {pct(e['kf_mult']['wape'])}, Kalman with vehicle carry-over **{pct(e['kf_carry']['wape'])}**.{chr(10)}")
    out.append("Full tables, intervals, robustness and limitations: [results/REPORT.md](results/REPORT.md).\n")
    return "\n".join(out)


def update_readme(cfg: Config, res: Path) -> None:
    readme = cfg.root / "README.md"
    if not readme.exists():
        return
    txt = readme.read_text(encoding="utf-8")
    block = "<!-- RESULTS:START -->\n" + headline(res) + "\n<!-- RESULTS:END -->"
    new = re.sub(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->", lambda _: block, txt, flags=re.S)
    readme.write_text(new, encoding="utf-8")


def run(cfg: Config) -> None:
    res = cfg.path("results_dir")
    fig_dir = res
    pre, dyn, ho = load(res, "pretrip_summary.json"), load(res, "dynamic_summary.json"), load(res, "handover.json")
    fig_prior(pre, fig_dir)
    fig_handover(ho, fig_dir)
    fig_shift(dyn, fig_dir)
    fig_calibration(dyn, fig_dir)
    fig_robust(load(res, "robustness.json"), fig_dir)
    fig_ved(dyn, fig_dir)
    fig_importance(load(res, "feature_importance.json"), load(res, "ablation_prior.json"), fig_dir)
    picks = fig_examples(cfg, fig_dir)
    (res / "REPORT.md").write_text(build_text(cfg, res, picks), encoding="utf-8")
    update_readme(cfg, res)
    log.info("report written to %s", res / "REPORT.md")
