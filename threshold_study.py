
from pathlib import Path
import hashlib
import json
import platform
import sys
import time

import joblib
import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from scipy.optimize import brentq
from scipy.stats import binomtest, chi2_contingency, norm, rankdata
from sklearn.metrics import (
    confusion_matrix,
    det_curve,
    roc_auc_score,
    roc_curve,
)

from features import extract_lfcc_from_file


# --------------------------------------------------
# 1. PROJECT PATHS AND SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "lfcc_rbf_svm.joblib"

ARTIFACT_DIR = ROOT / "artifacts"
OUTPUT_DIR = ARTIFACT_DIR / "threshold_study"

DEV_FEATURES_PATH = ARTIFACT_DIR / "dev_features.npz"

DATA_DIR = ROOT / "data" / "ASVspoof2019_LA" / "LA"

DEV_AUDIO_DIR = DATA_DIR / "ASVspoof2019_LA_dev" / "flac"

CM_PROTOCOL = (
    DATA_DIR
    / "ASVspoof2019_LA_cm_protocols"
    / "ASVspoof2019.LA.cm.dev.trl.txt"
)

ASV_DIR = DATA_DIR / "ASVspoof2019_LA_asv_protocols"

ASV_PROTOCOL = ASV_DIR / "ASVspoof2019.LA.asv.dev.gi.trl.txt"

ASV_SCORES = (
    DATA_DIR
    / "ASVspoof2019_LA_asv_scores"
    / "ASVspoof2019.LA.asv.dev.gi.trl.scores.txt"
)

N_BOOTSTRAP = 5000
SEED = 2019
PLATEAU_TOLERANCE = 0.001  # 0.1 percentage points of ACER

ATTACKS = ["A01", "A02", "A03", "A04", "A05", "A06"]

# Table 2, logical access, ASVspoof 2019 evaluation plan v0.4
COST_MODEL = {
    "Ptar": 0.9405,
    "Pnon": 0.0095,
    "Pspoof": 0.05,
    "Cmiss_asv": 1,
    "Cfa_asv": 10,
    "Cmiss_cm": 1,
    "Cfa_cm": 10,
}

# Baseline figures reported before this study, in percent
REPORTED_DEFAULT = {
    "accuracy": 96.19,
    "APCER": 2.45,
    "BPCER": 15.66,
    "ACER": 9.06,
}

REPORTED_0488 = {
    "accuracy": 92.33,
    "APCER": 7.81,
    "BPCER": 6.44,
    "ACER": 7.12,
}

# Thresholds chosen from data; the default is only a baseline and
# min_tDCF optimises a different objective
SELECTABLE = ["EER", "min_ACER", "plateau_midpoint"]

CANDIDATES = ["default", *SELECTABLE, "min_tDCF"]

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"

CANDIDATE_MARKERS = {
    "default": "o",
    "EER": "s",
    "min_ACER": "^",
    "plateau_midpoint": "D",
    "min_tDCF": "v",
}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 10,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.6,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "legend.frameon": False,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": SURFACE,
})


# --------------------------------------------------
# 2. VERIFICATION HELPERS
# --------------------------------------------------

CHECKS = []


def check(name, passed, detail=""):

    CHECKS.append({
        "check": name,
        "result": "PASS" if passed else "FAIL",
        "detail": detail,
    })

    print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}")


def sha256(path):

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected_files():
    """Existing files this study must leave byte-identical."""

    scripts = [
        ROOT / name
        for name in (
            "features.py",
            "train.py",
            "evaluate.py",
            "threshold_analysis.py",
            "test_features.py",
        )
    ]

    artifacts = [
        path for path in ARTIFACT_DIR.iterdir() if path.is_file()
    ]

    return [MODEL_PATH, *scripts, *artifacts]


# --------------------------------------------------
# 3. LOAD DEVELOPMENT DATA
# --------------------------------------------------

def load_cm_protocol(y):
    """
    Read the dev protocol and confirm its layout on the file itself.

    Layout: SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY
    The README's numbered list swaps columns 3 and 4, so the layout
    is enforced here rather than assumed.
    """

    protocol = pd.read_csv(
        CM_PROTOCOL,
        sep=r"\s+",
        header=None,
        names=["speaker", "file_id", "unused", "system", "key"],
        dtype=str,
        keep_default_na=False,
    )

    check(
        "Protocol column 3 is blank in every row",
        (protocol["unused"] == "-").all(),
    )

    check(
        "Protocol column 4 holds only '-' and A01-A06",
        set(protocol["system"]) == {"-", *ATTACKS},
    )

    check(
        "Bona fide rows have no attack ID and spoof rows do",
        ((protocol["key"] == "bonafide") == (protocol["system"] == "-")).all(),
    )

    labels = (protocol["key"] == "spoof").astype(int).to_numpy()

    check(
        "Cached labels match protocol order",
        len(labels) == len(y) and np.array_equal(labels, y),
        f"rows={len(labels)}",
    )

    return protocol


def load_asv():
    """Return the fixed ASV errors for t-DCF, or None if unavailable."""

    if not (ASV_SCORES.exists() and ASV_PROTOCOL.exists()):
        return None

    scores = pd.read_csv(
        ASV_SCORES,
        sep=r"\s+",
        header=None,
        names=["cm_key", "asv_key", "score"],
        dtype={"cm_key": str, "asv_key": str, "score": float},
    )

    trials = pd.read_csv(
        ASV_PROTOCOL,
        sep=r"\s+",
        header=None,
        names=["claimed", "file_id", "cm_key", "asv_key"],
        dtype=str,
    )

    check(
        "ASV scores line up with the ASV trial protocol",
        len(scores) == len(trials)
        and (scores["cm_key"].to_numpy() == trials["cm_key"].to_numpy()).all()
        and (scores["asv_key"].to_numpy() == trials["asv_key"].to_numpy()).all(),
        f"trials={len(scores)}",
    )

    target = scores.loc[scores["asv_key"] == "target", "score"].to_numpy()
    nontarget = scores.loc[scores["asv_key"] == "nontarget", "score"].to_numpy()
    spoof = scores.loc[scores["asv_key"] == "spoof", "score"].to_numpy()

    if min(len(target), len(nontarget), len(spoof)) == 0:
        return None

    # ASV threshold fixed at its own EER point, as the plan specifies
    frr, far, thresholds = official_det_curve(target, nontarget)

    index = np.argmin(np.abs(frr - far))

    threshold = thresholds[index]

    pmiss = np.mean(target < threshold)
    pfa = np.mean(nontarget >= threshold)
    pmiss_spoof = np.mean(spoof < threshold)

    costs = COST_MODEL

    c1 = (
        costs["Ptar"] * (costs["Cmiss_cm"] - costs["Cmiss_asv"] * pmiss)
        - costs["Pnon"] * costs["Cfa_asv"] * pfa
    )

    c2 = costs["Cfa_cm"] * costs["Pspoof"] * (1 - pmiss_spoof)

    check(
        "t-DCF priors sum to one and C1, C2 are positive",
        abs(costs["Ptar"] + costs["Pnon"] + costs["Pspoof"] - 1) < 1e-12
        and c1 > 0
        and c2 > 0,
        f"C1={c1:.5f} C2={c2:.5f}",
    )

    return {
        "n_target": len(target),
        "n_nontarget": len(nontarget),
        "n_spoof": len(spoof),
        "threshold": float(threshold),
        "eer": float((frr[index] + far[index]) / 2),
        "Pmiss_asv": float(pmiss),
        "Pfa_asv": float(pfa),
        "Pmiss_spoof_asv": float(pmiss_spoof),
        "C1": float(c1),
        "C2": float(c2),
        "spoof_by_attack": (
            scores[scores["asv_key"] == "spoof"]
            .groupby("cm_key")
            .size()
            .to_dict()
        ),
    }


# --------------------------------------------------
# 4. CORE CALCULATIONS
# --------------------------------------------------

# Decision rule used throughout, matching threshold_analysis.py:
# classify spoof when score >= threshold

def error_curve(scores, y):
    """Exact APCER and BPCER at every score, plus one point above the top."""

    spoof = np.sort(scores[y == 1])
    bonafide = np.sort(scores[y == 0])

    thresholds = np.unique(scores)
    thresholds = np.append(
        thresholds,
        np.nextafter(thresholds[-1], np.inf),
    )

    apcer = np.searchsorted(spoof, thresholds, side="left") / len(spoof)

    bpcer = 1 - np.searchsorted(bonafide, thresholds, side="left") / len(bonafide)

    return thresholds, apcer, bpcer


def rates_at(scores, y, threshold):

    spoof = scores[y == 1]
    bonafide = scores[y == 0]

    apcer = np.mean(spoof < threshold)
    bpcer = np.mean(bonafide >= threshold)

    accuracy = (
        np.sum(spoof >= threshold) + np.sum(bonafide < threshold)
    ) / len(scores)

    return apcer, bpcer, (apcer + bpcer) / 2, accuracy


def tdcf_norm(apcer, bpcer, asv):
    """Normalized t-DCF: CM miss is BPCER and CM false alarm is APCER."""

    if asv is None:
        return np.full(np.shape(apcer), np.nan)

    return (
        asv["C1"] * np.asarray(bpcer) + asv["C2"] * np.asarray(apcer)
    ) / min(asv["C1"], asv["C2"])


def select_thresholds(scores, y, asv):
    """Apply every threshold-selection rule to one set of scores."""

    thresholds, apcer, bpcer = error_curve(scores, y)

    acer = (apcer + bpcer) / 2

    eer_index = np.argmin(np.abs(apcer - bpcer))
    acer_index = np.argmin(acer)

    plateau = thresholds[acer <= acer[acer_index] + PLATEAU_TOLERANCE]

    selected = {
        "default": 0.0,
        "EER": thresholds[eer_index],
        "min_ACER": thresholds[acer_index],
        "plateau_midpoint": (plateau.min() + plateau.max()) / 2,
    }

    summary = {
        "EER": (apcer[eer_index] + bpcer[eer_index]) / 2,
        "min_ACER": acer[acer_index],
    }

    if asv is None:
        selected["min_tDCF"] = np.nan
        summary["min_tDCF"] = np.nan
    else:
        tdcf = tdcf_norm(apcer, bpcer, asv)
        tdcf_index = np.argmin(tdcf)
        selected["min_tDCF"] = thresholds[tdcf_index]
        summary["min_tDCF"] = tdcf[tdcf_index]

    return selected, summary


def fast_auc(scores, y):

    ranks = rankdata(scores)

    n_spoof = np.sum(y == 1)
    n_bonafide = np.sum(y == 0)

    return (
        ranks[y == 1].sum() - n_spoof * (n_spoof + 1) / 2
    ) / (n_spoof * n_bonafide)


def clopper_pearson(errors, total):

    interval = binomtest(int(errors), int(total)).proportion_ci(
        confidence_level=0.95,
        method="exact",
    )

    return interval.low, interval.high


def official_det_curve(target_scores, nontarget_scores):
    """Curve as written in the ASVspoof 2019 t-DCF scoring code."""

    n_scores = target_scores.size + nontarget_scores.size

    all_scores = np.concatenate((target_scores, nontarget_scores))

    labels = np.concatenate((
        np.ones(target_scores.size),
        np.zeros(nontarget_scores.size),
    ))

    indices = np.argsort(all_scores, kind="mergesort")

    labels = labels[indices]

    tar_trial_sums = np.cumsum(labels)

    nontarget_trial_sums = (
        nontarget_scores.size
        - (np.arange(1, n_scores + 1) - tar_trial_sums)
    )

    frr = np.concatenate((
        np.atleast_1d(0),
        tar_trial_sums / target_scores.size,
    ))

    far = np.concatenate((
        np.atleast_1d(1),
        nontarget_trial_sums / nontarget_scores.size,
    ))

    thresholds = np.concatenate((
        np.atleast_1d(all_scores[indices[0]] - 0.001),
        all_scores[indices],
    ))

    return frr, far, thresholds


# --------------------------------------------------
# 5. BOOTSTRAP AND HELD-OUT SPEAKER ANALYSIS
# --------------------------------------------------

def run_speaker_bootstrap(scores, y, systems, speakers, asv, fixed, grid, rng):
    """
    Resample whole speakers, separately for speakers with and without
    spoof files so every replicate contains both classes.
    """

    spoof_speakers = np.unique(speakers[y == 1])
    other_speakers = np.setdiff1d(np.unique(speakers), spoof_speakers)

    rows_of = {
        speaker: np.where(speakers == speaker)[0]
        for speaker in np.unique(speakers)
    }

    n_candidates = len(CANDIDATES)

    selected = np.full((N_BOOTSTRAP, n_candidates), np.nan)
    summary = np.full((N_BOOTSTRAP, 4), np.nan)
    at_fixed = np.full((N_BOOTSTRAP, n_candidates, 5), np.nan)
    attack_apcer = np.full((N_BOOTSTRAP, n_candidates, len(ATTACKS)), np.nan)
    grid_acer = np.full((N_BOOTSTRAP, len(grid)), np.nan)
    out_of_bag = np.full((N_BOOTSTRAP, n_candidates, 3), np.nan)

    start_time = time.time()

    for replicate in range(N_BOOTSTRAP):

        drawn = np.concatenate([
            rng.choice(spoof_speakers, len(spoof_speakers)),
            rng.choice(other_speakers, len(other_speakers)),
        ])

        rows = np.concatenate([rows_of[speaker] for speaker in drawn])

        s_r, y_r, system_r = scores[rows], y[rows], systems[rows]

        picked, stats = select_thresholds(s_r, y_r, asv)

        selected[replicate] = [picked[name] for name in CANDIDATES]

        summary[replicate] = [
            stats["EER"],
            stats["min_ACER"],
            stats["min_tDCF"],
            fast_auc(s_r, y_r),
        ]

        for c, name in enumerate(CANDIDATES):

            if np.isnan(fixed[name]):
                continue

            apcer, bpcer, acer, accuracy = rates_at(s_r, y_r, fixed[name])

            at_fixed[replicate, c] = [
                apcer,
                bpcer,
                acer,
                accuracy,
                tdcf_norm(apcer, bpcer, asv),
            ]

            for a, attack in enumerate(ATTACKS):
                attack_apcer[replicate, c, a] = np.mean(
                    s_r[system_r == attack] < fixed[name]
                )

        spoof_sorted = np.sort(s_r[y_r == 1])
        bonafide_sorted = np.sort(s_r[y_r == 0])

        grid_acer[replicate] = (
            np.searchsorted(spoof_sorted, grid, side="left") / len(spoof_sorted)
            + 1
            - np.searchsorted(bonafide_sorted, grid, side="left") / len(bonafide_sorted)
        ) / 2

        # Speakers not drawn test the thresholds chosen without them
        held_out = np.setdiff1d(np.unique(speakers), drawn)
        oob_mask = np.isin(speakers, held_out)

        if (y[oob_mask] == 1).any() and (y[oob_mask] == 0).any():

            for c, name in enumerate(CANDIDATES):

                if np.isnan(picked[name]):
                    continue

                apcer, bpcer, acer, _ = rates_at(
                    scores[oob_mask],
                    y[oob_mask],
                    picked[name],
                )

                out_of_bag[replicate, c] = [apcer, bpcer, acer]

        if (replicate + 1) % 500 == 0:
            print(
                f"Speaker bootstrap {replicate + 1}/{N_BOOTSTRAP} | "
                f"Time: {time.time() - start_time:.1f}s"
            )

    return {
        "selected": selected,
        "summary": summary,
        "at_fixed": at_fixed,
        "attack_apcer": attack_apcer,
        "grid_acer": grid_acer,
        "out_of_bag": out_of_bag,
    }


def run_file_bootstrap(scores, y, asv, fixed, rng):
    """Resample individual files within each class, ignoring speakers."""

    bonafide_rows = np.where(y == 0)[0]
    spoof_rows = np.where(y == 1)[0]

    n_candidates = len(CANDIDATES)

    selected = np.full((N_BOOTSTRAP, n_candidates), np.nan)
    summary = np.full((N_BOOTSTRAP, 4), np.nan)
    at_fixed = np.full((N_BOOTSTRAP, n_candidates, 5), np.nan)

    start_time = time.time()

    for replicate in range(N_BOOTSTRAP):

        rows = np.concatenate([
            rng.choice(bonafide_rows, len(bonafide_rows)),
            rng.choice(spoof_rows, len(spoof_rows)),
        ])

        s_r, y_r = scores[rows], y[rows]

        picked, stats = select_thresholds(s_r, y_r, asv)

        selected[replicate] = [picked[name] for name in CANDIDATES]

        summary[replicate] = [
            stats["EER"],
            stats["min_ACER"],
            stats["min_tDCF"],
            fast_auc(s_r, y_r),
        ]

        for c, name in enumerate(CANDIDATES):

            if np.isnan(fixed[name]):
                continue

            apcer, bpcer, acer, accuracy = rates_at(s_r, y_r, fixed[name])

            at_fixed[replicate, c] = [
                apcer,
                bpcer,
                acer,
                accuracy,
                tdcf_norm(apcer, bpcer, asv),
            ]

        if (replicate + 1) % 1000 == 0:
            print(
                f"File bootstrap {replicate + 1}/{N_BOOTSTRAP} | "
                f"Time: {time.time() - start_time:.1f}s"
            )

    return {
        "selected": selected,
        "summary": summary,
        "at_fixed": at_fixed,
    }


def leave_one_speaker_out(scores, y, speakers, asv):
    """Choose thresholds without each speaker, then score that speaker."""

    rows = []

    for speaker in np.unique(speakers):

        held_out = speakers == speaker

        picked, _ = select_thresholds(scores[~held_out], y[~held_out], asv)

        for name in CANDIDATES:

            if np.isnan(picked[name]):
                continue

            spoof = scores[held_out & (y == 1)]
            bonafide = scores[held_out & (y == 0)]

            rows.append({
                "speaker": speaker,
                "candidate": name,
                "threshold": picked[name],
                "n_bonafide": len(bonafide),
                "n_spoof": len(spoof),
                "bpcer_errors": int(np.sum(bonafide >= picked[name])),
                "apcer_errors": int(np.sum(spoof < picked[name])),
            })

    per_speaker = pd.DataFrame(rows)

    pooled = per_speaker.groupby("candidate", sort=False).agg(
        threshold_sd=("threshold", "std"),
        n_bonafide=("n_bonafide", "sum"),
        n_spoof=("n_spoof", "sum"),
        bpcer_errors=("bpcer_errors", "sum"),
        apcer_errors=("apcer_errors", "sum"),
    ).reset_index()

    pooled["APCER"] = pooled["apcer_errors"] / pooled["n_spoof"]
    pooled["BPCER"] = pooled["bpcer_errors"] / pooled["n_bonafide"]
    pooled["ACER"] = (pooled["APCER"] + pooled["BPCER"]) / 2

    return per_speaker, pooled


def percentile_ci(values):

    return np.nanpercentile(values, [2.5, 97.5], axis=0)


# --------------------------------------------------
# 6. PLOTS
# --------------------------------------------------

def mark_thresholds(ax, marks, top=0.97, step=0.075, ha="left"):
    """Thin labelled verticals, staggered so close labels stay readable."""

    transform = ax.get_xaxis_transform()

    for i, (name, value) in enumerate(sorted(marks.items(), key=lambda m: m[1])):

        ax.axvline(value, color=INK_SECONDARY, linewidth=0.8, zorder=1)

        ax.text(
            value,
            top - i * step,
            f" {name} {value:.3f} ",
            transform=transform,
            color=INK_SECONDARY,
            fontsize=8,
            va="top",
            ha=ha,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.8},
        )


def plot_candidate_points(ax, points):

    for name, (x, y) in points.items():

        ax.plot(
            x,
            y,
            CANDIDATE_MARKERS[name],
            markerfacecolor="none",
            markeredgecolor=INK,
            markersize=9,
            markeredgewidth=1.2,
            linestyle="none",
            label=name,
            zorder=3,
        )


def plot_score_distributions(scores, y, systems, marks):

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8))

    bins = np.linspace(scores.min(), scores.max(), 90)

    left.hist(
        scores[y == 0],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.6,
        color=BLUE,
        label=f"Bona fide (n={np.sum(y == 0):,})",
    )

    left.hist(
        scores[y == 1],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.6,
        color=ORANGE,
        label=f"Spoof (n={np.sum(y == 1):,})",
    )

    mark_thresholds(left, marks, ha="right")

    left.set_xlabel("SVM decision score (higher means more spoof-like)")
    left.set_ylabel("Density")
    left.set_title("Scores by class, each class normalised", loc="left")
    left.legend(loc="center right")

    groups = ["Bona fide", *ATTACKS]

    for i, name in enumerate(groups):

        values = (
            scores[y == 0]
            if name == "Bona fide"
            else scores[systems == name]
        )

        low, q1, median, q3, high = np.percentile(values, [5, 25, 50, 75, 95])

        color = BLUE if name == "Bona fide" else ORANGE

        right.plot([low, high], [i, i], color=color, linewidth=1.2)
        right.plot([q1, q3], [i, i], color=color, linewidth=4, solid_capstyle="round")
        right.plot(
            median,
            i,
            "o",
            color=SURFACE,
            markeredgecolor=color,
            markersize=5,
        )

    right.set_yticks(range(len(groups)), groups)
    right.invert_yaxis()
    right.grid(False, axis="y")

    for value in marks.values():
        right.axvline(value, color=INK_SECONDARY, linewidth=0.8, zorder=1)

    right.set_xlabel("SVM decision score")
    right.set_title(
        "Median, middle 50% and 5th-95th percentile, with the same thresholds",
        loc="left",
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "score_distributions.png")
    plt.close(fig)


def plot_roc(apcer, bpcer, auc, points):

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 5))

    for ax in (left, right):
        ax.plot(bpcer * 100, (1 - apcer) * 100, color=BLUE, label=f"LFCC + RBF-SVM (AUC {auc:.4f})")
        ax.set_xlabel("BPCER: bona fide flagged as spoof (%)")
        ax.set_ylabel("Spoof detected, 1 - APCER (%)")

    left.plot([0, 100], [0, 100], color=AXIS, linewidth=1, label="Chance")
    left.set_xlim(0, 100)
    left.set_ylim(0, 100)
    left.set_title("Full range", loc="left")
    left.legend(loc="lower right")

    plot_candidate_points(
        right,
        {
            name: (p[1] * 100, (1 - p[0]) * 100)
            for name, p in points.items()
        },
    )

    right.set_xlim(0, 30)
    right.set_ylim(70, 100)
    right.set_title("Zoomed to the operating region", loc="left")
    right.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc.png")
    plt.close(fig)


def plot_det(y, scores, points, eer):

    false_positive, false_negative, _ = det_curve(y, scores)

    ticks = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 40, 60])
    tick_positions = norm.ppf(ticks / 100)

    keep = (false_positive > 0) & (false_negative > 0)

    fig, ax = plt.subplots(figsize=(6.6, 6.2))

    ax.plot(
        norm.ppf(false_positive[keep]),
        norm.ppf(false_negative[keep]),
        color=BLUE,
        label="LFCC + RBF-SVM",
    )

    ax.plot(
        tick_positions[[0, -1]],
        tick_positions[[0, -1]],
        color=AXIS,
        linewidth=1,
        label=f"APCER = BPCER (EER {eer * 100:.2f}%)",
    )

    plot_candidate_points(
        ax,
        {
            name: (norm.ppf(p[1]), norm.ppf(p[0]))
            for name, p in points.items()
        },
    )

    ax.set_xticks(tick_positions, [f"{t:g}" for t in ticks])
    ax.set_yticks(tick_positions, [f"{t:g}" for t in ticks])
    ax.set_xlim(tick_positions[0], tick_positions[-1])
    ax.set_ylim(tick_positions[0], tick_positions[-1])
    ax.set_xlabel("BPCER (%), normal deviate scale")
    ax.set_ylabel("APCER (%), normal deviate scale")
    ax.set_title("DET curve, development set", loc="left")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "det.png")
    plt.close(fig)


def plot_apcer_bpcer(thresholds, apcer, bpcer, marks):

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8))

    for ax in (left, right):
        ax.plot(thresholds, apcer * 100, color=BLUE, label="APCER")
        ax.plot(thresholds, bpcer * 100, color=ORANGE, label="BPCER")
        ax.set_xlabel("SVM decision threshold")
        ax.set_ylabel("Error rate (%)")

    left.set_title("Full score range", loc="left")
    left.legend(loc="center right")

    zoom = (thresholds >= -0.25) & (thresholds <= 0.9)

    right.set_xlim(-0.25, 0.9)
    right.set_ylim(0, max(apcer[zoom].max(), bpcer[zoom].max()) * 105)
    right.set_title("Zoomed to the candidate thresholds", loc="left")
    right.legend(loc="center right")

    mark_thresholds(right, marks)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "apcer_bpcer.png")
    plt.close(fig)


def plot_acer(thresholds, acer, grid, grid_acer, marks):

    low, high = percentile_ci(grid_acer)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.fill_between(
        grid,
        low * 100,
        high * 100,
        color=AQUA,
        alpha=0.18,
        linewidth=0,
        label="95% speaker-bootstrap interval",
    )

    zoom = (thresholds >= grid[0]) & (thresholds <= grid[-1])

    ax.plot(
        thresholds[zoom],
        acer[zoom] * 100,
        color=AQUA,
        label="ACER, full development set",
    )

    mark_thresholds(ax, marks)

    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(0, high.max() * 100)
    ax.set_xlabel("SVM decision threshold")
    ax.set_ylabel("ACER (%)")
    ax.set_title("ACER against threshold", loc="left")
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "acer.png")
    plt.close(fig)


def plot_threshold_stability(speaker_boot, file_boot, point):

    names = SELECTABLE

    fig, axes = plt.subplots(
        len(names),
        1,
        figsize=(9, 7),
        sharex=True,
    )

    speaker_values = speaker_boot["selected"]
    file_values = file_boot["selected"]

    all_values = np.concatenate([
        speaker_values[:, [CANDIDATES.index(n) for n in names]].ravel(),
        file_values[:, [CANDIDATES.index(n) for n in names]].ravel(),
    ])

    bins = np.linspace(*np.percentile(all_values, [0.1, 99.9]), 70)

    for ax, name in zip(axes, names):

        c = CANDIDATES.index(name)

        ax.hist(
            speaker_values[:, c],
            bins=bins,
            histtype="step",
            linewidth=1.5,
            color=BLUE,
            label="Resampling speakers",
        )

        ax.hist(
            file_values[:, c],
            bins=bins,
            histtype="step",
            linewidth=1.5,
            color=ORANGE,
            label="Resampling files",
        )

        ax.axvline(point[name], color=INK_SECONDARY, linewidth=0.8)

        ax.set_ylabel("Replicates")
        ax.set_title(
            f"{name}: full-dev value {point[name]:.3f}, "
            f"speaker-bootstrap SD {np.nanstd(speaker_values[:, c]):.3f}",
            loc="left",
        )

    axes[0].legend(loc="upper left")
    axes[-1].set_xlabel("Threshold chosen in each bootstrap replicate")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "threshold_stability.png")
    plt.close(fig)


def plot_attack_apcer(attack_table, chosen):

    subset = attack_table[attack_table["candidate"] == chosen]

    default = attack_table[attack_table["candidate"] == "default"]

    fig, ax = plt.subplots(figsize=(8, 4.6))

    positions = np.arange(len(ATTACKS))

    ax.errorbar(
        subset["APCER"] * 100,
        positions - 0.13,
        xerr=[
            (subset["APCER"] - subset["speaker_ci_low"]) * 100,
            (subset["speaker_ci_high"] - subset["APCER"]) * 100,
        ],
        fmt="o",
        color=BLUE,
        markersize=6,
        elinewidth=1.2,
        label=f"{chosen} threshold",
    )

    ax.errorbar(
        default["APCER"] * 100,
        positions + 0.13,
        xerr=[
            (default["APCER"] - default["speaker_ci_low"]) * 100,
            (default["speaker_ci_high"] - default["APCER"]) * 100,
        ],
        fmt="s",
        color=ORANGE,
        markersize=6,
        elinewidth=1.2,
        label="Default threshold (0)",
    )

    ax.set_yticks(positions, ATTACKS)
    ax.invert_yaxis()
    ax.grid(False, axis="y")
    ax.set_xlim(left=0)
    ax.set_xlabel("APCER (%), with 95% speaker-bootstrap interval")
    ax.set_title("Spoof files missed, by attack", loc="left")
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "attack_apcer.png")
    plt.close(fig)


def plot_speakers(speaker_table, chosen):

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 6))

    for ax, metric, low, high, color, title in (
        (left, "BPCER", "BPCER_ci_low", "BPCER_ci_high", ORANGE, "BPCER, all 20 speakers"),
        (right, "APCER", "APCER_ci_low", "APCER_ci_high", BLUE, "APCER, the 10 speakers with spoof files"),
    ):

        table = (
            speaker_table.dropna(subset=[metric])
            .sort_values(metric)
            .reset_index(drop=True)
        )

        ax.errorbar(
            table[metric] * 100,
            table.index,
            xerr=[
                (table[metric] - table[low]) * 100,
                (table[high] - table[metric]) * 100,
            ],
            fmt="o",
            color=color,
            markersize=5,
            elinewidth=1.1,
        )

        ax.set_yticks(table.index, table["speaker"])
        ax.grid(False, axis="y")
        ax.set_xlim(left=0)
        ax.set_xlabel(f"{metric} (%), with 95% Clopper-Pearson interval")
        ax.set_title(f"{title}, at the {chosen} threshold", loc="left")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "speaker_errors.png")
    plt.close(fig)


def plot_tdcf(thresholds, tdcf, point, value):

    fig, ax = plt.subplots(figsize=(9, 4.8))

    ax.plot(thresholds, tdcf, color=BLUE)

    ax.plot(
        point,
        value,
        "v",
        color=INK,
        markersize=8,
        markeredgecolor=SURFACE,
    )

    ax.annotate(
        f"min t-DCF {value:.4f} at threshold {point:.3f}",
        (point, value),
        xytext=(0, -24),
        textcoords="offset points",
        ha="center",
        color=INK_SECONDARY,
        fontsize=9,
    )

    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("SVM decision threshold")
    ax.set_ylabel("Normalized t-DCF")
    ax.set_title(
        "Normalized t-DCF against threshold (ASV baseline fixed at its EER point)",
        loc="left",
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tdcf.png")
    plt.close(fig)


# --------------------------------------------------
# 7. MAIN PIPELINE
# --------------------------------------------------

def main():

    print("====================================")
    print(" THRESHOLD STUDY")
    print(" LFCC + RBF-SVM, development set only")
    print("====================================")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    hashes_before = {str(p): sha256(p) for p in protected_files()}

    model_hash = hashes_before[str(MODEL_PATH)]

    # ---------- Model and cached features ----------

    model = joblib.load(MODEL_PATH)

    data = np.load(DEV_FEATURES_PATH)

    X = data["X"]
    y = data["y"]

    check("Model expects 80 features", model.n_features_in_ == 80)
    check("Cached features are 80-dimensional", X.shape[1] == 80, str(X.shape))
    check("Cached features are finite", bool(np.isfinite(X).all()))

    protocol = load_cm_protocol(y)

    file_ids = protocol["file_id"].to_numpy(dtype="U32")
    speakers = protocol["speaker"].to_numpy(dtype="U16")
    systems = protocol["system"].to_numpy(dtype="U8")

    # Row order: re-extract a spread of files and compare to the cache
    sample = np.linspace(0, len(y) - 1, 12).astype(int)

    largest_gap = max(
        np.abs(
            extract_lfcc_from_file(DEV_AUDIO_DIR / f"{file_ids[i]}.flac") - X[i]
        ).max()
        for i in sample
    )

    check(
        "Re-extracted features match cached rows",
        largest_gap < 1e-5,
        f"12 files, largest gap {largest_gap:.2e}",
    )

    # ---------- Scores ----------

    scores = model.decision_function(X)
    predictions = model.predict(X)

    check(
        "predict() equals score > 0",
        np.array_equal(predictions, (scores > 0).astype(int)),
    )

    np.savez_compressed(
        OUTPUT_DIR / "dev_scores.npz",
        file_id=file_ids,
        speaker=speakers,
        system=systems,
        y=y,
        score=scores,
    )

    # ---------- t-DCF inputs ----------

    asv = load_asv()

    tdcf_status = "COMPLETED" if asv is not None else "SKIPPED: ASV inputs missing"

    if asv is not None:

        check(
            "ASV spoof trials per attack match the CM protocol",
            asv["spoof_by_attack"]
            == {a: int(np.sum(systems == a)) for a in ATTACKS},
        )

    # ---------- Exact sweep ----------

    thresholds, apcer, bpcer = error_curve(scores, y)

    acer = (apcer + bpcer) / 2

    n_spoof = np.sum(y == 1)
    n_bonafide = np.sum(y == 0)

    accuracy = (
        n_spoof * (1 - apcer) + n_bonafide * (1 - bpcer)
    ) / len(y)

    tdcf = tdcf_norm(apcer, bpcer, asv)

    pd.DataFrame({
        "threshold": thresholds,
        "accuracy": accuracy,
        "APCER": apcer,
        "BPCER": bpcer,
        "ACER": acer,
        "tDCF_norm": tdcf,
    }).to_csv(OUTPUT_DIR / "threshold_sweep_full.csv", index=False)

    point, point_stats = select_thresholds(scores, y, asv)

    fpr, tpr, _ = roc_curve(y, scores, pos_label=1, drop_intermediate=False)

    eer_interpolated = brentq(
        lambda x: 1.0 - x - np.interp(x, fpr, tpr),
        0.0,
        1.0,
    )

    auc = roc_auc_score(y, scores)

    # ---------- Verification of the sweep ----------

    rng = np.random.default_rng(SEED)

    probe = np.concatenate([
        rng.choice(thresholds, 200, replace=False),
        [point[name] for name in CANDIDATES if not np.isnan(point[name])],
    ])

    largest_gap = 0.0

    for t in probe:

        tn, fp, fn, tp = confusion_matrix(
            y,
            (scores >= t).astype(int),
            labels=[0, 1],
        ).ravel()

        a, b, _, acc = rates_at(scores, y, t)

        largest_gap = max(
            largest_gap,
            abs(a - fn / (fn + tp)),
            abs(b - fp / (fp + tn)),
            abs(acc - (tp + tn) / (tp + tn + fp + fn)),
        )

    check(
        "Sweep rates match sklearn confusion matrices",
        largest_gap < 1e-12,
        f"{len(probe)} thresholds, largest gap {largest_gap:.1e}",
    )

    indices = np.searchsorted(thresholds, probe)

    sweep_gap = np.abs(
        apcer[indices]
        - np.array([rates_at(scores, y, t)[0] for t in probe])
    ).max()

    check("Vectorised sweep matches direct counting", sweep_gap < 1e-12)

    roc_acer = ((1 - tpr) + fpr).min() / 2

    check(
        "Minimum ACER matches an independent ROC scan",
        abs(roc_acer - point_stats["min_ACER"]) < 1e-12,
        f"{roc_acer * 100:.4f}%",
    )

    det_fpr, det_fnr, _ = det_curve(y, scores, drop_intermediate=False)

    check(
        "Minimum ACER matches sklearn det_curve",
        abs((det_fpr + det_fnr).min() / 2 - point_stats["min_ACER"]) < 1e-12,
    )

    check(
        "AUC matches sklearn",
        abs(fast_auc(scores, y) - auc) < 1e-10,
        f"{auc:.5f}",
    )

    frr_cm, far_cm, cm_thresholds = official_det_curve(
        -scores[y == 0],
        -scores[y == 1],
    )

    official_index = np.argmin(np.abs(frr_cm - far_cm))

    official_eer = (frr_cm[official_index] + far_cm[official_index]) / 2

    check(
        "EER matches official-style scoring code",
        abs(official_eer - point_stats["EER"]) < 1e-3,
        f"official {official_eer * 100:.4f}%, "
        f"study {point_stats['EER'] * 100:.4f}%",
    )

    check(
        "Interpolated EER matches evaluate.py method and baseline",
        round(eer_interpolated * 100, 2) == 7.20,
        f"{eer_interpolated * 100:.4f}%",
    )

    for label, threshold, reported in (
        ("default threshold", 0.0, REPORTED_DEFAULT),
        ("threshold 0.488", 0.488, REPORTED_0488),
    ):

        a, b, c, acc = rates_at(scores, y, threshold)

        measured = {
            "accuracy": acc * 100,
            "APCER": a * 100,
            "BPCER": b * 100,
            "ACER": c * 100,
        }

        check(
            f"Reported figures reproduced at {label}",
            all(
                abs(round(measured[key], 2) - reported[key]) < 0.0051
                for key in reported
            ),
            ", ".join(f"{k} {v:.3f}" for k, v in measured.items()),
        )

    check("Reported AUC reproduced", round(auc, 4) == 0.9818, f"{auc:.5f}")

    previous = pd.read_csv(ARTIFACT_DIR / "threshold_results.csv")

    previous_gap = max(
        abs(rates_at(scores, y, row.threshold)[2] - row.ACER)
        for row in previous.itertuples()
    )

    check(
        "Existing threshold_results.csv agrees with the sweep",
        previous_gap < 1e-12,
        f"{len(previous)} rows, largest ACER gap {previous_gap:.1e}",
    )

    if asv is not None:

        c1, c2 = asv["C1"], asv["C2"]

        official_tdcf = (c1 * frr_cm + c2 * far_cm) / min(c1, c2)

        check(
            "Min t-DCF matches official-style scoring code",
            abs(official_tdcf.min() - point_stats["min_tDCF"]) < 1e-10,
            f"{official_tdcf.min():.5f}",
        )

        check(
            "Min t-DCF does not exceed the default cost of 1",
            point_stats["min_tDCF"] <= 1,
        )

    # ---------- Threshold comparison ----------

    comparison = []

    for name in CANDIDATES:

        if np.isnan(point[name]):
            continue

        a, b, c, acc = rates_at(scores, y, point[name])

        worst_attack = max(
            np.mean(scores[systems == attack] < point[name])
            for attack in ATTACKS
        )

        comparison.append({
            "candidate": name,
            "threshold": point[name],
            "accuracy": acc,
            "APCER": a,
            "BPCER": b,
            "ACER": c,
            "worst_attack_APCER": worst_attack,
            "ACER_with_worst_attack": (worst_attack + b) / 2,
            "tDCF_norm": float(tdcf_norm(a, b, asv)),
        })

    comparison = pd.DataFrame(comparison)

    comparison.to_csv(OUTPUT_DIR / "threshold_comparison.csv", index=False)

    # ---------- ACER plateau ----------

    plateau_rows = []

    for tolerance in (0.0005, 0.001, 0.0025, 0.005):

        inside = acer <= point_stats["min_ACER"] + tolerance

        low, high = thresholds[inside].min(), thresholds[inside].max()

        span = (thresholds >= low) & (thresholds <= high)

        plateau_rows.append({
            "ACER_tolerance_pp": tolerance * 100,
            "threshold_low": low,
            "threshold_high": high,
            "width": high - low,
            "share_of_span_within_tolerance": inside[span].mean(),
            "worst_ACER_in_span": acer[span].max(),
        })

    plateau_table = pd.DataFrame(plateau_rows)

    plateau_table.to_csv(OUTPUT_DIR / "acer_plateau.csv", index=False)

    # ---------- Score distributions ----------

    def describe(name, values):

        q = np.percentile(values, [0, 1, 5, 25, 50, 75, 95, 99, 100])

        return {
            "group": name,
            "n": len(values),
            "mean": values.mean(),
            "sd": values.std(ddof=1),
            "min": q[0],
            "p01": q[1],
            "p05": q[2],
            "p25": q[3],
            "median": q[4],
            "p75": q[5],
            "p95": q[6],
            "p99": q[7],
            "max": q[8],
            "share_inside_margin": np.mean(np.abs(values) <= 1),
        }

    score_stats = pd.DataFrame([
        describe("bonafide", scores[y == 0]),
        describe("spoof", scores[y == 1]),
        *[describe(a, scores[systems == a]) for a in ATTACKS],
    ])

    score_stats.to_csv(OUTPUT_DIR / "score_stats.csv", index=False)

    # ---------- Bootstrap ----------

    grid = np.round(np.arange(-0.25, 0.9001, 0.005), 3)

    speaker_boot = run_speaker_bootstrap(
        scores,
        y,
        systems,
        speakers,
        asv,
        point,
        grid,
        np.random.default_rng(SEED + 1),
    )

    file_boot = run_file_bootstrap(
        scores,
        y,
        asv,
        point,
        np.random.default_rng(SEED + 2),
    )

    # Drawing each speaker exactly once must rebuild the full dev set
    all_rows = np.concatenate([
        np.where(speakers == speaker)[0]
        for speaker in np.unique(speakers)
    ])

    rebuilt_selected, rebuilt_stats = select_thresholds(
        scores[all_rows],
        y[all_rows],
        asv,
    )

    check(
        "Drawing each speaker once reproduces every full-dev value",
        len(all_rows) == len(y)
        and len(np.unique(all_rows)) == len(y)
        and all(
            rebuilt_selected[name] == point[name]
            for name in CANDIDATES
            if not np.isnan(point[name])
        )
        and all(
            rebuilt_stats[name] == point_stats[name]
            for name in point_stats
            if not np.isnan(point_stats[name])
        ),
    )

    check(
        "Every bootstrap replicate produced finite results",
        bool(np.isfinite(speaker_boot["at_fixed"][:, :, :4]).all())
        and bool(np.isfinite(file_boot["at_fixed"][:, :, :4]).all())
        and bool(np.isfinite(speaker_boot["attack_apcer"]).all()),
        f"{N_BOOTSTRAP} speaker and {N_BOOTSTRAP} file replicates",
    )

    # ---------- Confidence intervals ----------

    metric_names = ["APCER", "BPCER", "ACER", "accuracy", "tDCF_norm"]

    ci_rows = []

    for c, name in enumerate(CANDIDATES):

        if np.isnan(point[name]):
            continue

        a, b, ac, acc = rates_at(scores, y, point[name])

        values = {
            "APCER": a,
            "BPCER": b,
            "ACER": ac,
            "accuracy": acc,
            "tDCF_norm": float(tdcf_norm(a, b, asv)),
        }

        exact = {
            "APCER": clopper_pearson(np.sum(scores[y == 1] < point[name]), n_spoof),
            "BPCER": clopper_pearson(np.sum(scores[y == 0] >= point[name]), n_bonafide),
            "accuracy": clopper_pearson(
                np.sum((scores >= point[name]).astype(int) == y),
                len(y),
            ),
        }

        for m, metric in enumerate(metric_names):

            if np.isnan(values[metric]):
                continue

            speaker_values = speaker_boot["at_fixed"][:, c, m]
            file_values = file_boot["at_fixed"][:, c, m]

            speaker_low, speaker_high = percentile_ci(speaker_values)
            file_low, file_high = percentile_ci(file_values)

            cp_low, cp_high = exact.get(metric, (np.nan, np.nan))

            ci_rows.append({
                "operating_point": name,
                "threshold": point[name],
                "metric": metric,
                "value": values[metric],
                "speaker_ci_low": speaker_low,
                "speaker_ci_high": speaker_high,
                "speaker_se": np.nanstd(speaker_values, ddof=1),
                "file_ci_low": file_low,
                "file_ci_high": file_high,
                "file_se": np.nanstd(file_values, ddof=1),
                "clopper_pearson_low": cp_low,
                "clopper_pearson_high": cp_high,
                "design_effect": (
                    np.nanvar(speaker_values, ddof=1)
                    / np.nanvar(file_values, ddof=1)
                ),
            })

    free_metrics = [
        ("EER", point_stats["EER"], 0),
        ("min_ACER", point_stats["min_ACER"], 1),
        ("min_tDCF", point_stats["min_tDCF"], 2),
        ("AUC", auc, 3),
    ]

    for metric, value, column in free_metrics:

        if np.isnan(value):
            continue

        speaker_values = speaker_boot["summary"][:, column]
        file_values = file_boot["summary"][:, column]

        speaker_low, speaker_high = percentile_ci(speaker_values)
        file_low, file_high = percentile_ci(file_values)

        ci_rows.append({
            "operating_point": "threshold-free",
            "threshold": np.nan,
            "metric": metric,
            "value": value,
            "speaker_ci_low": speaker_low,
            "speaker_ci_high": speaker_high,
            "speaker_se": np.nanstd(speaker_values, ddof=1),
            "file_ci_low": file_low,
            "file_ci_high": file_high,
            "file_se": np.nanstd(file_values, ddof=1),
            "clopper_pearson_low": np.nan,
            "clopper_pearson_high": np.nan,
            "design_effect": (
                np.nanvar(speaker_values, ddof=1)
                / np.nanvar(file_values, ddof=1)
            ),
        })

    ci_table = pd.DataFrame(ci_rows)

    ci_table.to_csv(OUTPUT_DIR / "bootstrap_metric_ci.csv", index=False)

    fixed_rows = ci_table[ci_table["operating_point"] != "threshold-free"]

    check(
        "Every speaker-bootstrap interval contains its fixed-threshold value",
        bool(
            (
                (fixed_rows["speaker_ci_low"] <= fixed_rows["value"])
                & (fixed_rows["value"] <= fixed_rows["speaker_ci_high"])
            ).all()
        ),
    )

    # ---------- Threshold stability ----------

    pd.DataFrame(
        speaker_boot["selected"],
        columns=[f"speaker_{n}" for n in CANDIDATES],
    ).join(
        pd.DataFrame(
            file_boot["selected"],
            columns=[f"file_{n}" for n in CANDIDATES],
        )
    ).to_csv(OUTPUT_DIR / "bootstrap_thresholds.csv", index=False)

    stability_rows = []

    for c, name in enumerate(CANDIDATES):

        if name == "default" or np.isnan(point[name]):
            continue

        for kind, boot in (("speaker", speaker_boot), ("file", file_boot)):

            values = boot["selected"][:, c]

            low, median, high = np.nanpercentile(values, [2.5, 50, 97.5])

            # What each replicate's threshold would cost on the full dev set
            indices = np.clip(
                np.searchsorted(thresholds, values),
                0,
                len(acer) - 1,
            )

            stability_rows.append({
                "candidate": name,
                "bootstrap": kind,
                "full_dev_threshold": point[name],
                "median": median,
                "sd": np.nanstd(values, ddof=1),
                "ci_low": low,
                "ci_high": high,
                "ci_width": high - low,
                "mean_full_dev_ACER_at_replicate_threshold": np.nanmean(acer[indices]),
            })

    stability = pd.DataFrame(stability_rows)

    stability.to_csv(OUTPUT_DIR / "threshold_stability.csv", index=False)

    # ---------- Held-out speakers ----------

    loso_speakers, loso_pooled = leave_one_speaker_out(scores, y, speakers, asv)

    loso_speakers.to_csv(OUTPUT_DIR / "loso_per_speaker.csv", index=False)
    loso_pooled.to_csv(OUTPUT_DIR / "loso_pooled.csv", index=False)

    oob = speaker_boot["out_of_bag"]

    valid_oob = ~np.isnan(oob[:, CANDIDATES.index("EER"), 2])

    oob_rows = []

    for c, name in enumerate(CANDIDATES):

        if np.isnan(point[name]):
            continue

        low, high = percentile_ci(oob[valid_oob, c, 2])

        oob_rows.append({
            "candidate": name,
            "valid_replicates": int(valid_oob.sum()),
            "mean_oob_APCER": np.nanmean(oob[valid_oob, c, 0]),
            "mean_oob_BPCER": np.nanmean(oob[valid_oob, c, 1]),
            "mean_oob_ACER": np.nanmean(oob[valid_oob, c, 2]),
            "oob_ACER_ci_low": low,
            "oob_ACER_ci_high": high,
        })

    oob_table = pd.DataFrame(oob_rows)

    oob_table.to_csv(OUTPUT_DIR / "out_of_bag_acer.csv", index=False)

    # ---------- Selection rule ----------

    pooled_acer = loso_pooled.set_index("candidate")["ACER"]

    best = min(SELECTABLE, key=lambda name: pooled_acer[name])

    best_column = CANDIDATES.index(best)

    paired_rows = []

    tied = [best]

    for name in CANDIDATES:

        if name == best or np.isnan(point[name]):
            continue

        difference = (
            oob[valid_oob, CANDIDATES.index(name), 2]
            - oob[valid_oob, best_column, 2]
        )

        low, high = percentile_ci(difference)

        paired_rows.append({
            "candidate": name,
            "compared_to": best,
            "mean_ACER_difference": np.nanmean(difference),
            "ci_low": low,
            "ci_high": high,
            "interval_includes_zero": bool(low <= 0 <= high),
        })

        if name in SELECTABLE and low <= 0 <= high:
            tied.append(name)

    paired_table = pd.DataFrame(paired_rows)

    paired_table.to_csv(OUTPUT_DIR / "out_of_bag_paired_differences.csv", index=False)

    speaker_sd = stability[stability["bootstrap"] == "speaker"].set_index("candidate")["sd"]

    chosen = min(tied, key=lambda name: speaker_sd[name])

    chosen_threshold = float(point[chosen])

    selection = {
        "selected_candidate": chosen,
        "threshold": chosen_threshold,
        "decision_rule": "classify spoof when decision_function(x) >= threshold",
        "selection_data": "ASVspoof 2019 LA development set only; evaluation data not read",
        "rule": (
            "Lowest leave-one-speaker-out pooled ACER among "
            f"{SELECTABLE}; candidates whose paired out-of-bag ACER "
            "difference interval includes zero count as tied, and the tied "
            "candidate with the lowest speaker-bootstrap threshold SD is chosen."
        ),
        "lowest_held_out_ACER_candidate": best,
        "tied_candidates": tied,
        "model_sha256": model_hash,
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
    }

    with open(OUTPUT_DIR / "selected_threshold.json", "w") as file:
        json.dump(selection, file, indent=2)

    # ---------- Attack-specific APCER ----------

    attack_rows = []

    for c, name in enumerate(CANDIDATES):

        if np.isnan(point[name]):
            continue

        for a, attack in enumerate(ATTACKS):

            values = scores[systems == attack]

            errors = np.sum(values < point[name])

            cp_low, cp_high = clopper_pearson(errors, len(values))

            low, high = percentile_ci(speaker_boot["attack_apcer"][:, c, a])

            attack_rows.append({
                "candidate": name,
                "threshold": point[name],
                "attack": attack,
                "n_files": len(values),
                "errors": int(errors),
                "APCER": errors / len(values),
                "speaker_ci_low": low,
                "speaker_ci_high": high,
                "clopper_pearson_low": cp_low,
                "clopper_pearson_high": cp_high,
            })

    attack_table = pd.DataFrame(attack_rows)

    attack_table.to_csv(OUTPUT_DIR / "apcer_by_attack.csv", index=False)

    attack_eer_rows = []

    for attack in ATTACKS:

        mask = (y == 0) | (systems == attack)

        t_a, a_a, b_a = error_curve(scores[mask], y[mask])

        index = np.argmin(np.abs(a_a - b_a))

        attack_eer_rows.append({
            "attack": attack,
            "EER": (a_a[index] + b_a[index]) / 2,
            "EER_threshold": t_a[index],
            "AUC": roc_auc_score(y[mask], scores[mask]),
        })

    attack_eer = pd.DataFrame(attack_eer_rows)

    attack_eer.to_csv(OUTPUT_DIR / "eer_by_attack.csv", index=False)

    pooled_gap = abs(
        attack_table[attack_table["candidate"] == chosen]["APCER"].mean()
        - rates_at(scores, y, chosen_threshold)[0]
    )

    check(
        "Mean attack APCER equals pooled APCER (attacks are balanced)",
        pooled_gap < 1e-12,
    )

    # ---------- Speaker-level analysis ----------

    genders = {}

    for gender, filename in (
        ("male", "ASVspoof2019.LA.asv.dev.male.trn.txt"),
        ("female", "ASVspoof2019.LA.asv.dev.female.trn.txt"),
    ):

        with open(ASV_DIR / filename, "r") as file:
            for line in file:
                genders[line.split()[0]] = gender

    speaker_rows = []

    for speaker in np.unique(speakers):

        mask = speakers == speaker

        bonafide = scores[mask & (y == 0)]
        spoof = scores[mask & (y == 1)]

        bpcer_errors = np.sum(bonafide >= chosen_threshold)
        apcer_errors = np.sum(spoof < chosen_threshold)

        b_low, b_high = clopper_pearson(bpcer_errors, len(bonafide))

        row = {
            "speaker": speaker,
            "enrolled_gender": genders.get(speaker, "not in ASV enrolment lists"),
            "n_bonafide": len(bonafide),
            "n_spoof": len(spoof),
            "bonafide_mean_score": bonafide.mean(),
            "BPCER": bpcer_errors / len(bonafide),
            "BPCER_ci_low": b_low,
            "BPCER_ci_high": b_high,
            "bpcer_errors": int(bpcer_errors),
            "apcer_errors": int(apcer_errors),
            "spoof_mean_score": spoof.mean() if len(spoof) else np.nan,
            "APCER": np.nan,
            "APCER_ci_low": np.nan,
            "APCER_ci_high": np.nan,
            "ACER": np.nan,
            "speaker_EER": np.nan,
        }

        if len(spoof):

            a_low, a_high = clopper_pearson(apcer_errors, len(spoof))

            row["APCER"] = apcer_errors / len(spoof)
            row["APCER_ci_low"] = a_low
            row["APCER_ci_high"] = a_high
            row["ACER"] = (row["APCER"] + row["BPCER"]) / 2

            t_s, a_s, b_s = error_curve(
                scores[mask],
                y[mask],
            )

            index = np.argmin(np.abs(a_s - b_s))

            row["speaker_EER"] = (a_s[index] + b_s[index]) / 2

        speaker_rows.append(row)

    speaker_table = pd.DataFrame(speaker_rows)

    speaker_table.to_csv(OUTPUT_DIR / "speaker_analysis.csv", index=False)

    check(
        "Speaker error counts sum to the pooled counts",
        speaker_table["bpcer_errors"].sum() == np.sum(scores[y == 0] >= chosen_threshold)
        and speaker_table["apcer_errors"].sum() == np.sum(scores[y == 1] < chosen_threshold),
    )

    bpcer_table = np.column_stack([
        speaker_table["bpcer_errors"],
        speaker_table["n_bonafide"] - speaker_table["bpcer_errors"],
    ])

    spoof_speakers_table = speaker_table[speaker_table["n_spoof"] > 0]

    apcer_table = np.column_stack([
        spoof_speakers_table["apcer_errors"],
        spoof_speakers_table["n_spoof"] - spoof_speakers_table["apcer_errors"],
    ])

    bpcer_test = chi2_contingency(bpcer_table)
    apcer_test = chi2_contingency(apcer_table)

    with_spoof = speaker_table["n_spoof"] > 0

    group_bpcer = {
        "with_spoof": (
            speaker_table.loc[with_spoof, "bpcer_errors"].sum()
            / speaker_table.loc[with_spoof, "n_bonafide"].sum()
        ),
        "without_spoof": (
            speaker_table.loc[~with_spoof, "bpcer_errors"].sum()
            / speaker_table.loc[~with_spoof, "n_bonafide"].sum()
        ),
    }

    # ---------- Plots ----------

    marks = {
        name: float(point[name])
        for name in ("EER", "min_ACER", "plateau_midpoint")
    }

    points = {}

    for name in CANDIDATES:
        if not np.isnan(point[name]):
            a, b, _, _ = rates_at(scores, y, point[name])
            points[name] = (a, b)

    plot_score_distributions(scores, y, systems, marks)
    plot_roc(apcer, bpcer, auc, points)
    plot_det(y, scores, points, point_stats["EER"])
    plot_apcer_bpcer(thresholds, apcer, bpcer, marks)
    plot_acer(thresholds, acer, grid, speaker_boot["grid_acer"], marks)
    plot_threshold_stability(speaker_boot, file_boot, point)
    plot_attack_apcer(attack_table, chosen)
    plot_speakers(speaker_table, chosen)

    if asv is not None:
        plot_tdcf(
            thresholds,
            tdcf,
            float(point["min_tDCF"]),
            float(point_stats["min_tDCF"]),
        )

    # ---------- Protected files ----------

    changed = [
        path
        for path, digest in hashes_before.items()
        if sha256(path) != digest
    ]

    check(
        "Model, scripts and existing artifacts are byte-identical",
        not changed,
        f"{len(hashes_before)} files hashed" if not changed else str(changed),
    )

    # ---------- Report ----------

    check_table = pd.DataFrame(CHECKS)

    check_table.to_csv(OUTPUT_DIR / "verification_checks.csv", index=False)

    failures = (check_table["result"] == "FAIL").sum()

    pct = lambda value: f"{value * 100:.2f}%"

    def interval(metric, operating_point):

        row = ci_table[
            (ci_table["metric"] == metric)
            & (ci_table["operating_point"] == operating_point)
        ].iloc[0]

        return row

    lines = [
        "THRESHOLD STUDY: LFCC + RBF-SVM, ASVspoof 2019 LA DEVELOPMENT SET",
        "=" * 66,
        "",
        "Every number below is from the development set. Evaluation data was",
        "not read, so these results do not estimate performance on the unseen",
        "A07-A19 attacks.",
        "",
        f"Decision rule: spoof when decision_function(x) >= threshold",
        f"Bootstrap replicates: {N_BOOTSTRAP} (seed {SEED})",
        f"Model SHA-256: {model_hash}",
        (
            f"Python {platform.python_version()}, numpy {np.__version__}, "
            f"scipy {scipy.__version__}, scikit-learn {sklearn.__version__}, "
            f"pandas {pd.__version__}, matplotlib {matplotlib.__version__}"
        ),
        "",
        "SELECTED THRESHOLD",
        "-" * 66,
        f"Selected: {chosen} = {chosen_threshold:.6f}",
        f"Lowest held-out ACER: {best}; tied candidates: {tied}",
        f"Rule: {selection['rule']}",
        "",
        "OPERATING POINTS",
        "-" * 66,
        comparison.to_string(
            index=False,
            float_format=lambda v: f"{v:.5f}",
        ),
        "",
        "THRESHOLD-FREE METRICS",
        "-" * 66,
        f"EER (interpolated, as in evaluate.py): {eer_interpolated * 100:.4f}%",
        f"EER (mean of APCER and BPCER at the closest score): {point_stats['EER'] * 100:.4f}%",
    ]

    for metric, _, _ in free_metrics:

        row = interval(metric, "threshold-free")

        lines.append(
            f"{metric}: {row['value']:.5f} | speaker CI "
            f"{row['speaker_ci_low']:.5f}-{row['speaker_ci_high']:.5f} | file CI "
            f"{row['file_ci_low']:.5f}-{row['file_ci_high']:.5f}"
        )

    lines += [
        "",
        "HELD-OUT PERFORMANCE",
        "-" * 66,
        "Leave-one-speaker-out, errors pooled across all 20 speakers:",
        loso_pooled[["candidate", "threshold_sd", "APCER", "BPCER", "ACER"]].to_string(
            index=False,
            float_format=lambda v: f"{v:.5f}",
        ),
        "",
        "Bootstrap out-of-bag ACER:",
        oob_table.to_string(index=False, float_format=lambda v: f"{v:.5f}"),
        "",
        f"Paired out-of-bag ACER difference against {best}:",
        paired_table.to_string(index=False, float_format=lambda v: f"{v:.5f}"),
        "",
        "THRESHOLD STABILITY",
        "-" * 66,
        stability.to_string(index=False, float_format=lambda v: f"{v:.5f}"),
        "",
        "ACER plateau around the minimum:",
        plateau_table.to_string(index=False, float_format=lambda v: f"{v:.5f}"),
        "",
        "ATTACK-SPECIFIC APCER (selected threshold)",
        "-" * 66,
        attack_table[attack_table["candidate"] == chosen][[
            "attack",
            "n_files",
            "errors",
            "APCER",
            "speaker_ci_low",
            "speaker_ci_high",
            "clopper_pearson_low",
            "clopper_pearson_high",
        ]].to_string(index=False, float_format=lambda v: f"{v:.5f}"),
        "",
        "EER by attack (all bona fide files against that attack):",
        attack_eer.to_string(index=False, float_format=lambda v: f"{v:.5f}"),
        "",
        "SPEAKER-LEVEL ANALYSIS (selected threshold)",
        "-" * 66,
        speaker_table[[
            "speaker",
            "enrolled_gender",
            "n_bonafide",
            "n_spoof",
            "BPCER",
            "APCER",
            "ACER",
            "speaker_EER",
        ]].to_string(index=False, float_format=lambda v: f"{v:.4f}"),
        "",
        (
            f"BPCER heterogeneity across 20 speakers: chi2={bpcer_test.statistic:.1f}, "
            f"df={bpcer_test.dof}, p={bpcer_test.pvalue:.2e}"
        ),
        (
            f"APCER heterogeneity across 10 speakers: chi2={apcer_test.statistic:.1f}, "
            f"df={apcer_test.dof}, p={apcer_test.pvalue:.2e}"
        ),
        (
            f"Pooled BPCER: speakers with spoof files {pct(group_bpcer['with_spoof'])}, "
            f"speakers without {pct(group_bpcer['without_spoof'])}"
        ),
        "",
        "Limitations of the speaker analysis:",
        "- Only 20 speakers exist in dev, and only 10 have spoof files. APCER,",
        "  ACER and per-speaker EER cannot be calculated for the other 10.",
        "- Percentile bootstrap intervals with this few speakers can be too",
        "  narrow, so treat them as a lower bound on uncertainty.",
        "- The 2 strata are resampled separately to keep both classes in every",
        "  replicate, which assumes those groups are fixed by the design.",
        "- Each speaker has only 77-154 bona fide files, so a single speaker's",
        "  BPCER is imprecise. Check the intervals in speaker_errors.png.",
        "- Chi-square tests treat files as independent within a speaker. That is",
        "  unlikely to hold exactly, so their p-values may be too small.",
        "- These patterns describe these 20 people, not a wider population, and",
        "  the study does not explain why any speaker scores differently.",
        "- Gender comes from ASV enrolment lists covering 10 speakers, too few",
        "  to support any comparison between genders.",
        "",
        "MIN t-DCF",
        "-" * 66,
        f"Status: {tdcf_status}",
    ]

    if asv is not None:

        lines += [
            "Costs: ASVspoof 2019 evaluation plan v0.4, Table 2, logical access",
            json.dumps(COST_MODEL),
            f"Formula: evaluation plan Eq. 1-5, with C1 = {asv['C1']:.6f}, C2 = {asv['C2']:.6f}",
            (
                f"ASV trials: {asv['n_target']} target, {asv['n_nontarget']} nontarget, "
                f"{asv['n_spoof']} spoof"
            ),
            (
                f"ASV threshold {asv['threshold']:.5f} at its EER point: EER {pct(asv['eer'])}, "
                f"Pmiss {pct(asv['Pmiss_asv'])}, Pfa {pct(asv['Pfa_asv'])}, "
                f"Pmiss_spoof {pct(asv['Pmiss_spoof_asv'])}"
            ),
            (
                f"Min normalized t-DCF: {point_stats['min_tDCF']:.5f} at threshold "
                f"{float(point['min_tDCF']):.5f}"
            ),
            (
                "Bootstrap intervals resample CM trials only, holding the ASV "
                "system fixed, as the plan treats it."
            ),
            (
                "This is the 2019 formulation. Later ASVspoof scoring tools changed "
                "it, so compare only with results using the same version."
            ),
        ]

    lines += [
        "",
        "VERIFICATION",
        "-" * 66,
        check_table.to_string(index=False),
        "",
        f"Checks passed: {(check_table['result'] == 'PASS').sum()} of {len(check_table)}",
    ]

    with open(OUTPUT_DIR / "report.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")

    print("\n".join(lines))

    print(f"\nOutputs saved to: {OUTPUT_DIR}")

    if failures:
        print(f"\n{failures} verification check(s) FAILED.")
        sys.exit(1)

    print("\n====================================")
    print(" STUDY COMPLETED")
    print("====================================")


if __name__ == "__main__":
    main()
