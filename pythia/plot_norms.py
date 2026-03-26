import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Pythia syntax-semantic norm fractions.")
    parser.add_argument("--model", required=True, help="Model name, e.g. pythia6p9b_step4000")
    parser.add_argument("--min-token-length", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=2018)
    parser.add_argument("--global-center-flag", type=int, default=1)
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=Path("/home/acevedo/syn-sem/pythia/results/figs"),
    )
    return parser.parse_args()


def load_run(base, avg_tokens, min_token_length, n_samples, global_center_flag):
    root = (
        base
        / f"avg_tokens_{avg_tokens}"
        / f"min_token_length_{min_token_length}"
        / f"n_samples_{n_samples}"
        / "norms"
        / f"global_center_flag_{global_center_flag}"
    )
    path = root / "norms.npz"
    meta_path = root / "metadata.json"
    if not path.exists():
        return None

    metadata = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())
    return np.load(path), metadata


def main():
    args = parse_args()

    base = Path("/home/acevedo/syn-sem/pythia/results") / f"model_{args.model}"
    configs = [0, 1]
    runs = {
        avg_tokens: load_run(
            base,
            avg_tokens,
            args.min_token_length,
            args.n_samples,
            args.global_center_flag,
        )
        for avg_tokens in configs
    }

    if all(run is None for run in runs.values()):
        raise FileNotFoundError(
            f"No norms runs found for model={args.model}, "
            f"n_samples={args.n_samples}, global_center_flag={args.global_center_flag}"
        )

    rcpsize = 18
    plt.rcParams["xtick.labelsize"] = rcpsize
    plt.rcParams["ytick.labelsize"] = rcpsize
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["font.size"] = rcpsize
    plt.rcParams.update({"figure.autolayout": True})

    colors = plt.style.library["seaborn-v0_8-dark-palette"]["axes.prop_cycle"].by_key()["color"]
    syn_color = colors[0]
    sem_color = colors[1]
    residual_color = "#676767"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    bar_width = 0.042
    panel_labels = {0: r"$\mathbf{a)}$ Concatenation", 1: r"$\mathbf{b)}$ Average"}

    for ax, avg_tokens in zip(axes, configs):
        run = runs[avg_tokens]
        if run is None:
            ax.set_title(f"avg_tokens={avg_tokens} missing")
            ax.axis("off")
            continue

        data, metadata = run
        rel_depths = data["rel_depths"]
        syn_means = data["syn_means"]
        sem_means = data["sem_means"]
        residual_means = data["residual_means"]
        x = np.linspace(1 / len(rel_depths), 1, len(rel_depths))
        ax.bar(x, syn_means, width=bar_width, color=syn_color, label="syntactic", alpha=0.9)
        ax.bar(x, sem_means, width=bar_width, bottom=syn_means, color=sem_color, label="semantic", alpha=0.9)
        ax.bar(
            x,
            residual_means,
            width=bar_width,
            bottom=syn_means + sem_means,
            color=residual_color,
            label="residual",
            alpha=0.9,
        )
        ax.set_xlim(0.03, 1.01)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(np.linspace(0.0, 1.0, 6))
        ax.set_xticklabels([f"{tick:.1f}" for tick in np.linspace(0.0, 1.0, 6)])
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_title(panel_labels[avg_tokens], fontsize=rcpsize)
        ax.set_xlabel("relative depth", fontsize=rcpsize)

    fig.supylabel("fraction of squared norm", fontsize=rcpsize, x=-0.01)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)
    plt.subplots_adjust(wspace=0.18, top=0.80)

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.fig_dir / f"{args.model}_syn_sem_norms_global_centered_n{args.n_samples}.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    print(output_path)


if __name__ == "__main__":
    main()
