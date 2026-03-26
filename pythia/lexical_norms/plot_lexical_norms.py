import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_MODELS = ["pythia6p9b_step0", "pythia6p9b_step143000"]
SERIES_CONFIGS = [
    ("Concat", 0, None),
    ("Avg", 1, None),
    ("Last token", 0, 1),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Pythia lexical/syntax/semantic norm fractions.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--min-token-length", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=2018)
    parser.add_argument("--global-center-flag", type=int, default=1)
    return parser.parse_args()


def load_run(base, avg_tokens, min_token_length, n_samples, global_center_flag, n_tokens=None):
    root = base / f"avg_tokens_{avg_tokens}" / f"min_token_length_{min_token_length}" / f"n_samples_{n_samples}"
    if avg_tokens == 0 and n_tokens not in (None, min_token_length):
        root = root / f"n_tokens_{n_tokens}"
    root = root / "norms" / f"global_center_flag_{global_center_flag}"
    path = root / "lexical_norms.npz"
    meta_path = root / "metadata.json"
    if not path.exists():
        return None

    metadata = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())
    return path, np.load(path), metadata


def main():
    args = parse_args()

    rcpsize = 16
    plt.rcParams["xtick.labelsize"] = rcpsize
    plt.rcParams["ytick.labelsize"] = rcpsize
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["font.size"] = rcpsize
    plt.rcParams.update({"figure.autolayout": True})

    lex_color = "#b35806"
    syn_color = "#1b9e77"
    sem_color = "#7570b3"
    residual_color = "#666666"

    fig_dir = Path("/home/acevedo/syn-sem/pythia/lexical_norms/figs")
    fig_dir.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        base = Path("/home/acevedo/syn-sem/pythia/lexical_norms/results") / f"model_{model}"
        runs = {
            label: load_run(
                base,
                avg_tokens,
                args.min_token_length,
                args.n_samples,
                args.global_center_flag,
                n_tokens=n_tokens,
            )
            for label, avg_tokens, n_tokens in SERIES_CONFIGS
        }

        fig, axes = plt.subplots(1, len(SERIES_CONFIGS), figsize=(15.0, 4.2), sharey=True)
        bar_width = 0.042

        for ax, (label, _, _) in zip(axes, SERIES_CONFIGS):
            run = runs[label]
            if run is None:
                ax.set_title(f"{label} missing")
                ax.axis("off")
                continue

            _, data, _ = run
            rel_depths = data["rel_depths"]
            lex_means = data["lex_means"]
            syn_means = data["syn_means"]
            sem_means = data["sem_means"]
            residual_means = data["residual_means"]

            x = np.linspace(1 / len(rel_depths), 1, len(rel_depths))
            ax.bar(x, lex_means, width=bar_width, color=lex_color, label="lexical", alpha=0.95)
            ax.bar(x, syn_means, width=bar_width, bottom=lex_means, color=syn_color, label="syntactic", alpha=0.9)
            ax.bar(
                x,
                sem_means,
                width=bar_width,
                bottom=lex_means + syn_means,
                color=sem_color,
                label="semantic",
                alpha=0.9,
            )
            ax.bar(
                x,
                residual_means,
                width=bar_width,
                bottom=lex_means + syn_means + sem_means,
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
            ax.set_title(label, fontsize=rcpsize)
            ax.set_xlabel("relative depth", fontsize=rcpsize)

        checkpoint_label = model.replace("pythia6p9b_", "").replace("_", " ")
        fig.suptitle(f"Pythia-6.9B {checkpoint_label}", fontsize=rcpsize + 2, y=1.06)
        fig.supylabel("fraction of squared norm", fontsize=rcpsize, x=-0.01)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=4, frameon=False)
        plt.subplots_adjust(wspace=0.16, top=0.72)

        output_path = fig_dir / f"{model}_lexical_syn_sem_norms_global_centered_n{args.n_samples}.pdf"
        fig.savefig(output_path, bbox_inches="tight")
        print(output_path)


if __name__ == "__main__":
    main()
