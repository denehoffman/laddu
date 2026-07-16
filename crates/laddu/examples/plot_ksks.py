# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "matplotlib==3.10.7",
#   "numpy==2.3.4",
# ]
# ///
"""Plot the binned K-short closure projection emitted by generate_and_fit_ksks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="projection JSON from the Laddu example")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output image (default: input path with a .png suffix)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.input.with_suffix(".png")
    with args.input.open(encoding="utf-8") as source:
        payload = json.load(source)
    if payload.get("schema_version") != 2:
        raise ValueError(f"unsupported projection schema: {payload.get('schema_version')!r}")

    edges = np.asarray(payload["data"]["histogram"]["bin_edges"], dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    data = np.asarray(payload["data"]["histogram"]["counts"], dtype=float)
    if len(edges) != len(data) + 1:
        raise ValueError("bin_edges must contain one more entry than each histogram")

    fig, ax = plt.subplots(figsize=(8.0, 5.5), constrained_layout=True)
    ax.errorbar(
        centers,
        data,
        xerr=0.5 * widths,
        yerr=np.sqrt(np.clip(data, 0.0, None)),
        fmt="o",
        color="black",
        markersize=4,
        linewidth=1,
        capsize=0,
        label=payload["data"]["label"],
        zorder=4,
    )

    styles = {
        "fit": {"color": "#d62728", "linewidth": 2.2},
        "f0": {"color": "#1f77b4", "linewidth": 1.8, "linestyle": "--"},
        "f2": {"color": "#2ca02c", "linewidth": 1.8, "linestyle": ":"},
    }
    for series in payload["projections"]:
        values = np.asarray(series["histogram"]["counts"], dtype=float)
        if len(values) != len(data):
            raise ValueError(f"projection {series['id']!r} has the wrong number of bins")
        ax.stairs(
            values,
            edges,
            label=series["label"],
            **styles.get(series["id"], {"linewidth": 1.8}),
        )

    ax.set(
        xlabel=f"{payload['observable']} [{payload['unit']}]",
        ylabel=f"Events / {widths[0]:.3f} {payload['unit']}",
        title=payload["title"],
        xlim=(edges[0], edges[-1]),
        ylim=(0.0, None),
    )
    ax.legend(frameon=False)
    ax.tick_params(direction="in", top=True, right=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(output)


if __name__ == "__main__":
    main()
