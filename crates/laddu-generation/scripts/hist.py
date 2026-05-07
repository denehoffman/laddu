#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = ["matplotlib"]
# ///
import argparse
import json
import sys

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Read histogram JSON from stdin and create a histogram plot.'
    )
    parser.add_argument(
        '-o', '--output', default='histogram.png', help='Output image path'
    )
    parser.add_argument('--title', default='Histogram')
    parser.add_argument('--xlabel', default='Value')
    parser.add_argument('--ylabel', default='Count')
    args = parser.parse_args()

    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        msg = f'Invalid JSON: {e}'
        raise SystemExit(msg) from e

    # Supported inputs:
    #    {"values": [1, 2, 2, 3, 4], "bins": 10, "min": 1.0, "max": 2.0}

    values = data['values']
    bins = data['bins']
    min_edge = data['min']
    max_edge = data['max']

    plt.hist(values, bins=bins, range=(min_edge, max_edge), histtype='step')

    plt.title(data.get('title', args.title))
    plt.xlabel(data.get('xlabel', args.xlabel))
    plt.ylabel(data.get('ylabel', args.ylabel))
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)


if __name__ == '__main__':
    main()
