"""Profile projection dimensions without changing the frozen CPU benchmark cases."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import cpu_benchmark as benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', choices=benchmark.SIZES, default='reference')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--variation', action='store_true')
    parser.add_argument('--stress', action='store_true', help='include 200 fitted paired replicas')
    parser.add_argument('--batch-profile', action='store_true', help='also time one batch per component; adds work')
    parser.add_argument(
        '--verify-blocks',
        action='store_true',
        help='compare parallel and serial event evaluation at identical fitted inputs; correctness only',
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    base = benchmark.make_plan(
        SimpleNamespace(
            sizes=args.size,
            stress=False,
            workflows='bootstrap',
            variation=args.variation,
            repeats=args.repeats,
            memory_limit_gb=28.0,
        )
    )
    # Vary one dimension at a time around twelve components, eight replicas,
    # and two projections; retain the twelve-wave model in every case.
    dimensions = [(components, 8, 2) for components in (0, 1, 6, 12)]
    dimensions += [(12, 8, projections) for projections in (1, 4)]
    dimensions += [(12, 2, 2)]
    if args.stress:
        dimensions += [(12, 200, 2)]
    plan = {
        **base,
        'cases': [
            {
                **config,
                'projection_components': components,
                'replicas': replicas,
                'projection_count': projections,
                **({'profile_batch': True} if args.batch_profile or args.verify_blocks else {}),
                **({'verify_blocks': True} if args.verify_blocks else {}),
            }
            for components, replicas, projections in dimensions
            for config in base['cases']
        ],
    }
    run_args = SimpleNamespace(binary=args.binary.resolve(), no_build=True, output=args.output)
    return 0 if benchmark.run_matrix(run_args, plan) else 1


if __name__ == '__main__':
    raise SystemExit(main())
