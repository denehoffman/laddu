#!/usr/bin/env bash

set -euo pipefail

mode="${1:---check}"
case "$mode" in
    --check | --update) ;;
    *)
        echo "usage: $0 [--check|--update]" >&2
        exit 2
        ;;
esac

if ! command -v cargo-public-api >/dev/null 2>&1; then
    echo "cargo-public-api is required; install it with: cargo install cargo-public-api@0.52.0 --locked" >&2
    exit 1
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
snapshot_root="$project_root/public-api"
temporary_root="$(mktemp -d)"
trap 'rm -rf "$temporary_root"' EXIT
public_api_toolchain="${PUBLIC_API_TOOLCHAIN:-nightly-2026-07-21}"

# Keep this list explicit: adding a public Rust crate requires deciding whether
# it belongs to the workspace's compatibility contract.
packages=(
    laddu
    laddu-amplitudes
    laddu-autodiff
    laddu-compile
    laddu-data
    laddu-expr
    laddu-fit
    laddu-generation
    laddu-kernel
    laddu-likelihood
    laddu-memory
    laddu-physics
    laddu-runtime
    laddu-wgpu
)

feature_sets=(default all-features)

mkdir -p "$snapshot_root"
status=0
for package in "${packages[@]}"; do
    for feature_set in "${feature_sets[@]}"; do
        public_api_args=(
            --manifest-path "$project_root/Cargo.toml"
            --package "$package"
            --omit blanket-impls
            --omit auto-trait-impls
            --color never
        )
        if [[ "$feature_set" == "all-features" ]]; then
            public_api_args+=(--all-features)
        fi

        actual="$temporary_root/$package.$feature_set.txt"
        expected="$snapshot_root/$package.$feature_set.txt"
        command_log="$temporary_root/$package.$feature_set.log"
        if ! cargo "+$public_api_toolchain" public-api "${public_api_args[@]}" \
            >"$actual" 2>"$command_log"; then
            echo "failed to generate public API for $package ($feature_set)" >&2
            cat "$command_log" >&2
            status=1
            continue
        fi

        if [[ "$mode" == "--update" ]]; then
            cp "$actual" "$expected"
        elif [[ ! -f "$expected" ]]; then
            echo "missing public API snapshot: $expected" >&2
            status=1
        elif ! diff -u "$expected" "$actual"; then
            status=1
        fi
    done
done

if [[ "$mode" == "--update" ]]; then
    echo "updated public API snapshots in $snapshot_root"
elif [[ "$status" -eq 0 ]]; then
    echo "public API snapshots are current"
fi

exit "$status"
