# ruff: noqa: S101

import json
from pathlib import Path

import laddu as ld
import numpy as np
import pytest

ENSEMBLE_SOURCE_ID = 4242


def likelihood() -> ld.Likelihood:
    def dataset(values: list[float]) -> ld.Dataset:
        return ld.Dataset.from_arrays(p4s={}, scalars={'x': values}, weights=[1.0] * len(values))

    x = ld.scalar('x')
    model = ld.Model((ld.parameter('a', initial=0.5) * x + 1.0).norm_sqr())
    return ld.Likelihood(
        [
            ld.NLL(
                model,
                data=dataset([0.2, 0.4, 0.8]),
                accepted_mc=dataset([0.1, 0.3, 0.5, 0.9]),
                name='signal',
            )
        ]
    )


def test_fit_result_exposes_stable_converged_fields_and_raw_summary() -> None:
    fit = likelihood().fit(terminators=[ld.ganesh.MaxSteps(100)])

    assert isinstance(fit, ld.FitResult)
    assert fit.parameter_names == ['a']
    assert list(fit.named_parameters) == ['a']
    np.testing.assert_allclose(fit.values, [fit.named_parameters['a']])
    assert np.isfinite(fit.objective)
    assert fit.outcome == 'converged'
    assert fit.converged
    assert 'Success' in fit.terminal_message
    assert fit.covariance is not None
    assert fit.standard_errors is not None
    assert fit.diagnostics['function_evaluations'] > 0
    assert isinstance(fit.raw_ganesh_summary, ld.ganesh.MinimizationSummary)
    np.testing.assert_allclose(fit.raw_ganesh_summary.x, fit.values)


def test_fit_result_distinguishes_terminal_nonconvergence() -> None:
    fit = likelihood().fit(terminators=[ld.ganesh.MaxSteps(1)])

    assert fit.outcome == 'not_converged'
    assert not fit.converged
    assert 'Maximum number of steps' in fit.terminal_message


def test_fit_result_marks_uncomputed_inference_metadata_absent() -> None:
    fit = likelihood().fit(
        config=ld.ganesh.NelderMeadConfig(),
        terminators=[ld.ganesh.MaxSteps(100)],
    )

    assert fit.covariance is None
    assert fit.standard_errors is None


def test_fit_artifact_is_inspectable_and_binds_strictly() -> None:
    target = likelihood()
    fit = target.fit(terminators=[ld.ganesh.MaxSteps(100)])
    artifact = fit.artifact()

    assert artifact.artifact_kind == 'laddu.fit'
    assert artifact.schema_version == 1
    assert artifact.fingerprint_version == 1
    assert artifact.parameter_names == ['a']
    assert artifact.outcome == fit.outcome
    assert not hasattr(artifact, 'data')
    assert not hasattr(artifact, 'accepted_mc')
    bound = artifact.bind(target)
    assert bound.parameter_names == target.parameter_names
    np.testing.assert_allclose(bound.values, fit.values)
    assert bound.outcome == fit.outcome

    x = ld.scalar('x')
    changed = ld.Likelihood(
        [
            ld.NLL(
                ld.Model((ld.parameter('renamed', initial=0.5) * x + 1.0).norm_sqr()),
                data=ld.Dataset.from_arrays(p4s={}, scalars={'x': [0.2]}, weights=[1.0]),
                accepted_mc=ld.Dataset.from_arrays(p4s={}, scalars={'x': [0.1]}, weights=[1.0]),
                name='signal',
            )
        ]
    )
    with np.testing.assert_raises_regex(ld.LadduError, 'parameter schema'):
        artifact.bind(changed)

    structurally_changed = ld.Likelihood(
        [
            ld.NLL(
                ld.Model((ld.parameter('a', initial=0.5) * x + 2.0).norm_sqr()),
                data=ld.Dataset.from_arrays(p4s={}, scalars={'x': [0.2]}, weights=[1.0]),
                accepted_mc=ld.Dataset.from_arrays(p4s={}, scalars={'x': [0.1]}, weights=[1.0]),
                name='signal',
            )
        ]
    )
    with np.testing.assert_raises_regex(ld.LadduError, 'fingerprint'):
        artifact.bind(structurally_changed)


def test_fit_artifact_binds_order_changes_by_name() -> None:
    data = ld.Dataset.from_arrays(p4s={}, scalars={'x': [0.2, 0.8]}, weights=[1.0, 1.0])
    x = ld.scalar('x')
    first = ld.NLL(
        ld.Model((ld.parameter('a', initial=0.5) * x + 1.0).norm_sqr()),
        data=data,
        accepted_mc=data,
        name='first',
    )
    second = ld.NLL(
        ld.Model((ld.parameter('b', initial=0.25) * x + 1.0).norm_sqr()),
        data=data,
        accepted_mc=data,
        name='second',
    )
    original = ld.Likelihood([first, second])
    reordered = ld.Likelihood([second, first])
    fit = original.fit(terminators=[ld.ganesh.MaxSteps(1)])

    bound = fit.artifact().bind(reordered)

    assert bound.parameter_names == ['b', 'a']
    expected = dict(zip(fit.parameter_names, fit.values, strict=True))
    np.testing.assert_allclose(bound.values, [expected['b'], expected['a']])
    assert bound.outcome == 'not_converged'


def test_fit_artifact_archive_round_trip_and_rejects_corruption(tmp_path: Path) -> None:
    target = likelihood()
    artifact = target.fit(terminators=[ld.ganesh.MaxSteps(100)]).artifact()
    path = tmp_path / 'fit.laddu'

    artifact.save(path)
    restored = ld.FitArtifact.load(path)

    assert restored.artifact_kind == artifact.artifact_kind
    assert restored.parameter_names == artifact.parameter_names
    np.testing.assert_allclose(restored.values, artifact.values)
    assert restored.covariance is not None
    assert artifact.covariance is not None
    np.testing.assert_allclose(restored.covariance, artifact.covariance)
    assert restored.diagnostics == artifact.diagnostics
    np.testing.assert_allclose(restored.bind(target).values, artifact.values)
    with pytest.raises(ld.LadduError, match='already exists'):
        artifact.save(path)
    artifact.save(path, overwrite=True)

    damaged = bytearray(path.read_bytes())
    damaged[-1] ^= 0xFF
    path.write_bytes(damaged)
    with pytest.raises(ld.LadduError, match='checksum'):
        ld.FitArtifact.load(path)

    path.write_bytes(b'LADDUFIT\n')
    with pytest.raises(ld.LadduError, match='truncated'):
        ld.FitArtifact.load(path)


def test_fit_artifact_rejects_malformed_covariance_shape(tmp_path: Path) -> None:
    artifact = likelihood().fit(terminators=[ld.ganesh.MaxSteps(100)]).artifact()
    path = tmp_path / 'malformed-shape.laddu'
    artifact.save(path)
    archive = path.read_bytes()
    header = len(b'LADDUFIT\n')
    manifest_length = int.from_bytes(archive[header : header + 8], 'little')
    start = header + 8
    manifest = json.loads(archive[start : start + manifest_length])
    next(entry for entry in manifest['payloads'] if entry['role'] == 'covariance')['shape'] = []
    encoded = json.dumps(manifest).encode()
    path.write_bytes(b'LADDUFIT\n' + len(encoded).to_bytes(8, 'little') + encoded + archive[start + manifest_length :])
    with pytest.raises(ld.LadduError, match='shape'):
        ld.FitArtifact.load(path)


def test_committed_v1_fit_artifact_fixture_remains_readable() -> None:
    artifact = ld.FitArtifact.load(Path(__file__).parent / 'fixtures' / 'fit_artifact_v1.laddu')
    assert artifact.schema_version == 1
    assert artifact.parameter_names == ['a']
    assert np.isfinite(artifact.objective)


def test_fit_artifact_round_trips_parameter_ensemble_identity(tmp_path: Path) -> None:
    artifact = likelihood().fit(terminators=[ld.ganesh.MaxSteps(100)]).artifact()
    ensemble = ld.Ensemble.from_arrays([[0.1], [0.2]], parameter_names=['a'], source_id=ENSEMBLE_SOURCE_ID)
    artifact = artifact.with_ensemble(ensemble)
    path = tmp_path / 'ensemble.laddu'

    artifact.save(path)
    restored = ld.FitArtifact.load(path)

    assert restored.ensemble is not None
    assert restored.ensemble.source_id == ENSEMBLE_SOURCE_ID
    assert restored.ensemble.draw_ids == [0, 1]
    assert restored.ensemble.draws == [[0.1], [0.2]]
    assert restored.ensemble.bootstrap_seed is None
    assert not restored.ensemble.requires_external_datasets
    rebuilt = restored.ensemble.to_ensemble()
    assert rebuilt.source_id == ensemble.source_id
    np.testing.assert_allclose(rebuilt.draws, ensemble.draws)


def test_artifact_reconstructs_yields_without_refitting(tmp_path: Path) -> None:
    target = likelihood()
    fit = target.fit(terminators=[ld.ganesh.MaxSteps(100)])
    generated = ld.Dataset.from_arrays(p4s={}, scalars={'x': [0.1, 0.3, 0.5, 0.9]}, weights=[1.0] * 4)
    ensemble = ld.Ensemble.from_arrays(
        [[float(fit.values[0])], [float(fit.values[0]) + 0.01]],
        parameter_names=['a'],
        source_id=73,
    )
    original = target.yield_context('signal', generated_mc=generated, parameters=fit.values, ensemble=ensemble)
    path = tmp_path / 'reconstruct.laddu'
    fit.artifact().with_ensemble(ensemble).save(path)

    reconstructed = ld.FitArtifact.load(path).yield_context(target, 'signal', generated_mc=generated)

    assert reconstructed.selected_yield().central == original.selected_yield().central
    assert reconstructed.corrected_observed_yield().central == pytest.approx(
        original.corrected_observed_yield().central
    )
    assert reconstructed.corrected_observed_yield().draws.tolist() == pytest.approx(
        original.corrected_observed_yield().draws.tolist()
    )


def test_explicit_parameter_migration_and_snapshot_identity(tmp_path: Path) -> None:
    source = likelihood()
    artifact = source.fit(terminators=[ld.ganesh.MaxSteps(1)]).artifact()
    data = ld.Dataset.from_arrays(p4s={}, scalars={'x': [0.2, 0.4]}, weights=[1.0, 1.0])
    x = ld.scalar('x')
    renamed = ld.Likelihood(
        [
            ld.NLL(
                ld.Model((ld.parameter('renamed', initial=0.5) * x + 1.0).norm_sqr()),
                data=data,
                accepted_mc=data,
                name='signal',
            )
        ]
    )
    bound = artifact.bind_with_parameter_map(renamed, {'a': 'renamed'})
    assert bound.parameter_names == ['renamed']
    assert bound.migration == [('a', 'renamed')]

    structurally_changed = ld.Likelihood(
        [
            ld.NLL(
                ld.Model((ld.parameter('renamed', initial=0.5) * x + 2.0).norm_sqr()),
                data=data,
                accepted_mc=data,
                name='signal',
            )
        ]
    )
    with pytest.raises(ld.LadduError, match='fingerprint'):
        artifact.bind_with_parameter_map(structurally_changed, {'a': 'renamed'})

    snapshot = ld.AnalysisSnapshot()
    snapshot.add_fit('fits/main', artifact)
    snapshot.alias_fit('fits/reference', 'fits/main')
    assert snapshot.fit('fits/main') is snapshot.fit('fits/reference')
    with pytest.raises(ld.LadduError, match='duplicate'):
        snapshot.add_fit('fits/main', artifact)
    assert snapshot.fit('fits/main') is snapshot.fit('fits/reference')

    path = tmp_path / 'analysis.laddu'
    snapshot.save(path)
    restored = ld.AnalysisSnapshot.load(path)
    assert restored.fit('fits/main') is restored.fit('fits/reference')
    np.testing.assert_allclose(restored.fit('fits/main').values, artifact.values)
    np.testing.assert_allclose(restored.fit('fits/main').bind(source).values, artifact.values)
    generated = ld.Dataset.from_arrays(p4s={}, scalars={'x': [0.1, 0.3, 0.5]}, weights=[1.0] * 3)
    embedded_yield = restored.fit('fits/main').yield_context(source, 'signal', generated_mc=generated)
    standalone_yield = artifact.yield_context(source, 'signal', generated_mc=generated)
    assert embedded_yield.selected_yield().central == standalone_yield.selected_yield().central
    damaged = bytearray(path.read_bytes())
    damaged[-1] ^= 0xFF
    path.write_bytes(damaged)
    with pytest.raises(ld.LadduError, match=r'fits/main.*checksum'):
        ld.AnalysisSnapshot.load(path)


def test_failed_replica_ids_preserve_gaps(tmp_path: Path) -> None:
    artifact = likelihood().fit(terminators=[ld.ganesh.MaxSteps(1)]).artifact()
    ensemble = ld.Ensemble.from_arrays([[0.1], [0.3]], parameter_names=['a'], source_id=12)
    artifact = artifact.with_ensemble(ensemble).with_failed_replicas([(1, 7, 'failed')])
    attached = artifact.ensemble
    assert attached is not None
    assert attached.draw_ids == [0, 2]
    path = tmp_path / 'failed-replica.laddu'
    artifact.save(path)
    restored = ld.FitArtifact.load(path)
    restored_ensemble = restored.ensemble
    assert restored_ensemble is not None
    assert restored_ensemble.draw_ids == [0, 2]
