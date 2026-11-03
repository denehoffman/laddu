# ruff: noqa: S101, PLR2004 -- explicit numerical test expectations
"""Physical-group Poisson draws stay paired through native datasets and fits."""

import laddu as ld
import numpy as np
import pytest


def dataset(order=None, namespace='period', groups=('run', 'event')):
    keys = np.repeat(np.arange(40, dtype=np.uint64) + 2**54, 2)
    weights = np.tile([1.0, -0.25], 40)
    if order is None:
        order = np.arange(len(keys))
    return ld.Dataset.from_arrays(
        p4s={},
        scalars={'x': np.linspace(0.1, 0.9, len(keys))[order]},
        columns={'run': np.ones(len(keys), dtype=np.uint32)[order], 'event': keys[order]},
        weights=weights[order],
        bootstrap_groups=groups,
        bootstrap_namespace=namespace,
    )


def test_groups_share_one_multiplier_and_preserve_exact_keys_and_signed_weights():
    source = dataset()
    original = np.asarray(source.weights())
    # Stable version-one public draw convention, including adjacent keys above 2**54.
    np.testing.assert_array_equal(
        source.bootstrap(seed=7).weights()[:12], [1.0, -0.25, 4.0, -1.0, 3.0, -0.75, 1.0, -0.25, 0.0, 0.0, 0.0, 0.0]
    )
    for seed in (0, 7, 42, 2**64 - 1):
        replica = np.asarray(source.bootstrap(seed=seed).weights())
        multiplier = replica / original
        np.testing.assert_array_equal(multiplier[::2], multiplier[1::2])
        assert len(np.unique(multiplier[::2])) > 1
        reverse = dataset(np.arange(79, -1, -1)).bootstrap(seed=seed)
        np.testing.assert_array_equal(reverse.weights(), replica[::-1])
    np.testing.assert_array_equal(source.weights(), original)
    np.testing.assert_array_equal(source.column('event'), np.repeat(np.arange(40, dtype=np.uint64) + 2**54, 2))


def test_groups_pair_across_filters_and_separate_namespaces():
    source = dataset()
    filtered = source.select(ld.scalar('x') > 0.4).bootstrap(seed=73)
    mask = np.linspace(0.1, 0.9, 80) > 0.4
    np.testing.assert_array_equal(filtered.weights(), np.asarray(source.bootstrap(seed=73).weights())[mask])
    assert not np.array_equal(
        source.bootstrap(seed=73).weights(), dataset(namespace='other-period').bootstrap(seed=73).weights()
    )
    for keys in ([], ['event', 'event'], ['missing'], ['x']):
        with pytest.raises((ValueError, ld.LadduError)):
            dataset(groups=keys)


def term(name):
    data = dataset(namespace=name)
    mc = ld.Dataset.from_arrays(p4s={}, scalars={'x': np.linspace(0, 1, 100)}, weights=np.ones(100))
    scale = ld.parameter(f'{name}_scale', initial=0.6, bounds=(0.01, 10.0))
    return ld.ExtendedNLL(ld.Model(scale.norm_sqr()), data=data, accepted_mc=mc, name=name), mc


def observed(likelihood):
    return {d.term: d.sum_weights for d in likelihood.diagnostics().datasets if d.role == 'observed'}


def test_grouped_likelihood_replicas_ignore_term_order_and_keep_mc_fixed():
    first, _mc = term('first')
    second, _ = term('second')
    direct = ld.Likelihood([first, second])
    reordered = ld.Likelihood([second, first])
    for seed in (7, 42):
        replica = direct.bootstrap(seed=seed)
        assert observed(replica) == observed(reordered.bootstrap(seed=seed))
        assert observed(replica) == {
            name: dataset(namespace=name).bootstrap(seed=seed).sum_weights() for name in ('first', 'second')
        }
        assert all(d.sum_weights == 100.0 for d in replica.diagnostics().datasets if d.role != 'observed')
    for name in ('first', 'second'):
        single, _mc = term(name)
        assert observed(ld.Likelihood([single]).bootstrap(seed=42)) == {name: observed(direct.bootstrap(seed=42))[name]}
    assert observed(direct) == {'first': 30.0, 'second': 30.0}


def test_native_bootstrap_fit_uses_configured_groups():
    signal, mc = term('signal')
    likelihood = ld.Likelihood([signal])
    fitted = likelihood.fit(default_callbacks=False, terminators=[ld.ganesh.LBFGSBInfNormGTerminator(eps_abs=1e-7)])
    ensemble = likelihood.bootstrap_fit(
        3,
        initial=fitted.values,
        seed=37,
        default_callbacks=False,
        terminators=[ld.ganesh.LBFGSBInfNormGTerminator(eps_abs=1e-7)],
    )
    for draw, values in enumerate(ensemble.draws):
        integral = likelihood.intensity_integrals('signal', generated_mc=mc).accepted_integral(values)
        expected = dataset(namespace='signal').bootstrap(seed=37 + draw).sum_weights()
        assert integral == pytest.approx(expected, rel=1e-7)


def test_bootstrap_fit_optionally_returns_convergence_results_in_draw_order():
    signal, _ = term('signal')
    likelihood = ld.Likelihood([signal])
    ensemble, fits = likelihood.bootstrap_fit(
        3,
        seed=37,
        return_fits=True,
        default_callbacks=False,
        terminators=[ld.ganesh.MaxSteps(1)],
    )
    assert len(fits) == len(ensemble) == 3
    assert any(not fit.converged for fit in fits)
    for values, fit in zip(ensemble.draws, fits, strict=True):
        np.testing.assert_array_equal(values, fit.values)
        assert fit.parameter_names == ensemble.parameter_names


def test_constructor_keywords_work_for_streams_and_file_sources(tmp_path):
    source = dataset()
    expected = source.bootstrap(seed=7).weights()

    def batches(**_kwargs):
        for start in range(0, 80, 13):
            section = slice(start, start + 13)
            yield {
                'p4s': {},
                'scalars': {'x': source.column('x')[section]},
                'columns': {name: source.column(name)[section] for name in ('run', 'event')},
                'weights': source.weights()[section],
            }

    streamed = ld.Dataset.from_batches(
        batches,
        schema={'p4s': [], 'scalars': ['x'], 'columns': {'run': 'u32', 'event': 'u64'}, 'weights': True},
        cache='streaming',
        bootstrap_groups=['run', 'event'],
        bootstrap_namespace='period',
    )
    np.testing.assert_array_equal(streamed.bootstrap(seed=7).weights(), expected)
    for suffix, sink, reader, file_source in (
        ('parquet', ld.ParquetSink, ld.read_parquet, ld.ParquetSource),
        ('root', ld.RootSink, ld.read_root, ld.RootSource),
    ):
        path = tmp_path / f'events.{suffix}'
        source.write_to(sink(path))
        np.testing.assert_array_equal(
            reader(path, bootstrap_groups=['run', 'event'], bootstrap_namespace='period').bootstrap(seed=7).weights(),
            expected,
        )
        np.testing.assert_array_equal(
            ld.Dataset(file_source(path), bootstrap_groups=['run', 'event'], bootstrap_namespace='period')
            .bootstrap(seed=7)
            .weights(),
            expected,
        )


def test_constructor_validation_and_default_row_resampling():
    with pytest.raises(ValueError, match='requires bootstrap_groups'):
        dataset(groups=None)
    ungrouped = dataset(groups=None, namespace='')
    assert not np.array_equal(ungrouped.bootstrap(seed=7).weights(), dataset().bootstrap(seed=7).weights())


def test_zero_group_multiplicity_is_absent_before_log_intensity_validation():
    data = ld.Dataset.from_arrays(
        p4s={},
        scalars={'x': [0.0, 1.0]},
        columns={'event': np.array([0, 1], dtype=np.uint64)},
        bootstrap_groups=['event'],
    )
    mc = ld.Dataset.from_arrays(p4s={}, scalars={'x': [1.0, 2.0]})
    model = ld.Model((ld.parameter('a', initial=1.0) * ld.scalar('x')).norm_sqr())
    target = ld.Likelihood([ld.ExtendedNLL(model, data=data, accepted_mc=mc, name='signal')])
    np.testing.assert_array_equal(data.bootstrap(seed=2).weights(), [0.0, 2.0])
    replica = target.bootstrap(seed=2)
    value, gradient = replica.value_and_gradient([1.0])
    assert value == pytest.approx(10.0)
    np.testing.assert_allclose(gradient, [12.0])
    assert [d.global_events for d in replica.diagnostics().datasets if d.role == 'observed'] == [1]
