# ruff: noqa: S101
"""The public objective is -2 log L; statistical inference keeps its scale."""

from pathlib import Path

import laddu as ld
import numpy as np
import pytest


def target(kind, *, penalties=False):
    a = ld.parameter('a', initial=0.7)
    model = ld.Model((ld.scalar('x') + a).norm_sqr())
    data = ld.Dataset.from_arrays(p4s={}, scalars={'x': [0.3, 1.1]}, weights=[1.0, 2.0])
    accepted = ld.Dataset.from_arrays(p4s={}, scalars={'x': [0.5, 1.7]}, weights=[1.5, 0.8])
    terms = [getattr(ld, kind)(model, data=data, accepted_mc=accepted, name='signal')]
    if penalties:
        terms.extend([ld.RidgePenalty(['a'], lambda_=2.0), ld.LassoPenalty(['a'], lambda_=3.0)])
    return ld.Likelihood(terms)


@pytest.mark.parametrize('kind', ['NLL', 'ExtendedNLL'])
@pytest.mark.parametrize('penalties', [False, True])
def test_values_and_gradients_use_minus_two_log_likelihood(kind, penalties):
    likelihood = target(kind, penalties=penalties)
    normalization = 1.5 * 1.2**2 + 0.8 * 2.4**2
    normalization_gradient = 2.0 * (1.5 * 1.2 + 0.8 * 2.4)
    log_sum = 2.0 * np.log(1.8**2)
    log_gradient = 2.0 * (1.0 + 2.0 / 1.8)
    if kind == 'NLL':
        expected = 2.0 * (3.0 * np.log(normalization) - log_sum)
        expected_gradient = 2.0 * (3.0 * normalization_gradient / normalization - log_gradient)
    else:
        expected = 2.0 * (normalization - log_sum)
        expected_gradient = 2.0 * (normalization_gradient - log_gradient)
    if penalties:
        expected += 2.0 * (2.0 * 0.7**2 + 3.0 * 0.7)
        expected_gradient += 2.0 * (4.0 * 0.7 + 3.0)
    value, gradient = likelihood.value_and_gradient([0.7])
    assert value == pytest.approx(expected)
    assert likelihood.value([0.7]) == pytest.approx(expected)
    assert likelihood.nll([0.7]) == pytest.approx(expected)
    np.testing.assert_allclose(gradient, [expected_gradient])
    stochastic = likelihood.fit_stochastic(
        initial=[0.7],
        fraction=1.0,
        default_callbacks=False,
        terminators=[ld.ganesh.MaxSteps(1)],
    )
    assert stochastic.fx == pytest.approx(likelihood.value(stochastic.x))


def poisson_rate_target():
    data = ld.Dataset.from_arrays(p4s={}, scalars={}, weights=[1.0] * 25)
    accepted = ld.Dataset.from_arrays(p4s={}, scalars={}, weights=[1.0])
    model = ld.Model(ld.parameter('rate', initial=20.0, bounds=(0.0, None)))
    return ld.Likelihood([ld.ExtendedNLL(model, data=data, accepted_mc=accepted, name='rate')])


def test_poisson_rate_fit_keeps_physical_covariance():
    likelihood = poisson_rate_target()
    fit = likelihood.fit(terminators=[ld.ganesh.MaxSteps(100)])
    np.testing.assert_allclose(fit.values, [25.0], rtol=1e-4)
    assert fit.objective == pytest.approx(likelihood.value(fit.values))
    # Q(rate) = 2(rate - N log(rate)); Var(rate) = 2 / Q'' = N.
    np.testing.assert_allclose(fit.covariance, [[25.0]], rtol=1e-3)
    np.testing.assert_allclose(fit.standard_errors, [5.0], rtol=1e-3)
    np.testing.assert_allclose(fit.covariance, 2.0 * fit.raw_ganesh_summary.covariance)


@pytest.mark.parametrize('sampler', ['AIES', 'ESS'])
def test_sampling_preserves_the_poisson_rate_posterior(sampler):
    likelihood = poisson_rate_target()
    init = getattr(ld.ganesh, sampler + 'Init')([[18.0], [22.0], [28.0], [32.0]])
    summary = likelihood.sample(init, seed=1729, default_callbacks=False, terminators=[ld.ganesh.MaxSteps(2000)])
    draws = summary.chain[:, 400:, 0].ravel()
    # Flat rate prior: rate | N ~ Gamma(N + 1, 1), with mean and variance 26.
    assert np.mean(draws) == pytest.approx(26.0, abs=0.8)
    assert np.var(draws) == pytest.approx(26.0, abs=4.0)


def test_legacy_fit_artifact_cannot_bind_across_objective_conventions():
    artifact = ld.FitArtifact.load(Path(__file__).parent / 'fixtures' / 'fit_artifact_v1.laddu')
    # This fixture has the former -log L identity and is still readable.
    from test_fit_result import likelihood  # noqa: PLC0415 -- reuse the exact fixture model

    with pytest.raises(ld.LadduError, match='fingerprint'):
        artifact.bind(likelihood())
