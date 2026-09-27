"""The public Python cross section follows the fitted generated intensity."""

# ruff: noqa: PT009, PT027, S101

import unittest

import laddu as ld
import numpy as np


def dataset(values: list[float], weights: list[float]) -> ld.Dataset:
    return ld.Dataset.from_arrays(p4s={}, scalars={'x': np.asarray(values)}, weights=np.asarray(weights))


class FittedCrossSectionTests(unittest.TestCase):
    def test_low_acceptance_reports_fitted_rate_without_data_anchoring(self) -> None:
        model = ld.Model((ld.scalar('x') + 1.0).norm_sqr())
        data = dataset([0.0], [1.0])
        accepted = dataset([0.0], [1.0])
        generated = dataset([0.0, 1.0, 2.0], [1.0, 1.0, 1.0])
        likelihood = ld.Likelihood([ld.ExtendedNLL(model, data=data, accepted_mc=accepted, name='signal')])
        section = likelihood.cross_section('signal', generated, ld.Luminosity(1.0, ld.AreaUnit.NANOBARN), [])
        self.assertAlmostEqual(section.total.central, 14.0)
        accepted_integral = section.accepted_integral
        generated_integral = section.generated_integral
        data_yield = section.data_yield
        closure = section.rate_closure
        assert accepted_integral is not None
        assert generated_integral is not None
        assert data_yield is not None
        assert closure is not None
        self.assertAlmostEqual(accepted_integral.central, 1.0)
        self.assertAlmostEqual(generated_integral.central, 14.0)
        self.assertAlmostEqual(data_yield.central, 1.0)
        self.assertEqual(closure.status, 'closed')
        self.assertGreater(section.total.central, 3.0)  # Flat-MC D / (1/3).
        integrals = likelihood.intensity_integrals('signal', generated_mc=generated)
        self.assertFalse(hasattr(integrals, 'acceptance'))
        self.assertFalse(hasattr(integrals, 'full_accepted_integral'))

        axis = ld.Axis(ld.scalar('x'), edges=[-0.5, 0.5, 2.5])
        projected = section.project([axis])
        np.testing.assert_allclose(projected.total.central, [1.0, 6.5])

    def test_shape_only_is_rejected(self) -> None:
        model = ld.Model(ld.scalar('x') + 1.0)
        data = dataset([0.0], [1.0])
        likelihood = ld.Likelihood([ld.NLL(model, data=data, accepted_mc=data, name='shape')])
        with self.assertRaises(ld.LadduError):
            likelihood.cross_section('shape', data, ld.Luminosity(1.0, ld.AreaUnit.NANOBARN), [])

    def test_only_new_cross_section_result_types_are_exported(self) -> None:
        self.assertTrue(hasattr(ld, 'CrossSection'))
        self.assertTrue(hasattr(ld, 'CrossSectionProjection'))
        for legacy in (
            'YieldCrossSection',
            'YieldCrossSectionProjection',
            'ReferenceCrossSection',
            'ExposureCombinedCrossSection',
            'ReferenceCorrectedYield',
        ):
            self.assertFalse(hasattr(ld, legacy), legacy)

    def test_physical_period_combination(self) -> None:
        model = ld.Model((ld.scalar('x') + 1.0).norm_sqr())
        data = dataset([0.0], [1.0])
        generated = dataset([0.0, 1.0], [1.0, 1.0])
        likelihood = ld.Likelihood([ld.ExtendedNLL(model, data=data, accepted_mc=data, name='signal')])
        first = likelihood.cross_section('signal', generated, ld.Luminosity(2.0, ld.AreaUnit.NANOBARN), [])
        second = likelihood.cross_section('signal', generated, ld.Luminosity(4.0, ld.AreaUnit.NANOBARN), [])
        combined = ld.CrossSection.combine([first, second])
        self.assertAlmostEqual(combined.total_effective_exposure, 6.0)
        self.assertAlmostEqual(combined.total.central, 10.0 / 6.0)
        self.assertFalse(hasattr(ld, 'ExposureFactor'))
        self.assertFalse(hasattr(ld.CrossSection, 'combine_with_factors'))

        first_factor = ld.Estimate(1.0, standard_error=0.1)
        self.assertAlmostEqual(first_factor.std(), 0.1)
        self.assertAlmostEqual((first_factor * 2.0).std(), 0.2)
        known = ld.CrossSection.combine(
            [
                (first, first_factor),
                (second, ld.Estimate(0.5, standard_error=0.1)),
            ]
        )
        self.assertAlmostEqual(known.total.central, 2.5)
        error = known.total.error(
            data_fill=False,
            accepted_mc_fill=False,
            generated_mc_fill=False,
            branching_exposure=True,
        )
        self.assertAlmostEqual(error.error**2, 0.078125)
        covariance = ld.CrossSection.combine(
            [(first, 1.0), (second, 0.5)],
            factor_covariance=[[0.01, 0.005], [0.005, 0.04]],
        )
        self.assertAlmostEqual(covariance.total.central, 2.5)
        with self.assertRaises(ld.LadduError):
            ld.CrossSection.combine(
                [(first, 1.0), (second, 0.5)],
                factor_covariance=[[1.0, 2.0], [2.0, 1.0]],
            )

    def test_fit_artifact_evaluates_the_same_fitted_cross_section(self) -> None:
        model = ld.Model(1.0 + ld.parameter('scale', initial=0.1).norm_sqr())
        data = dataset([0.0], [1.0])
        generated = dataset([0.0, 1.0], [1.0, 1.0])
        likelihood = ld.Likelihood([ld.ExtendedNLL(model, data=data, accepted_mc=data, name='signal')])
        fit = likelihood.fit(initial=[0.1], terminators=[ld.ganesh.MaxSteps(30)])
        luminosity = ld.Luminosity(2.0, ld.AreaUnit.NANOBARN)
        direct = likelihood.cross_section('signal', generated, luminosity, fit.values)
        restored = fit.artifact().cross_section(likelihood, 'signal', generated_mc=generated, luminosity=luminosity)
        self.assertAlmostEqual(restored.total.central, direct.total.central)


if __name__ == '__main__':
    unittest.main()
