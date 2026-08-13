# ruff: noqa: PT027, S101

import unittest

import laddu as ld

PION_MASS = 0.1349768


class ParticleConstructionTests(unittest.TestCase):
    def test_constructor_applies_related_properties_as_one_validated_update(self) -> None:
        particle = ld.Particle(
            'pi0',
            species='pi0',
            self_conjugate=True,
            spin=0,
            c_parity=ld.Parity.POSITIVE,
            statistics=ld.Statistics.BOSON,
            mass=PION_MASS,
            ids={'pdg': 111},
        )

        assert particle.species == 'pi0'
        assert particle.antiparticle_species == 'pi0'
        assert particle.self_conjugate is True
        assert particle.charge == 0
        assert particle.statistics == ld.Statistics.BOSON
        assert particle.mass == PION_MASS
        assert particle.ids == {'pdg': 111}

    def test_constructor_rejects_inconsistent_invariants(self) -> None:
        with self.assertRaises(ld.LadduError):
            ld.Particle(self_conjugate=True, charge=1)

        with self.assertRaises(ld.LadduError):
            ld.Particle(spin=0, statistics=ld.Statistics.FERMION)

    def test_constructor_rejects_invalid_mass(self) -> None:
        for mass in (-1.0, float('inf'), float('nan')):
            with (
                self.subTest(mass=mass),
                self.assertRaisesRegex(
                    ValueError,
                    'particle mass must be finite and non-negative',
                ),
            ):
                ld.Particle(mass=mass)


if __name__ == '__main__':
    unittest.main()
