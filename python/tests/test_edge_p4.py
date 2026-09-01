# ruff: noqa: PT027, S101

import unittest

import laddu as ld
import numpy as np


class EdgeFourVectorTests(unittest.TestCase):
    def test_edge_preserves_event_column_input(self) -> None:
        edge = ld.Edge('beam', p4='beam')

        assert edge.p4 == 'beam'
        assert 'p4="beam"' in repr(edge)

        channel = ld.Channel('legacy', edges=[edge], vertices=[])
        restored = ld.Channel.from_json(channel.to_json())
        assert restored.edge_p4('beam') == 'beam'

    def test_edge_accepts_and_preserves_vec4_input(self) -> None:
        expression = ld.Vec4(5.0, 1.0, 2.0, 3.0)
        edge = ld.Edge('target', p4=expression)

        assert isinstance(edge.p4, ld.Vec4)
        np.testing.assert_allclose(
            [
                ld.Model(edge.p4.e()).evaluate(real=True),
                ld.Model(edge.p4.px()).evaluate(real=True),
            ],
            [5.0, 1.0],
            rtol=0,
            atol=1e-12,
        )

    def test_edge_rejects_unsupported_p4_input(self) -> None:
        with self.assertRaisesRegex(TypeError, 'expected p4 to be a str or Vec4'):
            ld.Edge('target', p4=1.0)  # ty: ignore[invalid-argument-type]

    def test_composed_vec4_is_used_by_channel_and_survives_round_trip(self) -> None:
        expression = ld.Vec4.event('beam') + ld.Vec4(1.0, 0.0, 0.0, 0.0)
        channel = ld.Channel('composed', edges=[ld.Edge('target', p4=expression)], vertices=[])
        restored = ld.Channel.from_json(channel.to_json())
        assert isinstance(restored.edge_p4('target'), ld.Vec4)
        dataset = ld.Dataset.from_arrays(
            p4s={'beam': np.array([[5.0, 0.0, 0.0, 4.0]])},
            scalars={},
        )

        expected = np.sqrt(20.0)
        np.testing.assert_allclose(
            ld.Model(channel.mass('target')).evaluate(dataset, real=True),
            [expected],
            rtol=0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            ld.Model(restored.mass('target')).evaluate(dataset, real=True),
            [expected],
            rtol=0,
            atol=1e-12,
        )
        missing = ld.Dataset.from_arrays(p4s={}, scalars={'x': np.array([0.0], dtype=np.float64)})
        with self.assertRaisesRegex(ld.LadduError, 'beam'):
            ld.Model(restored.mass('target')).evaluate(missing, real=True)


if __name__ == '__main__':
    unittest.main()
