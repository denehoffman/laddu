import tempfile

# ruff: noqa: PT009, PT027
import unittest
from pathlib import Path

import laddu as ld


class VisualizationTests(unittest.TestCase):
    def test_custom_rules_apply_to_all_renderers(self) -> None:
        expression = ld.scalar('x').named('debug')
        rule = ld.NodeStyleRule(
            ld.NodeSelector.name('debug'),
            ld.NodeStyle(
                foreground=ld.DisplayColor(1, 2, 3),
                fill=ld.DisplayColor(4, 5, 6),
                border=ld.DisplayColor(7, 8, 9),
            ),
        )

        equation = expression.equation(style_rules=[rule])
        latex = expression.latex(style_rules=[rule])
        tree = expression.tree(style_rules=[rule])
        dot = expression.dot(style_rules=[rule])
        with tempfile.TemporaryDirectory() as directory:
            svg_path = Path(directory) / 'expression.svg'
            result = expression.svg(svg_path, style_rules=[rule])
            svg = svg_path.read_text()
            string_path = Path(directory) / 'expression-from-string.svg'
            string_result = expression.svg(str(string_path))
            string_svg = string_path.read_text()

        self.assertIn('\x1b[38;2;1;2;3m', equation)
        self.assertIn(r'\color[RGB]{1,2,3}', latex)
        self.assertIn('\x1b[48;2;4;5;6m', tree)
        self.assertIn('fontcolor="#010203"', dot)
        self.assertIn('fillcolor="#040506"', dot)
        self.assertIn('color="#070809"', dot)
        self.assertIsNone(result)
        self.assertIsNone(string_result)
        self.assertIn('<svg', svg)
        self.assertIn('</svg>', svg)
        self.assertIn('<svg', string_svg)

    def test_presets_and_expand_repeated_flag(self) -> None:
        shared = ld.scalar('x')
        expression = (shared + 1.0) * (shared + 2.0)

        equation = expression.equation(colors='dark')
        latex = expression.latex(colors='dark')
        tree = expression.tree(
            colors='light',
            expand_repeated=False,
        )

        self.assertIn('\x1b[38;2;', equation)
        self.assertIn(r'\color[RGB]', latex)
        self.assertIn('<reference to #', tree)
        with self.assertRaises(ValueError):
            expression.equation(colors='sepia')

    def test_latex_is_a_math_mode_fragment(self) -> None:
        parameter = ld.parameter('alpha_internal', latex=r'\alpha')
        expression = parameter * ld.scalar('x_value') / ld.scalar('y').sqrt()
        latex = expression.latex()

        self.assertIn(r'\frac{', latex)
        self.assertIn(r'\alpha', latex)
        self.assertNotIn('alpha_internal', latex)
        self.assertIn(r'x\_value', latex)
        self.assertIn(r'\sqrt{y}', latex)
        self.assertNotIn(r'\color', latex)


if __name__ == '__main__':
    unittest.main()
