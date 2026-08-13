mod adjoint;
mod emit;
mod rules;

use laddu_expr::{UnaryOp, parameters::ParamId};
use laddu_kernel::ir::{
    GradientKernelIr, KernelInstruction, KernelIrBuilder, KernelValueId, KernelValueKind,
    OutputComponent, ScalarKernelIr,
};

use self::adjoint::AdjointStore;
use crate::{AutodiffError, AutodiffResult};

/// Differentiates a scalar kernel with respect to free parameters.
///
/// # Errors
///
/// Returns [`AutodiffError`] when an instruction has no supported derivative
/// rule or the generated gradient kernel IR is invalid.
pub fn gradient_ir(
    primal: &ScalarKernelIr,
    free_params: &[ParamId],
    component: OutputComponent,
) -> AutodiffResult<GradientKernelIr> {
    ReverseState::new(primal, component).build(free_params)
}

struct ReverseState<'a> {
    pub(super) primal: &'a ScalarKernelIr,
    pub(super) builder: KernelIrBuilder,
    adjoints: AdjointStore,
    pub(super) component: OutputComponent,
}

impl<'a> ReverseState<'a> {
    fn new(primal: &'a ScalarKernelIr, component: OutputComponent) -> Self {
        Self {
            primal,
            builder: KernelIrBuilder::from_scalar(primal),
            adjoints: AdjointStore::new(primal.values()),
            component,
        }
    }

    fn build(mut self, free_params: &[ParamId]) -> AutodiffResult<GradientKernelIr> {
        let root = self.primal.root();
        self.seed_root_adjoint()?;

        for index in (0..self.primal.values().len()).rev() {
            let primal = KernelValueId::from_index(index);
            let Some(adjoint) = self.resolve_adjoint(primal)? else {
                continue;
            };
            self.propagate_instruction(primal, &adjoint)?;
            self.adjoints.set_resolved(primal, adjoint);
        }

        let outputs = self.collect_parameter_outputs(free_params)?;
        self.builder
            .finish_gradient(root, outputs, self.component)
            .map_err(Into::into)
    }

    fn seed_root_adjoint(&mut self) -> AutodiffResult<()> {
        let root = self.primal.root();
        let root_kind = self.primal.values()[root.index()].kind;
        let seed = match (root_kind, self.component) {
            (KernelValueKind::Real, OutputComponent::Real) => self.real(1.0)?,
            (KernelValueKind::Real, OutputComponent::Imag) => self.real(0.0)?,
            (KernelValueKind::Complex, OutputComponent::Real) => self.real(1.0)?,
            (KernelValueKind::Complex, OutputComponent::Imag) => self.complex(0.0, 1.0)?,
            _ => {
                return Err(AutodiffError::InvalidKernel(
                    "gradient root must be scalar".into(),
                ));
            }
        };
        self.adjoints.add_element(root, 0, seed)
    }

    fn collect_parameter_outputs(
        &mut self,
        free_params: &[ParamId],
    ) -> AutodiffResult<Vec<KernelValueId>> {
        let mut outputs = Vec::with_capacity(free_params.len());
        for parameter in free_params {
            let mut terms = Vec::new();
            for (index, value) in self.primal.values().iter().enumerate() {
                if matches!(value.instruction, KernelInstruction::Parameter(id) if id == *parameter)
                    && let Some(adjoint) = self.adjoints.resolved(KernelValueId::from_index(index))
                {
                    terms.push(self.unary(UnaryOp::Real, adjoint[0])?);
                }
            }
            outputs.push(self.sum_or_zero(terms)?);
        }
        Ok(outputs)
    }

    fn resolve_adjoint(
        &mut self,
        primal: KernelValueId,
    ) -> AutodiffResult<Option<Vec<KernelValueId>>> {
        let Some(pending) = self.adjoints.take_pending(primal) else {
            return Ok(None);
        };
        let mut values = Vec::with_capacity(pending.len());
        for terms in pending {
            values.push(self.sum_or_zero(terms)?);
        }
        Ok(Some(values))
    }

    fn propagate_instruction(
        &mut self,
        primal: KernelValueId,
        adjoint: &[KernelValueId],
    ) -> AutodiffResult<()> {
        let value = &self.primal.values()[primal.index()];
        match &value.instruction {
            KernelInstruction::Cached(_)
            | KernelInstruction::RealConstant(_)
            | KernelInstruction::ComplexConstant(_)
            | KernelInstruction::Parameter(_) => {}
            KernelInstruction::Unary { op, input } => {
                let contribution = self.unary_pullback(*op, *input, primal, adjoint[0])?;
                self.accumulate(*input, &[contribution])?;
            }
            KernelInstruction::Binary { op, lhs, rhs } => {
                let (lhs_contribution, rhs_contribution) =
                    self.binary_pullback(*op, *lhs, *rhs, adjoint[0])?;
                self.accumulate(*lhs, &[lhs_contribution])?;
                self.accumulate(*rhs, &[rhs_contribution])?;
            }
            KernelInstruction::Add(terms) => {
                for term in terms {
                    self.accumulate(*term, adjoint)?;
                }
            }
            KernelInstruction::Mul(factors) => self.product_pullback(factors, adjoint[0])?,
            KernelInstruction::Complex { re, im } => {
                self.complex_pullback(*re, *im, adjoint[0])?;
            }
            KernelInstruction::Vector(entries)
            | KernelInstruction::Matrix {
                elements: entries, ..
            } => {
                self.aggregate_pullback(entries, adjoint)?;
            }
            KernelInstruction::Component { input, index } => {
                self.accumulate_element(*input, *index, adjoint[0])?;
            }
            KernelInstruction::MatrixElement { input, row, col } => {
                let KernelValueKind::Matrix { cols, .. } = self.kind(*input) else {
                    unreachable!()
                };
                self.accumulate_element(*input, row * cols + col, adjoint[0])?;
            }
            KernelInstruction::MatMul { lhs, rhs } => {
                self.matmul_pullback(*lhs, *rhs, adjoint)?;
            }
            KernelInstruction::MatVec { matrix, vector } => {
                self.matvec_pullback(*matrix, *vector, adjoint)?;
            }
            KernelInstruction::Dot { lhs, rhs } => {
                self.dot_pullback(*lhs, *rhs, adjoint[0])?;
            }
            KernelInstruction::Solve { matrix, rhs } => {
                self.solve_pullback(*matrix, *rhs, primal, adjoint)?;
            }
            KernelInstruction::SolveRow { row_slot, rhs } => {
                self.solve_row_pullback(*row_slot, rhs, adjoint[0])?;
            }
            KernelInstruction::SolveRowAdjointElement { .. } => {
                return Err(AutodiffError::InvalidKernel(
                    "cannot differentiate derivative-only solve-row adjoint instruction".into(),
                ));
            }
        }
        Ok(())
    }

    fn accumulate(
        &mut self,
        target: KernelValueId,
        contribution: &[KernelValueId],
    ) -> AutodiffResult<()> {
        self.adjoints.add_value(target, contribution)
    }

    fn accumulate_element(
        &mut self,
        target: KernelValueId,
        element: usize,
        contribution: KernelValueId,
    ) -> AutodiffResult<()> {
        self.adjoints.add_element(target, element, contribution)
    }
}

#[cfg(test)]
mod tests {
    use laddu_expr::BinaryOp;
    use laddu_expr::parameters::{ParamRegistry, Parameter};
    use laddu_kernel::ir::{KernelValue, KernelValueClass};
    use num::complex::Complex64;

    use super::*;

    fn value(kind: KernelValueKind, instruction: KernelInstruction) -> KernelValue {
        KernelValue {
            kind,
            class: KernelValueClass::Invariant,
            instruction,
        }
    }

    fn event_value(kind: KernelValueKind, instruction: KernelInstruction) -> KernelValue {
        KernelValue {
            kind,
            class: KernelValueClass::Event,
            instruction,
        }
    }

    fn parameter(registry: &mut ParamRegistry, name: &str) -> ParamId {
        registry.register(Parameter::free(name)).unwrap()
    }

    fn fingerprint(ir: &GradientKernelIr) -> u64 {
        format!("{ir:?}")
            .bytes()
            .fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
                (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
            })
    }

    fn unary_fixture(op: UnaryOp) -> GradientKernelIr {
        let mut registry = ParamRegistry::new();
        let x = parameter(&mut registry, "x");
        let output_kind = match op {
            UnaryOp::Real | UnaryOp::Imag | UnaryOp::NormSqr => KernelValueKind::Real,
            _ => KernelValueKind::Complex,
        };
        let primal = ScalarKernelIr::new(
            vec![
                value(KernelValueKind::Real, KernelInstruction::Parameter(x)),
                value(
                    KernelValueKind::Real,
                    KernelInstruction::RealConstant(-0.25),
                ),
                value(
                    KernelValueKind::Complex,
                    KernelInstruction::Complex {
                        re: KernelValueId::from_index(0),
                        im: KernelValueId::from_index(1),
                    },
                ),
                value(
                    output_kind,
                    KernelInstruction::Unary {
                        op,
                        input: KernelValueId::from_index(2),
                    },
                ),
            ],
            KernelValueId::from_index(3),
        )
        .unwrap();
        gradient_ir(&primal, &[x], OutputComponent::Real).unwrap()
    }

    fn binary_fixture(op: BinaryOp) -> GradientKernelIr {
        let mut registry = ParamRegistry::new();
        let x = parameter(&mut registry, "x");
        let (values, root) = if op == BinaryOp::Atan2 {
            (
                vec![
                    value(KernelValueKind::Real, KernelInstruction::Parameter(x)),
                    value(KernelValueKind::Real, KernelInstruction::RealConstant(0.75)),
                    value(
                        KernelValueKind::Real,
                        KernelInstruction::Binary {
                            op,
                            lhs: KernelValueId::from_index(0),
                            rhs: KernelValueId::from_index(1),
                        },
                    ),
                ],
                KernelValueId::from_index(2),
            )
        } else {
            (
                vec![
                    value(KernelValueKind::Real, KernelInstruction::Parameter(x)),
                    value(
                        KernelValueKind::Real,
                        KernelInstruction::RealConstant(-0.25),
                    ),
                    value(
                        KernelValueKind::Complex,
                        KernelInstruction::Complex {
                            re: KernelValueId::from_index(0),
                            im: KernelValueId::from_index(1),
                        },
                    ),
                    value(
                        KernelValueKind::Complex,
                        KernelInstruction::ComplexConstant(Complex64::new(0.75, 0.5)),
                    ),
                    value(
                        KernelValueKind::Complex,
                        KernelInstruction::Binary {
                            op,
                            lhs: KernelValueId::from_index(2),
                            rhs: KernelValueId::from_index(3),
                        },
                    ),
                ],
                KernelValueId::from_index(4),
            )
        };
        let primal = ScalarKernelIr::new(values, root).unwrap();
        gradient_ir(&primal, &[x], OutputComponent::Real).unwrap()
    }

    fn product_fixture(factors: &[usize]) -> GradientKernelIr {
        let mut registry = ParamRegistry::new();
        let x = parameter(&mut registry, "x");
        let mut values = vec![
            value(KernelValueKind::Real, KernelInstruction::Parameter(x)),
            value(KernelValueKind::Real, KernelInstruction::RealConstant(0.0)),
            value(KernelValueKind::Real, KernelInstruction::RealConstant(2.0)),
        ];
        values.push(value(
            KernelValueKind::Real,
            KernelInstruction::Mul(
                factors
                    .iter()
                    .copied()
                    .map(KernelValueId::from_index)
                    .collect(),
            ),
        ));
        let primal = ScalarKernelIr::new(values, KernelValueId::from_index(3)).unwrap();
        gradient_ir(&primal, &[x], OutputComponent::Real).unwrap()
    }

    fn seed_fixture(complex_root: bool, component: OutputComponent) -> GradientKernelIr {
        let mut registry = ParamRegistry::new();
        let x = parameter(&mut registry, "x");
        let (values, root) = if complex_root {
            (
                vec![
                    value(KernelValueKind::Real, KernelInstruction::Parameter(x)),
                    value(KernelValueKind::Real, KernelInstruction::RealConstant(0.5)),
                    value(
                        KernelValueKind::Complex,
                        KernelInstruction::Complex {
                            re: KernelValueId::from_index(0),
                            im: KernelValueId::from_index(1),
                        },
                    ),
                ],
                KernelValueId::from_index(2),
            )
        } else {
            (
                vec![value(
                    KernelValueKind::Real,
                    KernelInstruction::Parameter(x),
                )],
                KernelValueId::from_index(0),
            )
        };
        let primal = ScalarKernelIr::new(values, root).unwrap();
        gradient_ir(&primal, &[x], component).unwrap()
    }

    fn structured_fixture() -> GradientKernelIr {
        let mut registry = ParamRegistry::new();
        let x = parameter(&mut registry, "x");
        let id = KernelValueId::from_index;
        let primal = ScalarKernelIr::new(
            vec![
                value(KernelValueKind::Real, KernelInstruction::Parameter(x)),
                value(KernelValueKind::Real, KernelInstruction::RealConstant(0.5)),
                value(
                    KernelValueKind::Matrix { rows: 2, cols: 2 },
                    KernelInstruction::Matrix {
                        rows: 2,
                        cols: 2,
                        elements: vec![id(0), id(1), id(1), id(0)],
                    },
                ),
                value(
                    KernelValueKind::Matrix { rows: 2, cols: 2 },
                    KernelInstruction::Matrix {
                        rows: 2,
                        cols: 2,
                        elements: vec![id(1), id(0), id(0), id(1)],
                    },
                ),
                value(
                    KernelValueKind::Matrix { rows: 2, cols: 2 },
                    KernelInstruction::MatMul {
                        lhs: id(2),
                        rhs: id(3),
                    },
                ),
                value(
                    KernelValueKind::Vector { len: 2 },
                    KernelInstruction::Vector(vec![id(0), id(1)]),
                ),
                value(
                    KernelValueKind::Vector { len: 2 },
                    KernelInstruction::MatVec {
                        matrix: id(4),
                        vector: id(5),
                    },
                ),
                value(
                    KernelValueKind::Vector { len: 2 },
                    KernelInstruction::Vector(vec![id(1), id(0)]),
                ),
                value(
                    KernelValueKind::Complex,
                    KernelInstruction::Dot {
                        lhs: id(6),
                        rhs: id(7),
                    },
                ),
            ],
            id(8),
        )
        .unwrap();
        gradient_ir(&primal, &[x], OutputComponent::Real).unwrap()
    }

    #[test]
    fn derivative_rule_ir_goldens_cover_every_scalar_rule() {
        let fixtures = [
            (
                "neg",
                fingerprint(&unary_fixture(UnaryOp::Neg)),
                0x9860_60da_2ea2_b6e9,
            ),
            (
                "real",
                fingerprint(&unary_fixture(UnaryOp::Real)),
                0x8c61_544e_6d88_ced9,
            ),
            (
                "imag",
                fingerprint(&unary_fixture(UnaryOp::Imag)),
                0x63c6_a006_836a_b18b,
            ),
            (
                "conj",
                fingerprint(&unary_fixture(UnaryOp::Conj)),
                0x4468_7f36_fa3e_a6b1,
            ),
            (
                "norm_sqr",
                fingerprint(&unary_fixture(UnaryOp::NormSqr)),
                0x2e07_78ad_eaf6_71d4,
            ),
            (
                "sqrt",
                fingerprint(&unary_fixture(UnaryOp::Sqrt)),
                0x92b6_1640_ec1a_a401,
            ),
            (
                "exp",
                fingerprint(&unary_fixture(UnaryOp::Exp)),
                0xeeea_5a8c_6227_a337,
            ),
            (
                "sin",
                fingerprint(&unary_fixture(UnaryOp::Sin)),
                0x8c00_c6fa_666a_6d3d,
            ),
            (
                "cos",
                fingerprint(&unary_fixture(UnaryOp::Cos)),
                0xa46f_725b_44cf_7460,
            ),
            (
                "log",
                fingerprint(&unary_fixture(UnaryOp::Log)),
                0xa4c4_b4b1_c239_ddfa,
            ),
            (
                "powi_zero",
                fingerprint(&unary_fixture(UnaryOp::PowI(0))),
                0x642e_b6f8_891e_1e04,
            ),
            (
                "powi_three",
                fingerprint(&unary_fixture(UnaryOp::PowI(3))),
                0xdcbf_dbc2_d727_e3fe,
            ),
            (
                "powi_min",
                fingerprint(&unary_fixture(UnaryOp::PowI(i32::MIN))),
                0xf454_847b_314e_cde2,
            ),
            (
                "add",
                fingerprint(&binary_fixture(BinaryOp::Add)),
                0xb3af_2d89_7508_039d,
            ),
            (
                "sub",
                fingerprint(&binary_fixture(BinaryOp::Sub)),
                0xdf01_86a3_8215_3021,
            ),
            (
                "mul",
                fingerprint(&binary_fixture(BinaryOp::Mul)),
                0x6d97_23e9_1609_93c3,
            ),
            (
                "div",
                fingerprint(&binary_fixture(BinaryOp::Div)),
                0xa899_3a7d_7abc_26c0,
            ),
            (
                "atan2",
                fingerprint(&binary_fixture(BinaryOp::Atan2)),
                0x8ee1_4f99_8134_7827,
            ),
        ];
        for (name, actual, expected) in fixtures {
            assert_eq!(actual, expected, "generated IR changed for {name}");
        }
    }

    #[test]
    fn product_rule_ir_goldens_cover_factor_edge_cases() {
        let fixtures = [
            (
                "one_factor",
                fingerprint(&product_fixture(&[0])),
                0xcf70_0613_7d79_51cb,
            ),
            (
                "repeated_factor",
                fingerprint(&product_fixture(&[0, 0])),
                0x52d6_fdee_17d0_c79c,
            ),
            (
                "multiple_with_zero",
                fingerprint(&product_fixture(&[0, 2, 1])),
                0xf7e4_600f_9869_cb91,
            ),
        ];
        for (name, actual, expected) in fixtures {
            assert_eq!(actual, expected, "generated IR changed for {name}");
        }
    }

    #[test]
    fn root_seed_ir_goldens_cover_kind_and_component_pairs() {
        let fixtures = [
            (
                "real_real",
                fingerprint(&seed_fixture(false, OutputComponent::Real)),
                0xff1d_9891_190c_d442,
            ),
            (
                "real_imag",
                fingerprint(&seed_fixture(false, OutputComponent::Imag)),
                0x940a_cee8_1e00_11c9,
            ),
            (
                "complex_real",
                fingerprint(&seed_fixture(true, OutputComponent::Real)),
                0x2ebe_0abb_850f_aad9,
            ),
            (
                "complex_imag",
                fingerprint(&seed_fixture(true, OutputComponent::Imag)),
                0x3e5f_2480_2743_1916,
            ),
        ];
        for (name, actual, expected) in fixtures {
            assert_eq!(actual, expected, "generated IR changed for {name}");
        }
    }

    #[test]
    fn structured_rule_ir_golden_covers_matrix_vector_and_dot_routing() {
        let actual = fingerprint(&structured_fixture());
        assert_eq!(
            actual, 0x4929_101d_3cd2_5fae,
            "generated structured-operation IR changed"
        );
    }

    #[test]
    fn empty_product_is_rejected_at_the_kernel_boundary() {
        let error = ScalarKernelIr::new(
            vec![value(
                KernelValueKind::Real,
                KernelInstruction::Mul(Vec::new()),
            )],
            KernelValueId::from_index(0),
        )
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "kernel value 0 has no operands for multiplication"
        );
    }

    #[test]
    fn requested_parameter_order_and_duplicates_are_preserved() {
        let mut registry = ParamRegistry::new();
        let x = parameter(&mut registry, "x");
        let y = parameter(&mut registry, "y");
        let primal = ScalarKernelIr::new(
            vec![
                value(KernelValueKind::Real, KernelInstruction::Parameter(x)),
                value(
                    KernelValueKind::Real,
                    KernelInstruction::Add(vec![
                        KernelValueId::from_index(0),
                        KernelValueId::from_index(0),
                    ]),
                ),
            ],
            KernelValueId::from_index(1),
        )
        .unwrap();

        let gradient = gradient_ir(&primal, &[y, x, x], OutputComponent::Real).unwrap();
        let outputs = gradient.outputs();

        assert_eq!(outputs.len(), 3);
        assert!(matches!(
            gradient.values()[outputs[0].index()].instruction,
            KernelInstruction::RealConstant(0.0)
        ));
        assert!(outputs[1].index() < outputs[2].index());
        assert!(gradient.values().iter().any(|value| {
            matches!(&value.instruction, KernelInstruction::Add(terms) if terms.len() == 2)
        }));
    }

    #[test]
    fn derivative_only_instruction_is_rejected_by_autodiff() {
        let primal = ScalarKernelIr::new(
            vec![
                value(KernelValueKind::Real, KernelInstruction::RealConstant(1.0)),
                event_value(
                    KernelValueKind::Complex,
                    KernelInstruction::SolveRowAdjointElement {
                        row_slot: 0,
                        index: 0,
                        len: 1,
                        adjoint: KernelValueId::from_index(0),
                    },
                ),
            ],
            KernelValueId::from_index(1),
        )
        .unwrap();

        let error = gradient_ir(&primal, &[], OutputComponent::Real).unwrap_err();
        assert_eq!(
            error.to_string(),
            "cannot differentiate kernel instruction: cannot differentiate derivative-only solve-row adjoint instruction"
        );
    }

    #[test]
    fn scalar_gradient_program_has_one_real_output_per_parameter() {
        let mut registry = ParamRegistry::new();
        let x = registry.register(Parameter::free("x")).unwrap();
        let primal = ScalarKernelIr::new(
            vec![
                value(KernelValueKind::Real, KernelInstruction::Parameter(x)),
                value(
                    KernelValueKind::Real,
                    KernelInstruction::Mul(vec![
                        KernelValueId::from_index(0),
                        KernelValueId::from_index(0),
                    ]),
                ),
            ],
            KernelValueId::from_index(1),
        )
        .unwrap();

        let gradient = gradient_ir(&primal, &[x], OutputComponent::Real).unwrap();

        assert_eq!(gradient.outputs().len(), 1);
        assert_eq!(
            gradient.values()[gradient.outputs()[0].index()].kind,
            KernelValueKind::Real
        );
        assert!(gradient.values().len() > primal.values().len());
    }

    #[test]
    fn solve_gradient_program_contains_adjoint_solve() {
        let mut registry = ParamRegistry::new();
        let x = registry.register(Parameter::free("x")).unwrap();
        let primal = ScalarKernelIr::new(
            vec![
                value(KernelValueKind::Real, KernelInstruction::Parameter(x)),
                value(
                    KernelValueKind::Complex,
                    KernelInstruction::ComplexConstant(Complex64::new(2.0, 0.5)),
                ),
                value(
                    KernelValueKind::Matrix { rows: 2, cols: 2 },
                    KernelInstruction::Matrix {
                        rows: 2,
                        cols: 2,
                        elements: vec![
                            KernelValueId::from_index(0),
                            KernelValueId::from_index(1),
                            KernelValueId::from_index(1),
                            KernelValueId::from_index(0),
                        ],
                    },
                ),
                value(
                    KernelValueKind::Vector { len: 2 },
                    KernelInstruction::Vector(vec![
                        KernelValueId::from_index(0),
                        KernelValueId::from_index(1),
                    ]),
                ),
                value(
                    KernelValueKind::Vector { len: 2 },
                    KernelInstruction::Solve {
                        matrix: KernelValueId::from_index(2),
                        rhs: KernelValueId::from_index(3),
                    },
                ),
                value(
                    KernelValueKind::Complex,
                    KernelInstruction::Component {
                        input: KernelValueId::from_index(4),
                        index: 0,
                    },
                ),
            ],
            KernelValueId::from_index(5),
        )
        .unwrap();

        let gradient = gradient_ir(&primal, &[x], OutputComponent::Imag).unwrap();
        let solves = gradient
            .values()
            .iter()
            .filter(|value| matches!(value.instruction, KernelInstruction::Solve { .. }))
            .count();

        assert_eq!(solves, 2);
        assert_eq!(gradient.component(), OutputComponent::Imag);
        let actual = fingerprint(&gradient);
        assert_eq!(actual, 0x3e77_bfe3_29e7_18a9, "generated solve IR changed");
    }

    #[test]
    fn solve_row_gradient_uses_elementwise_adjoint_loads() {
        let mut registry = ParamRegistry::new();
        let x = registry.register(Parameter::free("x")).unwrap();
        let primal = ScalarKernelIr::new(
            vec![
                value(KernelValueKind::Real, KernelInstruction::Parameter(x)),
                value(
                    KernelValueKind::Complex,
                    KernelInstruction::ComplexConstant(Complex64::new(2.0, 0.5)),
                ),
                event_value(
                    KernelValueKind::Complex,
                    KernelInstruction::SolveRow {
                        row_slot: 0,
                        rhs: vec![KernelValueId::from_index(0), KernelValueId::from_index(1)],
                    },
                ),
            ],
            KernelValueId::from_index(2),
        )
        .unwrap();

        let gradient = gradient_ir(&primal, &[x], OutputComponent::Real).unwrap();
        let adjoint_elements = gradient
            .values()
            .iter()
            .filter(|value| {
                matches!(
                    value.instruction,
                    KernelInstruction::SolveRowAdjointElement { len: 2, .. }
                )
            })
            .count();
        let generated_solve_rows = gradient.values()[primal.values().len()..]
            .iter()
            .filter(|value| matches!(value.instruction, KernelInstruction::SolveRow { .. }))
            .count();

        assert_eq!(adjoint_elements, 2);
        assert_eq!(generated_solve_rows, 0);
        let actual = fingerprint(&gradient);
        assert_eq!(
            actual, 0x5b93_b1fb_2607_49d5,
            "generated solve-row IR changed"
        );
    }
}
