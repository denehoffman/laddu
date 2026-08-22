use std::sync::{Arc, OnceLock};

use laddu_autodiff::{AutodiffMode, AutodiffPlan, AutodiffResult, gradient_ir};
use laddu_compile::CompiledModel;
use laddu_compile::ExecutablePlan;
use laddu_expr::parameters::ParamLayout;
use laddu_kernel::ir::{
    GradientKernelIr, KernelInstruction, KernelValueKind, OutputComponent, ScalarKernelIr,
};

use super::{
    CpuBackend, CpuExecutionMode, CpuPlan, GradientInterpreter, Precision, ScalarExecutor,
};

#[cfg(feature = "jit")]
use crate::jit::{JitGradientKernel, JitPrecision};

#[derive(Clone, Debug)]
pub(super) enum GradientExecutor {
    Interpreter(Option<GradientInterpreter>),
    #[cfg(feature = "jit")]
    Jit(JitGradientKernel),
}

impl GradientExecutor {
    fn prepare(
        plan: Option<&ScalarKernelIr>,
        params: &ParamLayout,
        mode: CpuExecutionMode,
        precision: Precision,
        gradient_ir: Option<(&GradientKernelIr, Option<&GradientKernelIr>)>,
    ) -> AutodiffResult<Self> {
        #[cfg(not(feature = "jit"))]
        let _ = (plan, params, mode, precision, gradient_ir);
        #[cfg(feature = "jit")]
        if mode == CpuExecutionMode::Auto
            && let Some(plan) = plan
            && let Ok(kernel) = (if let Some((real, imag)) = gradient_ir {
                JitGradientKernel::compile_gradient_ir(real, imag, JitPrecision::F32)
            } else {
                JitGradientKernel::compile_with_precision(
                    plan,
                    params.free_params(),
                    match precision {
                        Precision::F32 => JitPrecision::F32,
                        Precision::Auto | Precision::F64 => JitPrecision::F64,
                    },
                )
                .and_then(|kernel| kernel.ok_or_else(|| "missing gradient kernel".into()))
            })
        {
            return Ok(Self::Jit(kernel));
        }
        Ok(Self::Interpreter(
            plan.map(|plan| GradientInterpreter::new(plan, params.free_params()))
                .transpose()?,
        ))
    }
}
impl CpuPlan {
    pub(super) fn supports_f32_scalar_execution(&self) -> bool {
        self.scalar_kernel.as_ref().is_some_and(|kernel| {
            kernel.values().iter().all(|value| {
                matches!(value.kind, KernelValueKind::Real | KernelValueKind::Complex)
                    || !matches!(
                        value.instruction,
                        KernelInstruction::SolveRowAdjointElement { .. }
                    )
            })
        })
    }
}

impl CpuBackend {
    pub(super) fn prepare_with_modes_precision(
        &self,
        model: &CompiledModel,
        autodiff_mode: AutodiffMode,
        execution_mode: CpuExecutionMode,
        precision: Precision,
    ) -> AutodiffResult<CpuPlan> {
        let executable = ExecutablePlan::from_model(model)
            .map_err(|error| laddu_autodiff::AutodiffError::InvalidKernel(error.to_string()))?;
        let scalar_kernel = executable.scalar_kernel().cloned();
        let scalar_executor = scalar_kernel
            .as_ref()
            .and_then(|kernel| ScalarExecutor::prepare(kernel, execution_mode, precision));
        let f32_gradient_fallback_real = if precision == Precision::F32 {
            scalar_kernel
                .as_ref()
                .map(|kernel| {
                    gradient_ir(kernel, model.params().free_params(), OutputComponent::Real)
                })
                .transpose()?
        } else {
            None
        };
        let f32_gradient_fallback_imag = if precision == Precision::F32
            && scalar_kernel.as_ref().is_some_and(|kernel| {
                kernel.values()[kernel.root().index()].kind == KernelValueKind::Complex
            }) {
            scalar_kernel
                .as_ref()
                .map(|kernel| {
                    gradient_ir(kernel, model.params().free_params(), OutputComponent::Imag)
                })
                .transpose()?
        } else {
            None
        };
        let gradient_executor = GradientExecutor::prepare(
            scalar_kernel.as_ref(),
            model.params(),
            execution_mode,
            precision,
            f32_gradient_fallback_real
                .as_ref()
                .map(|real| (real, f32_gradient_fallback_imag.as_ref())),
        )?;
        let constant_factors = executable
            .constant_factor_matrices()
            .iter()
            .map(|_| Arc::new(OnceLock::new()))
            .collect();
        Ok(CpuPlan {
            precision,
            graph: executable.graph().clone(),
            params: executable.params().clone(),
            parameter_slots: executable.parameter_slots().to_vec(),
            autodiff: AutodiffPlan::from_model(model, autodiff_mode)?,
            cache_plan: executable.cache_plan().clone(),
            cache_slots: executable.cache_slots().to_vec(),
            cached_evaluation_nodes: executable.evaluation_nodes().to_vec(),
            cached_value_slots: executable.value_slots().to_vec(),
            scalar_kernel,
            scalar_executor,
            gradient_executor,
            f32_gradient_fallback_real,
            f32_gradient_fallback_imag,
            cache_materialization_nodes: executable.cache_materialization_nodes().to_vec(),
            solve_components: executable.solve_components().to_vec(),
            solve_rhs_elements: executable.solve_rhs_elements().to_vec(),
            solve_row_matrices: executable.solve_row_matrices().to_vec(),
            solve_row_keys: executable.solve_row_keys().to_vec(),
            factor_matrix_slots: executable.factor_matrix_slots().to_vec(),
            factor_matrices: executable.factor_matrices().to_vec(),
            constant_factor_slots: executable.constant_factor_slots().to_vec(),
            constant_factors,
        })
    }
}
