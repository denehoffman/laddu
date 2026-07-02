use std::{
    fmt,
    mem::{self, size_of},
    sync::{Arc, Mutex},
};

use cranelift::{
    codegen::ir::{
        BlockArg, FuncRef, MemFlagsData, StackSlotData, StackSlotKind, UserFuncName,
        condcodes::IntCC,
    },
    jit::{JITBuilder, JITModule},
    module::{FuncId, Linkage, Module, default_libcall_names},
    prelude::*,
};
use laddu_expr::{BinaryOp, UnaryOp, parameters::ParamValues};
use laddu_kernel::ir::{
    KernelInstruction, KernelValue, KernelValueClass, KernelValueId, KernelValueKind,
    ScalarKernelIr,
};
use nalgebra::{DMatrix, DVector};
use num::complex::Complex64;

use crate::{CpuBatchCache, RuntimeError, RuntimeResult};

#[repr(C)]
#[derive(Copy, Clone)]
struct CacheDescriptor {
    values: *const Complex64,
    width: usize,
}

pub(crate) struct JitCacheView {
    values: Vec<CacheDescriptor>,
    solve_rows: Vec<CacheDescriptor>,
}

// The descriptors borrow immutable cache allocations for the duration of an evaluation.
// Generated kernels only read through these pointers, so sharing a view across workers is safe.
unsafe impl Send for JitCacheView {}
unsafe impl Sync for JitCacheView {}

type BlockJitFn = unsafe extern "C" fn(
    *const f64,
    *const CacheDescriptor,
    *const CacheDescriptor,
    usize,
    usize,
    *mut Complex64,
) -> i32;

#[derive(Clone)]
pub(crate) struct JitScalarKernel {
    code: Arc<JitScalarCode>,
}

struct JitScalarCode {
    _module: Mutex<JITModule>,
    function: BlockJitFn,
}

impl fmt::Debug for JitScalarKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("JitScalarKernel")
            .finish_non_exhaustive()
    }
}

#[derive(Copy, Clone)]
struct ComplexValue {
    re: cranelift::prelude::Value,
    im: cranelift::prelude::Value,
}

#[derive(Clone)]
struct LoweredValue {
    kind: KernelValueKind,
    elements: Vec<ComplexValue>,
}

struct HelperFunctions {
    unary: FuncId,
    binary: FuncId,
    solve: FuncId,
}

struct FunctionHelpers {
    unary: FuncRef,
    binary: FuncRef,
    solve: FuncRef,
}

impl JitScalarKernel {
    pub(crate) fn compile(ir: &ScalarKernelIr) -> Result<Option<Self>, String> {
        let mut jit_builder =
            JITBuilder::new(default_libcall_names()).map_err(|error| error.to_string())?;
        jit_builder.symbol("laddu_jit_unary", unary_helper as *const u8);
        jit_builder.symbol("laddu_jit_binary", binary_helper as *const u8);
        jit_builder.symbol("laddu_jit_solve", solve_helper as *const u8);
        let mut module = JITModule::new(jit_builder);
        let pointer_type = module.target_config().pointer_type();
        if pointer_type != types::I64 {
            return Ok(None);
        }

        let helpers = declare_helpers(&mut module, pointer_type)?;
        let mut signature = module.make_signature();
        for _ in 0..3 {
            signature.params.push(AbiParam::new(pointer_type));
        }
        signature.params.push(AbiParam::new(pointer_type));
        signature.params.push(AbiParam::new(pointer_type));
        signature.params.push(AbiParam::new(pointer_type));
        signature.returns.push(AbiParam::new(types::I32));
        let function_id = module
            .declare_function("laddu_block_kernel", Linkage::Local, &signature)
            .map_err(|error| error.to_string())?;
        let mut context = module.make_context();
        context.func.signature = signature;
        context.func.name = UserFuncName::user(0, function_id.as_u32());
        let mut function_context = FunctionBuilderContext::new();

        {
            let function_helpers = FunctionHelpers {
                unary: module.declare_func_in_func(helpers.unary, &mut context.func),
                binary: module.declare_func_in_func(helpers.binary, &mut context.func),
                solve: module.declare_func_in_func(helpers.solve, &mut context.func),
            };
            let mut builder = FunctionBuilder::new(&mut context.func, &mut function_context);
            emit_kernel(&mut builder, ir, pointer_type, &function_helpers)?;
            builder.finalize();
        }

        module
            .define_function(function_id, &mut context)
            .map_err(|error| error.to_string())?;
        module
            .finalize_definitions()
            .map_err(|error| error.to_string())?;
        let code = module.get_finalized_function(function_id);
        let function = unsafe { mem::transmute::<*const u8, BlockJitFn>(code) };
        Ok(Some(Self {
            code: Arc::new(JitScalarCode {
                _module: Mutex::new(module),
                function,
            }),
        }))
    }

    pub(crate) fn evaluate(
        &self,
        parameters: &ParamValues,
        cache: &CpuBatchCache,
        start: usize,
        end: usize,
        output: &mut Vec<Complex64>,
    ) -> RuntimeResult<()> {
        let view = JitCacheView::new(cache);
        self.evaluate_prepared(parameters, &view, start, end, output)
    }

    pub(crate) fn prepare_cache(cache: &CpuBatchCache) -> JitCacheView {
        JitCacheView::new(cache)
    }

    pub(crate) fn evaluate_prepared(
        &self,
        parameters: &ParamValues,
        view: &JitCacheView,
        start: usize,
        end: usize,
        output: &mut Vec<Complex64>,
    ) -> RuntimeResult<()> {
        output.clear();
        output.resize(end - start, Complex64::ZERO);
        let status = unsafe {
            (self.code.function)(
                parameters.as_slice().as_ptr(),
                view.values.as_ptr(),
                view.solve_rows.as_ptr(),
                start,
                end - start,
                output.as_mut_ptr(),
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(RuntimeError::JitExecution(status))
        }
    }

    pub(crate) fn evaluate_invariant(&self, parameters: &ParamValues) -> RuntimeResult<Complex64> {
        let mut output = Complex64::ZERO;
        let status = unsafe {
            (self.code.function)(
                parameters.as_slice().as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                1,
                &mut output,
            )
        };
        if status == 0 {
            Ok(output)
        } else {
            Err(RuntimeError::JitExecution(status))
        }
    }
}

impl JitCacheView {
    fn new(cache: &CpuBatchCache) -> Self {
        let values = cache
            .slots
            .iter()
            .map(|slot| CacheDescriptor {
                values: slot.values().as_ptr(),
                width: slot.width(),
            })
            .collect();
        let solve_rows = cache
            .solve_row_slots
            .iter()
            .map(|slot| CacheDescriptor {
                values: slot.values.as_ptr(),
                width: slot.dimension,
            })
            .collect();
        Self { values, solve_rows }
    }
}

fn declare_helpers(module: &mut JITModule, pointer_type: Type) -> Result<HelperFunctions, String> {
    let mut unary = module.make_signature();
    unary.params.extend([
        AbiParam::new(types::I32),
        AbiParam::new(types::I32),
        AbiParam::new(types::F64),
        AbiParam::new(types::F64),
        AbiParam::new(pointer_type),
    ]);
    let unary = module
        .declare_function("laddu_jit_unary", Linkage::Import, &unary)
        .map_err(|e| e.to_string())?;
    let mut binary = module.make_signature();
    binary.params.extend([
        AbiParam::new(types::I32),
        AbiParam::new(types::F64),
        AbiParam::new(types::F64),
        AbiParam::new(types::F64),
        AbiParam::new(types::F64),
        AbiParam::new(pointer_type),
    ]);
    let binary = module
        .declare_function("laddu_jit_binary", Linkage::Import, &binary)
        .map_err(|e| e.to_string())?;
    let mut solve = module.make_signature();
    solve.params.extend([
        AbiParam::new(pointer_type),
        AbiParam::new(pointer_type),
        AbiParam::new(pointer_type),
        AbiParam::new(pointer_type),
    ]);
    solve.returns.push(AbiParam::new(types::I32));
    let solve = module
        .declare_function("laddu_jit_solve", Linkage::Import, &solve)
        .map_err(|e| e.to_string())?;
    Ok(HelperFunctions {
        unary,
        binary,
        solve,
    })
}

fn emit_kernel(
    builder: &mut FunctionBuilder<'_>,
    ir: &ScalarKernelIr,
    pointer_type: Type,
    helpers: &FunctionHelpers,
) -> Result<(), String> {
    let entry = builder.create_block();
    let loop_header = builder.create_block();
    let body = builder.create_block();
    let done = builder.create_block();
    let failed = builder.create_block();
    builder.append_block_params_for_function_params(entry);
    builder.append_block_param(loop_header, pointer_type);
    builder.switch_to_block(entry);
    let args = builder.block_params(entry).to_vec();
    let parameters = args[0];
    let cache = args[1];
    let solve_rows = args[2];
    let start = args[3];
    let len = args[4];
    let output = args[5];
    let end = builder.ins().iadd(start, len);

    let mut invariant = vec![None; ir.values().len()];
    let invariant_row = builder.ins().iconst(pointer_type, 0);
    for (index, value) in ir.values().iter().enumerate() {
        if value.class == KernelValueClass::Invariant {
            invariant[index] = Some(emit_instruction(
                builder,
                value,
                &invariant,
                parameters,
                cache,
                solve_rows,
                invariant_row,
                pointer_type,
                helpers,
                failed,
            )?);
        }
    }
    let start_arg = BlockArg::from(start);
    builder.ins().jump(loop_header, &[start_arg]);

    builder.switch_to_block(loop_header);
    let row = builder.block_params(loop_header)[0];
    let finished = builder
        .ins()
        .icmp(IntCC::UnsignedGreaterThanOrEqual, row, end);
    builder.ins().brif(finished, done, &[], body, &[]);

    builder.switch_to_block(body);
    let mut values = invariant.clone();
    for (index, value) in ir.values().iter().enumerate() {
        if value.class == KernelValueClass::Event {
            values[index] = Some(emit_instruction(
                builder,
                value,
                &values,
                parameters,
                cache,
                solve_rows,
                row,
                pointer_type,
                helpers,
                failed,
            )?);
        }
    }
    let root = values[ir.root().index()]
        .as_ref()
        .ok_or("kernel root was not lowered")?;
    let result = root.elements.first().ok_or("kernel root is empty")?;
    let output_row = builder.ins().isub(row, start);
    let byte_offset = builder
        .ins()
        .imul_imm(output_row, size_of::<Complex64>() as i64);
    let output_ptr = builder.ins().iadd(output, byte_offset);
    builder
        .ins()
        .store(MemFlagsData::trusted(), result.re, output_ptr, 0);
    builder
        .ins()
        .store(MemFlagsData::trusted(), result.im, output_ptr, 8);
    let next = builder.ins().iadd_imm(row, 1);
    let next_arg = BlockArg::from(next);
    builder.ins().jump(loop_header, &[next_arg]);

    builder.switch_to_block(done);
    let success_status = builder.ins().iconst(types::I32, 0);
    builder.ins().return_(&[success_status]);
    builder.switch_to_block(failed);
    let failure_status = builder.ins().iconst(types::I32, 1);
    builder.ins().return_(&[failure_status]);
    builder.seal_all_blocks();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn emit_instruction(
    builder: &mut FunctionBuilder<'_>,
    value: &KernelValue,
    values: &[Option<LoweredValue>],
    parameters: cranelift::prelude::Value,
    cache: cranelift::prelude::Value,
    solve_rows: cranelift::prelude::Value,
    row: cranelift::prelude::Value,
    pointer_type: Type,
    helpers: &FunctionHelpers,
    failed: Block,
) -> Result<LoweredValue, String> {
    let get = |id: KernelValueId| {
        values[id.index()]
            .as_ref()
            .ok_or_else(|| format!("kernel operand {} was not lowered", id.index()))
    };
    let scalar = |id: KernelValueId| -> Result<ComplexValue, String> {
        get(id)?
            .elements
            .first()
            .copied()
            .ok_or_else(|| "scalar operand is empty".into())
    };
    let zero = |builder: &mut FunctionBuilder<'_>| ComplexValue {
        re: builder.ins().f64const(0.0),
        im: builder.ins().f64const(0.0),
    };
    let elements = match &value.instruction {
        KernelInstruction::Cached(slot) => {
            load_descriptor(builder, cache, *slot, value.kind.width(), row, pointer_type)?
        }
        KernelInstruction::RealConstant(number) => vec![ComplexValue {
            re: builder.ins().f64const(*number),
            im: builder.ins().f64const(0.0),
        }],
        KernelInstruction::ComplexConstant(number) => vec![ComplexValue {
            re: builder.ins().f64const(number.re),
            im: builder.ins().f64const(number.im),
        }],
        KernelInstruction::Parameter(id) => {
            let offset = i32::try_from(id.index() * size_of::<f64>())
                .map_err(|_| "parameter offset exceeds JIT address range")?;
            vec![ComplexValue {
                re: builder
                    .ins()
                    .load(types::F64, MemFlagsData::trusted(), parameters, offset),
                im: builder.ins().f64const(0.0),
            }]
        }
        KernelInstruction::Unary { op, input } => vec![emit_unary(
            builder,
            *op,
            scalar(*input)?,
            pointer_type,
            helpers.unary,
        )],
        KernelInstruction::Binary { op, lhs, rhs } => vec![emit_binary(
            builder,
            *op,
            scalar(*lhs)?,
            scalar(*rhs)?,
            pointer_type,
            helpers.binary,
        )],
        KernelInstruction::Add(terms) => {
            let mut out = zero(builder);
            for term in terms {
                out = add(builder, out, scalar(*term)?);
            }
            vec![out]
        }
        KernelInstruction::Mul(factors) => {
            let mut out = ComplexValue {
                re: builder.ins().f64const(1.0),
                im: builder.ins().f64const(0.0),
            };
            for factor in factors {
                out = mul(builder, out, scalar(*factor)?);
            }
            vec![out]
        }
        KernelInstruction::Complex { re, im } => vec![ComplexValue {
            re: scalar(*re)?.re,
            im: scalar(*im)?.re,
        }],
        KernelInstruction::Vector(entries) => entries
            .iter()
            .map(|entry| scalar(*entry))
            .collect::<Result<_, _>>()?,
        KernelInstruction::Matrix { elements, .. } => elements
            .iter()
            .map(|entry| scalar(*entry))
            .collect::<Result<_, _>>()?,
        KernelInstruction::Component { input, index } => vec![get(*input)?.elements[*index]],
        KernelInstruction::MatrixElement { input, row, col } => {
            let KernelValueKind::Matrix { cols, .. } = get(*input)?.kind else {
                unreachable!()
            };
            vec![get(*input)?.elements[row * cols + col]]
        }
        KernelInstruction::MatMul { lhs, rhs } => {
            let lhs = get(*lhs)?;
            let rhs = get(*rhs)?;
            let (
                KernelValueKind::Matrix { rows, cols: inner },
                KernelValueKind::Matrix { cols, .. },
            ) = (lhs.kind, rhs.kind)
            else {
                unreachable!()
            };
            let mut out = Vec::with_capacity(rows * cols);
            for r in 0..rows {
                for c in 0..cols {
                    let mut sum = zero(builder);
                    for k in 0..inner {
                        let product = mul(
                            builder,
                            lhs.elements[r * inner + k],
                            rhs.elements[k * cols + c],
                        );
                        sum = add(builder, sum, product);
                    }
                    out.push(sum);
                }
            }
            out
        }
        KernelInstruction::MatVec { matrix, vector } => {
            let matrix = get(*matrix)?;
            let vector = get(*vector)?;
            let KernelValueKind::Matrix { rows, cols } = matrix.kind else {
                unreachable!()
            };
            let mut out = Vec::with_capacity(rows);
            for r in 0..rows {
                let mut sum = zero(builder);
                for c in 0..cols {
                    let product = mul(builder, matrix.elements[r * cols + c], vector.elements[c]);
                    sum = add(builder, sum, product);
                }
                out.push(sum);
            }
            out
        }
        KernelInstruction::Dot { lhs, rhs } => {
            let lhs = get(*lhs)?;
            let rhs = get(*rhs)?;
            let mut sum = zero(builder);
            for (lhs, rhs) in lhs.elements.iter().zip(&rhs.elements) {
                let product = mul(builder, *lhs, *rhs);
                sum = add(builder, sum, product);
            }
            vec![sum]
        }
        KernelInstruction::Solve { matrix, rhs } => {
            let matrix = get(*matrix)?;
            let rhs = get(*rhs)?;
            emit_solve(
                builder,
                &matrix.elements,
                &rhs.elements,
                pointer_type,
                helpers.solve,
                failed,
            )?
        }
        KernelInstruction::SolveRow { row_slot, rhs } => {
            let inverse =
                load_descriptor(builder, solve_rows, *row_slot, rhs.len(), row, pointer_type)?;
            let mut sum = zero(builder);
            for (coefficient, rhs) in inverse.iter().zip(rhs) {
                let product = mul(builder, *coefficient, scalar(*rhs)?);
                sum = add(builder, sum, product);
            }
            vec![sum]
        }
    };
    Ok(LoweredValue {
        kind: value.kind,
        elements,
    })
}

fn load_descriptor(
    builder: &mut FunctionBuilder<'_>,
    descriptors: cranelift::prelude::Value,
    slot: usize,
    width: usize,
    row: cranelift::prelude::Value,
    pointer_type: Type,
) -> Result<Vec<ComplexValue>, String> {
    let descriptor_size = size_of::<CacheDescriptor>();
    let descriptor_offset = i32::try_from(slot * descriptor_size)
        .map_err(|_| "cache descriptor offset exceeds JIT address range")?;
    let base = builder.ins().load(
        pointer_type,
        MemFlagsData::trusted(),
        descriptors,
        descriptor_offset,
    );
    let row_width = builder.ins().imul_imm(row, width as i64);
    let row_bytes = builder
        .ins()
        .imul_imm(row_width, size_of::<Complex64>() as i64);
    let row_ptr = builder.ins().iadd(base, row_bytes);
    let mut out = Vec::with_capacity(width);
    for index in 0..width {
        let offset = i32::try_from(index * size_of::<Complex64>())
            .map_err(|_| "cached value offset exceeds JIT address range")?;
        out.push(ComplexValue {
            re: builder
                .ins()
                .load(types::F64, MemFlagsData::trusted(), row_ptr, offset),
            im: builder
                .ins()
                .load(types::F64, MemFlagsData::trusted(), row_ptr, offset + 8),
        });
    }
    Ok(out)
}

fn add(builder: &mut FunctionBuilder<'_>, lhs: ComplexValue, rhs: ComplexValue) -> ComplexValue {
    ComplexValue {
        re: builder.ins().fadd(lhs.re, rhs.re),
        im: builder.ins().fadd(lhs.im, rhs.im),
    }
}

fn mul(builder: &mut FunctionBuilder<'_>, lhs: ComplexValue, rhs: ComplexValue) -> ComplexValue {
    let ac = builder.ins().fmul(lhs.re, rhs.re);
    let bd = builder.ins().fmul(lhs.im, rhs.im);
    let ad = builder.ins().fmul(lhs.re, rhs.im);
    let bc = builder.ins().fmul(lhs.im, rhs.re);
    ComplexValue {
        re: builder.ins().fsub(ac, bd),
        im: builder.ins().fadd(ad, bc),
    }
}

fn emit_unary(
    builder: &mut FunctionBuilder<'_>,
    op: UnaryOp,
    input: ComplexValue,
    pointer_type: Type,
    helper: FuncRef,
) -> ComplexValue {
    match op {
        UnaryOp::Neg => ComplexValue {
            re: builder.ins().fneg(input.re),
            im: builder.ins().fneg(input.im),
        },
        UnaryOp::Real => ComplexValue {
            re: input.re,
            im: builder.ins().f64const(0.0),
        },
        UnaryOp::Imag => ComplexValue {
            re: input.im,
            im: builder.ins().f64const(0.0),
        },
        UnaryOp::Conj => ComplexValue {
            re: input.re,
            im: builder.ins().fneg(input.im),
        },
        UnaryOp::NormSqr => {
            let re2 = builder.ins().fmul(input.re, input.re);
            let im2 = builder.ins().fmul(input.im, input.im);
            ComplexValue {
                re: builder.ins().fadd(re2, im2),
                im: builder.ins().f64const(0.0),
            }
        }
        _ => {
            let (code, power) = unary_code(op);
            let slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                16,
                3,
            ));
            let out = builder.ins().stack_addr(pointer_type, slot, 0);
            let code = builder.ins().iconst(types::I32, i64::from(code));
            let power = builder.ins().iconst(types::I32, i64::from(power));
            builder
                .ins()
                .call(helper, &[code, power, input.re, input.im, out]);
            ComplexValue {
                re: builder.ins().stack_load(types::F64, slot, 0),
                im: builder.ins().stack_load(types::F64, slot, 8),
            }
        }
    }
}

fn emit_binary(
    builder: &mut FunctionBuilder<'_>,
    op: BinaryOp,
    lhs: ComplexValue,
    rhs: ComplexValue,
    pointer_type: Type,
    helper: FuncRef,
) -> ComplexValue {
    match op {
        BinaryOp::Add => add(builder, lhs, rhs),
        BinaryOp::Sub => ComplexValue {
            re: builder.ins().fsub(lhs.re, rhs.re),
            im: builder.ins().fsub(lhs.im, rhs.im),
        },
        BinaryOp::Mul => mul(builder, lhs, rhs),
        BinaryOp::Div => {
            let c2 = builder.ins().fmul(rhs.re, rhs.re);
            let d2 = builder.ins().fmul(rhs.im, rhs.im);
            let denominator = builder.ins().fadd(c2, d2);
            let ac = builder.ins().fmul(lhs.re, rhs.re);
            let bd = builder.ins().fmul(lhs.im, rhs.im);
            let bc = builder.ins().fmul(lhs.im, rhs.re);
            let ad = builder.ins().fmul(lhs.re, rhs.im);
            let numerator_re = builder.ins().fadd(ac, bd);
            let numerator_im = builder.ins().fsub(bc, ad);
            ComplexValue {
                re: builder.ins().fdiv(numerator_re, denominator),
                im: builder.ins().fdiv(numerator_im, denominator),
            }
        }
        BinaryOp::Atan2 => {
            let slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                16,
                3,
            ));
            let out = builder.ins().stack_addr(pointer_type, slot, 0);
            let code = builder.ins().iconst(types::I32, 0);
            builder
                .ins()
                .call(helper, &[code, lhs.re, lhs.im, rhs.re, rhs.im, out]);
            ComplexValue {
                re: builder.ins().stack_load(types::F64, slot, 0),
                im: builder.ins().stack_load(types::F64, slot, 8),
            }
        }
    }
}

fn emit_solve(
    builder: &mut FunctionBuilder<'_>,
    matrix: &[ComplexValue],
    rhs: &[ComplexValue],
    pointer_type: Type,
    helper: FuncRef,
    failed: Block,
) -> Result<Vec<ComplexValue>, String> {
    let matrix_size = u32::try_from(matrix.len() * size_of::<Complex64>())
        .map_err(|_| "matrix is too large for JIT stack storage")?;
    let vector_size = u32::try_from(rhs.len() * size_of::<Complex64>())
        .map_err(|_| "vector is too large for JIT stack storage")?;
    let matrix_slot = builder.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        matrix_size,
        3,
    ));
    let rhs_slot = builder.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        vector_size,
        3,
    ));
    let out_slot = builder.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        vector_size,
        3,
    ));
    for (index, value) in matrix.iter().enumerate() {
        builder
            .ins()
            .stack_store(value.re, matrix_slot, (index * 16) as i32);
        builder
            .ins()
            .stack_store(value.im, matrix_slot, (index * 16 + 8) as i32);
    }
    for (index, value) in rhs.iter().enumerate() {
        builder
            .ins()
            .stack_store(value.re, rhs_slot, (index * 16) as i32);
        builder
            .ins()
            .stack_store(value.im, rhs_slot, (index * 16 + 8) as i32);
    }
    let dimension = builder.ins().iconst(pointer_type, rhs.len() as i64);
    let matrix_ptr = builder.ins().stack_addr(pointer_type, matrix_slot, 0);
    let rhs_ptr = builder.ins().stack_addr(pointer_type, rhs_slot, 0);
    let out_ptr = builder.ins().stack_addr(pointer_type, out_slot, 0);
    let call = builder
        .ins()
        .call(helper, &[dimension, matrix_ptr, rhs_ptr, out_ptr]);
    let status = builder.inst_results(call)[0];
    let failed_status = builder.ins().icmp_imm(IntCC::NotEqual, status, 0);
    let success = builder.create_block();
    builder.ins().brif(failed_status, failed, &[], success, &[]);
    builder.switch_to_block(success);
    let mut out = Vec::with_capacity(rhs.len());
    for index in 0..rhs.len() {
        out.push(ComplexValue {
            re: builder
                .ins()
                .stack_load(types::F64, out_slot, (index * 16) as i32),
            im: builder
                .ins()
                .stack_load(types::F64, out_slot, (index * 16 + 8) as i32),
        });
    }
    Ok(out)
}

fn unary_code(op: UnaryOp) -> (i32, i32) {
    match op {
        UnaryOp::Sqrt => (0, 0),
        UnaryOp::Exp => (1, 0),
        UnaryOp::Sin => (2, 0),
        UnaryOp::Cos => (3, 0),
        UnaryOp::Log => (4, 0),
        UnaryOp::PowI(power) => (5, power),
        _ => unreachable!(),
    }
}

unsafe extern "C" fn unary_helper(code: i32, power: i32, re: f64, im: f64, out: *mut Complex64) {
    let value = Complex64::new(re, im);
    let result = match code {
        0 => value.sqrt(),
        1 => value.exp(),
        2 => value.sin(),
        3 => value.cos(),
        4 => value.ln(),
        5 => value.powi(power),
        _ => Complex64::new(f64::NAN, f64::NAN),
    };
    unsafe { out.write(result) };
}

unsafe extern "C" fn binary_helper(
    code: i32,
    lhs_re: f64,
    _lhs_im: f64,
    rhs_re: f64,
    _rhs_im: f64,
    out: *mut Complex64,
) {
    let result = match code {
        0 => Complex64::from(lhs_re.atan2(rhs_re)),
        _ => Complex64::new(f64::NAN, f64::NAN),
    };
    unsafe { out.write(result) };
}

unsafe extern "C" fn solve_helper(
    dimension: usize,
    matrix: *const Complex64,
    rhs: *const Complex64,
    out: *mut Complex64,
) -> i32 {
    let matrix = unsafe { std::slice::from_raw_parts(matrix, dimension * dimension) };
    let rhs = unsafe { std::slice::from_raw_parts(rhs, dimension) };
    let Some(solution) = DMatrix::from_row_slice(dimension, dimension, matrix)
        .lu()
        .solve(&DVector::from_row_slice(rhs))
    else {
        return 1;
    };
    unsafe { std::ptr::copy_nonoverlapping(solution.as_ptr(), out, dimension) };
    0
}
