use std::sync::mpsc;

use laddu_compile::{CompiledModel, ExecutablePlan};
use laddu_expr::{BinaryOp, UnaryOp, parameters::ParamValues};
use laddu_kernel::ir::{KernelInstruction, KernelValueId, KernelValueKind, ScalarKernelIr};
use wgpu::util::DeviceExt;

use crate::{WgpuContext, WgpuError, WgpuResult};

#[derive(Debug)]
pub struct WgpuScalarKernel {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl WgpuScalarKernel {
    pub fn compile(context: &WgpuContext, model: &CompiledModel) -> WgpuResult<Self> {
        let executable = ExecutablePlan::from_model(model)
            .map_err(|error| WgpuError::UnsupportedInstruction(error.to_string()))?;
        let ir = executable
            .scalar_kernel()
            .ok_or(WgpuError::MissingScalarKernel)?;
        let source = Self::wgsl(ir)?;
        let module = context
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("laddu scalar kernel"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let bind_group_layout =
            context
                .device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("laddu scalar bindings"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
        let layout = context
            .device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("laddu scalar pipeline layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });
        let pipeline = context
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("laddu scalar pipeline"),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        Ok(Self {
            pipeline,
            bind_group_layout,
        })
    }

    pub fn evaluate(&self, context: &WgpuContext, params: &ParamValues) -> WgpuResult<(f64, f64)> {
        let values = params
            .as_slice()
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();
        let values = if values.is_empty() { vec![0.0] } else { values };
        let params_buffer =
            context
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("laddu parameters"),
                    contents: bytemuck::cast_slice(&values),
                    usage: wgpu::BufferUsages::STORAGE,
                });
        let output = context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("laddu scalar output"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("laddu scalar readback"),
            size: 8,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bind_group = context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("laddu scalar bind group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output.as_entire_binding(),
                    },
                ],
            });
        let mut encoder = context.device().create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, 8);
        context.queue().submit([encoder.finish()]);
        let slice = staging.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        context
            .device()
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|error| WgpuError::DevicePoll(error.to_string()))?;
        receiver
            .recv()
            .map_err(|error| WgpuError::BufferMap(error.to_string()))?
            .map_err(|error| WgpuError::BufferMap(error.to_string()))?;
        let mapped = slice.get_mapped_range();
        let result: &[f32] = bytemuck::cast_slice(&mapped);
        let value = (result[0] as f64, result[1] as f64);
        drop(mapped);
        staging.unmap();
        Ok(value)
    }

    fn wgsl(ir: &ScalarKernelIr) -> WgpuResult<String> {
        let mut source = String::from(
            "@group(0) @binding(0) var<storage, read> p: array<f32>;\n@group(0) @binding(1) var<storage, read_write> out: array<vec2<f32>>;\nfn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return vec2(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x); }\nfn cdiv(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { let d=b.x*b.x+b.y*b.y; return vec2((a.x*b.x+a.y*b.y)/d, (a.y*b.x-a.x*b.y)/d); }\n@compute @workgroup_size(1) fn main() {\n",
        );
        let v = |id: KernelValueId| format!("v{}", id.index());
        for (index, value) in ir.values().iter().enumerate() {
            if !matches!(value.kind, KernelValueKind::Real | KernelValueKind::Complex) {
                return Err(WgpuError::UnsupportedInstruction(format!(
                    "{:?}",
                    value.instruction
                )));
            }
            let expr = match &value.instruction {
                KernelInstruction::RealConstant(x) => format!("vec2<f32>({:?}, 0.0)", *x as f32),
                KernelInstruction::ComplexConstant(x) => {
                    format!("vec2<f32>({:?}, {:?})", x.re as f32, x.im as f32)
                }
                KernelInstruction::Parameter(id) => format!("vec2<f32>(p[{}], 0.0)", id.index()),
                KernelInstruction::Unary {
                    op: UnaryOp::Neg,
                    input,
                } => format!("-{}", v(*input)),
                KernelInstruction::Unary {
                    op: UnaryOp::Real,
                    input,
                } => format!("vec2<f32>({}.x, 0.0)", v(*input)),
                KernelInstruction::Unary {
                    op: UnaryOp::Imag,
                    input,
                } => format!("vec2<f32>({}.y, 0.0)", v(*input)),
                KernelInstruction::Unary {
                    op: UnaryOp::Conj,
                    input,
                } => format!("vec2<f32>({}.x, -{}.y)", v(*input), v(*input)),
                KernelInstruction::Unary {
                    op: UnaryOp::NormSqr,
                    input,
                } => format!("vec2<f32>(dot({0}, {0}), 0.0)", v(*input)),
                KernelInstruction::Binary { op, lhs, rhs } => match op {
                    BinaryOp::Add => format!("{} + {}", v(*lhs), v(*rhs)),
                    BinaryOp::Sub => format!("{} - {}", v(*lhs), v(*rhs)),
                    BinaryOp::Mul => format!("cmul({}, {})", v(*lhs), v(*rhs)),
                    BinaryOp::Div => format!("cdiv({}, {})", v(*lhs), v(*rhs)),
                    BinaryOp::Atan2 => {
                        format!("vec2<f32>(atan2({}.x, {}.x), 0.0)", v(*lhs), v(*rhs))
                    }
                },
                KernelInstruction::Add(ids) => {
                    ids.iter().map(|id| v(*id)).collect::<Vec<_>>().join(" + ")
                }
                KernelInstruction::Mul(ids) => ids
                    .iter()
                    .map(|id| v(*id))
                    .reduce(|a, b| format!("cmul({a}, {b})"))
                    .unwrap(),
                KernelInstruction::Complex { re, im } => {
                    format!("vec2<f32>({}.x, {}.x)", v(*re), v(*im))
                }
                instruction => {
                    return Err(WgpuError::UnsupportedInstruction(format!(
                        "{instruction:?}"
                    )));
                }
            };
            source.push_str(&format!("let v{index} = {expr};\n"));
        }
        source.push_str(&format!("out[0] = {};\n}}\n", v(ir.root())));
        Ok(source)
    }
}

#[cfg(test)]
mod tests {
    use laddu_compile::CompiledModel;
    use laddu_expr::{Expr, complex, parameter};
    use laddu_runtime::{
        CpuBackend, CpuOptions, Device, Execution, ExecutionOptions, GpuOptions, Precision,
    };

    use crate::{WgpuBackend, WgpuScalarKernel};

    #[test]
    #[ignore = "requires a WGPU-compatible hardware adapter"]
    fn gpu_scalar_kernel_matches_f32_cpu() {
        let x = Expr::from(parameter!("x", initial: 1.25));
        let y = Expr::from(parameter!("y", initial: -0.4));
        let expression = (complex(x.clone() * y.clone() + 2.0, y) * complex(x, -1.0)).norm_sqr();
        let model = CompiledModel::from_expr(&expression).unwrap();
        let params = model.params().default_values();
        let context = WgpuBackend::default()
            .open(&GpuOptions::default(), Precision::F32)
            .unwrap();
        let kernel = WgpuScalarKernel::compile(&context, &model).unwrap();
        let gpu = kernel.evaluate(&context, &params).unwrap();
        let execution = Execution::local(ExecutionOptions {
            device: Device::Cpu(CpuOptions::default()),
            precision: Precision::F32,
            ..ExecutionOptions::default()
        })
        .unwrap();
        let cpu = CpuBackend
            .prepare_for_execution(&model, &execution)
            .unwrap()
            .evaluate(&params)
            .unwrap();

        assert_eq!(gpu, (cpu.re, cpu.im));
    }
}
