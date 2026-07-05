use std::sync::mpsc;

use laddu_compile::{CompiledModel, ExecutablePlan, ReductionPlan, ReductionTransform};
use laddu_data::data::{EventBatch, accurate::AccurateF64};
use laddu_expr::{BinaryOp, ExprNode, P4Component, UnaryOp, parameters::ParamValues};
use laddu_kernel::ir::{KernelInstruction, KernelValueId, KernelValueKind, ScalarKernelIr};
use wgpu::util::DeviceExt;

use crate::{WgpuContext, WgpuError, WgpuResult};

#[derive(Debug)]
pub struct WgpuScalarKernel {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    reduction_pipeline: wgpu::ComputePipeline,
    reduction_bind_group_layout: wgpu::BindGroupLayout,
    event_inputs: Vec<EventInput>,
}

#[derive(Clone, Debug)]
enum EventInput {
    Scalar(String),
    P4(String, P4Component),
}

impl WgpuScalarKernel {
    pub fn compile(context: &WgpuContext, model: &CompiledModel) -> WgpuResult<Self> {
        let executable = ExecutablePlan::from_model(model)
            .map_err(|error| WgpuError::UnsupportedInstruction(error.to_string()))?;
        let ir = executable
            .scalar_kernel()
            .ok_or(WgpuError::MissingScalarKernel)?;
        let event_inputs = executable
            .cache_plan()
            .entries()
            .iter()
            .map(|entry| match executable.graph().node(entry.node()) {
                Some(ExprNode::EventScalar(name)) => Ok(EventInput::Scalar(name.to_string())),
                Some(ExprNode::EventP4Component { name, component }) => {
                    Ok(EventInput::P4(name.to_string(), *component))
                }
                node => Err(WgpuError::UnsupportedInstruction(format!(
                    "computed cache entry {node:?}"
                ))),
            })
            .collect::<WgpuResult<Vec<_>>>()?;
        let source = Self::wgsl(ir, event_inputs.len())?;
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
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
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
        let reduction_bind_group_layout =
            context
                .device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("laddu reduction bindings"),
                    entries: &[
                        Self::storage_binding(0, true),
                        Self::storage_binding(1, true),
                        Self::storage_binding(3, true),
                        Self::storage_binding(4, true),
                        Self::storage_binding(5, false),
                        Self::storage_binding(6, false),
                    ],
                });
        let reduction_layout =
            context
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("laddu reduction pipeline layout"),
                    bind_group_layouts: &[Some(&reduction_bind_group_layout)],
                    immediate_size: 0,
                });
        let reduction_pipeline =
            context
                .device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("laddu reduction pipeline"),
                    layout: Some(&reduction_layout),
                    module: &module,
                    entry_point: Some("reduce"),
                    compilation_options: Default::default(),
                    cache: None,
                });
        Ok(Self {
            pipeline,
            bind_group_layout,
            reduction_pipeline,
            reduction_bind_group_layout,
            event_inputs,
        })
    }

    fn storage_binding(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    pub fn evaluate(&self, context: &WgpuContext, params: &ParamValues) -> WgpuResult<(f64, f64)> {
        self.evaluate_packed(context, params, &[0.0, 0.0], 1)
            .map(|mut values| values.remove(0))
    }

    pub fn evaluate_batch(
        &self,
        context: &WgpuContext,
        params: &ParamValues,
        batch: &EventBatch,
    ) -> WgpuResult<Vec<(f64, f64)>> {
        let cache = self.pack_batch(batch)?;
        self.evaluate_packed(context, params, &cache, batch.len())
    }

    pub fn reduce_batch(
        &self,
        context: &WgpuContext,
        params: &ParamValues,
        batch: &EventBatch,
        reduction: ReductionPlan,
    ) -> WgpuResult<f64> {
        if batch.is_empty() {
            return Ok(0.0);
        }
        let parameters = params
            .as_slice()
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();
        let parameters = if parameters.is_empty() {
            vec![0.0]
        } else {
            parameters
        };
        let cache = self.pack_batch(batch)?;
        let weights = (0..batch.len())
            .map(|row| batch.weights_at(row) as f32)
            .collect::<Vec<_>>();
        let mode = match reduction.transform() {
            ReductionTransform::Real => 0_u32,
            ReductionTransform::PositiveReal => 1,
            ReductionTransform::LogPositiveReal => 2,
        };
        let groups = batch.len().div_ceil(64);
        let params_buffer =
            context
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("laddu parameters"),
                    contents: bytemuck::cast_slice(&parameters),
                    usage: wgpu::BufferUsages::STORAGE,
                });
        let cache_buffer = context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("laddu event cache"),
                contents: bytemuck::cast_slice(&cache),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let weights_buffer =
            context
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("laddu event weights"),
                    contents: bytemuck::cast_slice(&weights),
                    usage: wgpu::BufferUsages::STORAGE,
                });
        let config_buffer =
            context
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("laddu reduction config"),
                    contents: bytemuck::bytes_of(&mode),
                    usage: wgpu::BufferUsages::STORAGE,
                });
        let partials = context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("laddu reduction partials"),
            size: (groups * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let error = context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("laddu reduction error"),
                contents: bytemuck::bytes_of(&u32::MAX),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let staging_size = (groups * 4 + 4) as u64;
        let staging = context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("laddu reduction readback"),
            size: staging_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bind_group = context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("laddu reduction bind group"),
                layout: &self.reduction_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: cache_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: weights_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: config_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: partials.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: error.as_entire_binding(),
                    },
                ],
            });
        let mut encoder = context.device().create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.reduction_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&partials, 0, &staging, 0, (groups * 4) as u64);
        encoder.copy_buffer_to_buffer(&error, 0, &staging, (groups * 4) as u64, 4);
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
        let words: &[u32] = bytemuck::cast_slice(&mapped);
        let invalid = words[groups];
        if invalid != u32::MAX {
            return Err(WgpuError::NonPositiveEvent(invalid as usize));
        }
        let mut total = AccurateF64::zero();
        for bits in &words[..groups] {
            total.push(f32::from_bits(*bits) as f64);
        }
        Ok(total.finish())
    }

    fn pack_batch(&self, batch: &EventBatch) -> WgpuResult<Vec<f32>> {
        let mut cache = Vec::with_capacity(batch.len() * self.event_inputs.len() * 2);
        for row in 0..batch.len() {
            for input in &self.event_inputs {
                let value = match input {
                    EventInput::Scalar(name) => {
                        let col = batch
                            .schema()
                            .scalar_index(name)
                            .ok_or_else(|| WgpuError::MissingEventColumn(name.clone()))?;
                        batch.scalar_at(col, row)
                    }
                    EventInput::P4(name, component) => {
                        let col = batch
                            .schema()
                            .p4_index(name)
                            .ok_or_else(|| WgpuError::MissingEventColumn(name.clone()))?;
                        let p4 = batch.p4_at(col, row);
                        match component {
                            P4Component::Px => p4.x,
                            P4Component::Py => p4.y,
                            P4Component::Pz => p4.z,
                            P4Component::E => p4.t,
                        }
                    }
                };
                cache.extend([value as f32, 0.0]);
            }
        }
        if cache.is_empty() {
            cache.extend([0.0, 0.0]);
        }
        Ok(cache)
    }

    fn evaluate_packed(
        &self,
        context: &WgpuContext,
        params: &ParamValues,
        cache: &[f32],
        events: usize,
    ) -> WgpuResult<Vec<(f64, f64)>> {
        if events == 0 {
            return Ok(Vec::new());
        }
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
        let cache_buffer = context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("laddu event cache"),
                contents: bytemuck::cast_slice(cache),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let output_size = (events * 8) as u64;
        let output = context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("laddu scalar output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("laddu scalar readback"),
            size: output_size,
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
                        resource: cache_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.as_entire_binding(),
                    },
                ],
            });
        let mut encoder = context.device().create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((events as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, output_size);
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
        let values = result
            .chunks_exact(2)
            .map(|value| (value[0] as f64, value[1] as f64))
            .collect();
        drop(mapped);
        staging.unmap();
        Ok(values)
    }

    fn wgsl(ir: &ScalarKernelIr, cache_slots: usize) -> WgpuResult<String> {
        let mut source = String::from(
            "@group(0) @binding(0) var<storage, read> p: array<f32>;\n@group(0) @binding(1) var<storage, read> cache: array<vec2<f32>>;\n@group(0) @binding(2) var<storage, read_write> out: array<vec2<f32>>;\n@group(0) @binding(3) var<storage, read> weights: array<f32>;\n@group(0) @binding(4) var<storage, read> config: array<u32>;\n@group(0) @binding(5) var<storage, read_write> partials: array<f32>;\n@group(0) @binding(6) var<storage, read_write> reduction_error: array<atomic<u32>>;\nvar<workgroup> sums: array<f32, 64>;\nfn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return vec2(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x); }\nfn cdiv(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { let d=b.x*b.x+b.y*b.y; return vec2((a.x*b.x+a.y*b.y)/d, (a.y*b.x-a.x*b.y)/d); }\nfn model(row: u32) -> vec2<f32> {\n",
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
                KernelInstruction::Cached(slot) => {
                    format!("cache[row * {cache_slots}u + {slot}u]")
                }
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
        source.push_str(&format!(
            "return {};\n}}\n@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\nif (gid.x >= arrayLength(&out)) {{ return; }}\nout[gid.x] = model(gid.x);\n}}\n@compute @workgroup_size(64) fn reduce(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{\nvar contribution = 0.0;\nif (gid.x < arrayLength(&weights)) {{\nlet value = model(gid.x).x;\nif (config[0] == 0u) {{ contribution = value; }} else if (value <= 0.0) {{ atomicMin(&reduction_error[0], gid.x); }} else if (config[0] == 1u) {{ contribution = value; }} else {{ contribution = log(value); }}\ncontribution *= weights[gid.x];\n}}\nsums[lid.x] = contribution;\nworkgroupBarrier();\nvar stride = 32u;\nloop {{\nif (lid.x < stride) {{ sums[lid.x] += sums[lid.x + stride]; }}\nworkgroupBarrier();\nif (stride == 1u) {{ break; }}\nstride /= 2u;\n}}\nif (lid.x == 0u) {{ partials[wid.x] = sums[0]; }}\n}}\n",
            v(ir.root())
        ));
        Ok(source)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use laddu_compile::{CompiledModel, ReductionPlan};
    use laddu_data::{
        data::{Dataset, EventBatch, OwnedEvent},
        schema::Schema,
    };
    use laddu_expr::{Expr, complex, event_scalar, parameter};
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

    #[test]
    #[ignore = "requires a WGPU-compatible hardware adapter"]
    fn gpu_event_batch_matches_f32_cpu_across_partial_workgroup() {
        let scale = Expr::from(parameter!("scale", initial: 1.25));
        let expression = event_scalar("x") * scale + event_scalar("y") - 0.5;
        let model = CompiledModel::from_expr(&expression).unwrap();
        let params = model.params().default_values();
        let schema = Arc::new(Schema::new(std::iter::empty::<&str>(), ["x", "y"], false).unwrap());
        let batch = EventBatch::from_events(
            schema,
            (0..70).map(|index| {
                OwnedEvent::weighted(
                    vec![],
                    vec![index as f64 * 0.125, 2.0 - index as f64 * 0.01],
                    1.0 + index as f64 * 0.01,
                )
            }),
        )
        .unwrap();
        let context = WgpuBackend::default()
            .open(&GpuOptions::default(), Precision::F32)
            .unwrap();
        let kernel = WgpuScalarKernel::compile(&context, &model).unwrap();
        let gpu = kernel.evaluate_batch(&context, &params, &batch).unwrap();
        let execution = Execution::local(ExecutionOptions {
            device: Device::Cpu(CpuOptions::default()),
            precision: Precision::F32,
            ..ExecutionOptions::default()
        })
        .unwrap();
        let plan = CpuBackend.prepare_for_execution(&model, &execution).unwrap();
        let cpu = plan.evaluate_batch(&params, &batch).unwrap();

        assert_eq!(gpu.len(), cpu.len());
        for (gpu, cpu) in gpu.iter().zip(cpu) {
            assert_eq!(*gpu, (cpu.re, cpu.im));
        }
        let dataset = Dataset::from_batch(batch.clone());
        let prepared = plan.prepare_dataset(&execution, &dataset).unwrap();
        for reduction in [
            ReductionPlan::weighted_real(),
            ReductionPlan::weighted_positive_real(),
            ReductionPlan::weighted_log_positive_real(),
        ] {
            let gpu = kernel
                .reduce_batch(&context, &params, &batch, reduction)
                .unwrap();
            let cpu = plan
                .reduce(&execution, &params, &prepared, reduction)
                .unwrap();
            assert!((gpu - cpu).abs() <= 1.0e-4 * cpu.abs().max(1.0));
        }

        let invalid = EventBatch::from_events(
            Arc::new(Schema::new(std::iter::empty::<&str>(), ["x", "y"], false).unwrap()),
            [OwnedEvent::new(vec![], vec![0.0, 0.0])],
        )
        .unwrap();
        assert!(matches!(
            kernel.reduce_batch(
                &context,
                &params,
                &invalid,
                ReductionPlan::weighted_positive_real()
            ),
            Err(crate::WgpuError::NonPositiveEvent(0))
        ));
    }
}
