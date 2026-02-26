#[cfg(not(feature = "mpi"))]
fn main() {}

#[cfg(feature = "mpi")]
mod mpi_benches {
    use std::sync::Arc;

    use criterion::{black_box, BatchSize, BenchmarkId, Criterion, Throughput};
    use laddu::mpi::{finalize_mpi, get_world, use_mpi};
    use laddu::{
        amplitudes::{
            kmatrix::{KopfKMatrixA0, KopfKMatrixA2, KopfKMatrixF0, KopfKMatrixF2},
            parameter,
            zlm::Zlm,
        },
        data::{Dataset, DatasetReadOptions},
        extensions::NLL,
        io,
        traits::LikelihoodTerm,
        utils::{
            enums::{Frame, Sign},
            variables::{Angles, Mass, Polarization, Topology},
        },
        RngSubsetExtension,
    };
    use mpi::collective::SystemOperation;
    use mpi::traits::{Communicator, CommunicatorCollectives};

    const BENCH_DATASET_RELATIVE_PATH: &str = "benches/bench.parquet";
    const P4_NAMES: [&str; 4] = ["beam", "proton", "kshort1", "kshort2"];
    const AUX_NAMES: [&str; 2] = ["pol_magnitude", "pol_angle"];
    const KMATRIX_DATASET_SEED: u64 = 71;

    fn read_benchmark_dataset() -> Arc<Dataset> {
        let options = DatasetReadOptions::default()
            .p4_names(P4_NAMES)
            .aux_names(AUX_NAMES);
        let dataset_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(BENCH_DATASET_RELATIVE_PATH)
            .to_string_lossy()
            .into_owned();
        io::read_parquet(&dataset_path, &options).expect("benchmark dataset should open")
    }

    fn deterministic_parameter_vector(n_free: usize, offset: f64) -> Vec<f64> {
        (0..n_free)
            .map(|index| offset + (index as f64 + 1.0) * 0.25)
            .collect()
    }

    fn sample_dataset(dataset: &Arc<Dataset>, seed: u64, max_events: usize) -> Arc<Dataset> {
        let mut rng = fastrand::Rng::with_seed(seed);
        let n_total = dataset.n_events();
        let n_take = max_events.min(n_total);
        let indices = rng.subset(n_total, n_take);
        let events = indices
            .into_iter()
            .map(|index| {
                dataset
                    .event(index)
                    .expect("subset index should be valid")
                    .data_arc()
            })
            .collect();
        Arc::new(Dataset::new_with_metadata(events, dataset.metadata_arc()))
    }

    fn kmatrix_max_events_from_env() -> Option<usize> {
        std::env::var("LADDU_BENCH_MAX_EVENTS")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .filter(|value| *value > 0)
    }

    fn build_kmatrix_nll() -> Box<NLL> {
        let dataset = read_benchmark_dataset();
        let (ds_data, ds_mc) = if let Some(max_events) = kmatrix_max_events_from_env() {
            let sampled = sample_dataset(&dataset, KMATRIX_DATASET_SEED, max_events);
            (sampled.clone(), sampled)
        } else {
            (dataset.clone(), dataset)
        };
        let topology = Topology::missing_k2("beam", ["kshort1", "kshort2"], "proton");
        let angles = Angles::new(topology.clone(), "kshort1", Frame::Helicity);
        let polarization = Polarization::new(topology.clone(), "pol_magnitude", "pol_angle");
        let resonance_mass = Mass::new(["kshort1", "kshort2"]);
        let z00p = Zlm::new("Z00+", 0, 0, Sign::Positive, &angles, &polarization)
            .expect("z00+ should construct");
        let z00n = Zlm::new("Z00-", 0, 0, Sign::Negative, &angles, &polarization)
            .expect("z00- should construct");
        let z22p = Zlm::new("Z22+", 2, 2, Sign::Positive, &angles, &polarization)
            .expect("z22+ should construct");

        let f0p = KopfKMatrixF0::new(
            "f0+",
            [
                [
                    laddu::constant("f0+ c00 re", 0.0),
                    laddu::constant("f0+ c00 im", 0.0),
                ],
                [
                    parameter("f0(980)+ re"),
                    laddu::constant("f0(980)+ im_fix", 0.0),
                ],
                [parameter("f0(1370)+ re"), parameter("f0(1370)+ im")],
                [parameter("f0(1500)+ re"), parameter("f0(1500)+ im")],
                [parameter("f0(1710)+ re"), parameter("f0(1710)+ im")],
            ],
            0,
            &resonance_mass,
            None,
        )
        .expect("f0+ should construct");
        let a0p = KopfKMatrixA0::new(
            "a0+",
            [
                [parameter("a0(980)+ re"), parameter("a0(980)+ im")],
                [parameter("a0(1450)+ re"), parameter("a0(1450)+ im")],
            ],
            0,
            &resonance_mass,
            None,
        )
        .expect("a0+ should construct");
        let f0n = KopfKMatrixF0::new(
            "f0-",
            [
                [
                    laddu::constant("f0- c00 re", 0.0),
                    laddu::constant("f0- c00 im", 0.0),
                ],
                [
                    parameter("f0(980)- re"),
                    laddu::constant("f0(980)- im_fix", 0.0),
                ],
                [parameter("f0(1370)- re"), parameter("f0(1370)- im")],
                [parameter("f0(1500)- re"), parameter("f0(1500)- im")],
                [parameter("f0(1710)- re"), parameter("f0(1710)- im")],
            ],
            0,
            &resonance_mass,
            None,
        )
        .expect("f0- should construct");
        let a0n = KopfKMatrixA0::new(
            "a0-",
            [
                [parameter("a0(980)- re"), parameter("a0(980)- im")],
                [parameter("a0(1450)- re"), parameter("a0(1450)- im")],
            ],
            0,
            &resonance_mass,
            None,
        )
        .expect("a0- should construct");
        let f2 = KopfKMatrixF2::new(
            "f2",
            [
                [parameter("f2(1270) re"), parameter("f2(1270) im")],
                [parameter("f2(1525) re"), parameter("f2(1525) im")],
                [parameter("f2(1850) re"), parameter("f2(1850) im")],
                [parameter("f2(1910) re"), parameter("f2(1910) im")],
            ],
            2,
            &resonance_mass,
            None,
        )
        .expect("f2 should construct");
        let a2 = KopfKMatrixA2::new(
            "a2",
            [
                [parameter("a2(1320) re"), parameter("a2(1320) im")],
                [parameter("a2(1700) re"), parameter("a2(1700) im")],
            ],
            2,
            &resonance_mass,
            None,
        )
        .expect("a2 should construct");

        let s0p = f0p + a0p;
        let s0n = f0n + a0n;
        let d2p = f2 + a2;
        let pos_re = (&s0p * z00p.real() + &d2p * z22p.real()).norm_sqr();
        let pos_im = (&s0p * z00p.imag() + &d2p * z22p.imag()).norm_sqr();
        let neg_re = (&s0n * z00n.real()).norm_sqr();
        let neg_im = (&s0n * z00n.imag()).norm_sqr();
        let expr = pos_re + pos_im + neg_re + neg_im;
        NLL::new(&expr, &ds_data, &ds_mc).expect("k-matrix NLL should build")
    }

    fn kmatrix_nll_mpi_rank_parameterized_benchmarks(c: &mut Criterion) {
        use_mpi(true);
        let Some(world) = get_world() else {
            return;
        };
        let rank_count = world.size();

        let nll = build_kmatrix_nll();
        let params = deterministic_parameter_vector(nll.n_free(), -100.0);
        let n_events = nll.data_evaluator.dataset.n_events() as u64;
        let mut group = c.benchmark_group("kmatrix_nll_mpi_rank_parameterized");
        group.throughput(Throughput::Elements(n_events));
        group.bench_with_input(
            BenchmarkId::new("value_only", rank_count),
            &rank_count,
            |b, &_rank_count| {
                b.iter_batched(
                    || params.clone(),
                    |parameters| black_box(nll.evaluate(&parameters)),
                    BatchSize::SmallInput,
                )
            },
        );
        group.throughput(Throughput::Elements(n_events));
        group.bench_with_input(
            BenchmarkId::new("value_and_gradient", rank_count),
            &rank_count,
            |b, &_rank_count| {
                b.iter_batched(
                    || params.clone(),
                    |parameters| {
                        let value = nll.evaluate(&parameters);
                        let gradient = nll.evaluate_gradient(&parameters);
                        black_box((value, gradient))
                    },
                    BatchSize::SmallInput,
                )
            },
        );
        group.throughput(Throughput::Elements(n_events));
        group.bench_with_input(
            BenchmarkId::new("project_only", rank_count),
            &rank_count,
            |b, &_rank_count| {
                b.iter_batched(
                    || params.clone(),
                    |parameters| black_box(nll.project(&parameters, None)),
                    BatchSize::SmallInput,
                )
            },
        );
        group.throughput(Throughput::Elements(n_events));
        group.bench_with_input(
            BenchmarkId::new("project_gradient", rank_count),
            &rank_count,
            |b, &_rank_count| {
                b.iter_batched(
                    || params.clone(),
                    |parameters| black_box(nll.project_gradient(&parameters, None)),
                    BatchSize::SmallInput,
                )
            },
        );
        group.finish();
    }

    fn mpi_collective_overhead_isolation_benchmarks(c: &mut Criterion) {
        use_mpi(true);
        let Some(world) = get_world() else {
            return;
        };
        let rank_count = world.size();
        let rank_count_usize = rank_count as usize;
        let local_len: usize = 2048;
        let local_buffer: Vec<f64> = (0..local_len)
            .map(|idx| idx as f64 + world.rank() as f64 * 0.001)
            .collect();

        let mut group = c.benchmark_group("mpi_collective_overhead_isolation");
        group.throughput(Throughput::Elements(local_len as u64));
        group.bench_with_input(
            BenchmarkId::new("compute_only_local", rank_count),
            &rank_count,
            |b, &_rank_count| {
                b.iter(|| {
                    let mut acc = 0.0f64;
                    for value in &local_buffer {
                        acc += value * value;
                    }
                    black_box(acc)
                })
            },
        );

        group.throughput(Throughput::Elements((local_len * rank_count_usize) as u64));
        group.bench_with_input(
            BenchmarkId::new("collective_only_all_gather", rank_count),
            &rank_count,
            |b, &_rank_count| {
                b.iter_batched(
                    || vec![0.0f64; local_len * rank_count_usize],
                    |mut global_buffer| {
                        world.all_gather_into(&local_buffer, &mut global_buffer);
                        black_box(global_buffer)
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        group.throughput(Throughput::Elements((local_len * rank_count_usize) as u64));
        group.bench_with_input(
            BenchmarkId::new("compute_plus_collective_all_gather", rank_count),
            &rank_count,
            |b, &_rank_count| {
                b.iter_batched(
                    || vec![0.0f64; local_len * rank_count_usize],
                    |mut global_buffer| {
                        let mut computed = vec![0.0f64; local_len];
                        for (idx, value) in local_buffer.iter().enumerate() {
                            computed[idx] = value * value;
                        }
                        world.all_gather_into(&computed, &mut global_buffer);
                        black_box(global_buffer)
                    },
                    BatchSize::SmallInput,
                )
            },
        );
        group.finish();
    }

    fn mpi_reduction_materialization_microbenchmarks(c: &mut Criterion) {
        use_mpi(true);
        let Some(world) = get_world() else {
            return;
        };
        let rank_count = world.size();
        let rank_count_usize = rank_count as usize;
        let reduction_len = 16_384usize;
        let reduction_local: Vec<f64> = (0..reduction_len)
            .map(|idx| (idx as f64 + 1.0) * (world.rank() as f64 + 1.0))
            .collect();
        let gather_local_len = 8_192usize;
        let gather_local: Vec<f64> = (0..gather_local_len)
            .map(|idx| idx as f64 + world.rank() as f64 * 0.01)
            .collect();
        let gather_global_len = gather_local_len * rank_count_usize;
        let temp_sizes_bytes = [64 * 1024usize, 1024 * 1024usize, 8 * 1024 * 1024usize];

        let mut reduction_group = c.benchmark_group("mpi_reduction_latency");
        reduction_group.throughput(Throughput::Elements(reduction_len as u64));
        reduction_group.bench_with_input(
            BenchmarkId::new("all_reduce_vector_sum", rank_count),
            &rank_count,
            |b, &_rank_count| {
                b.iter_batched(
                    || vec![0.0f64; reduction_len],
                    |mut global_reduced| {
                        world.all_reduce_into(
                            reduction_local.as_slice(),
                            global_reduced.as_mut_slice(),
                            SystemOperation::sum(),
                        );
                        black_box(global_reduced)
                    },
                    BatchSize::SmallInput,
                )
            },
        );
        reduction_group.finish();

        let mut materialization_group = c.benchmark_group("mpi_materialization_temp_buffer");
        for &temp_buffer_bytes in &temp_sizes_bytes {
            materialization_group.throughput(Throughput::Bytes(temp_buffer_bytes as u64));
            materialization_group.bench_with_input(
                BenchmarkId::new(
                    format!(
                        "all_gather_then_materialize_temp_{}KiB",
                        temp_buffer_bytes / 1024
                    ),
                    rank_count,
                ),
                &temp_buffer_bytes,
                |b, &_temp_buffer_bytes| {
                    b.iter_batched(
                        || {
                            (
                                vec![0.0f64; gather_global_len],
                                vec![0.0f64; temp_buffer_bytes / std::mem::size_of::<f64>()],
                            )
                        },
                        |mut global_buffer| {
                            world.all_gather_into(gather_local.as_slice(), &mut global_buffer.0);
                            for (idx, value) in global_buffer.1.iter_mut().enumerate() {
                                let source_idx = idx % global_buffer.0.len();
                                *value = global_buffer.0[source_idx];
                            }
                            black_box(global_buffer)
                        },
                        BatchSize::SmallInput,
                    )
                },
            );
        }
        materialization_group.finish();
    }

    pub fn run() {
        let mut criterion = Criterion::default().sample_size(120).configure_from_args();
        kmatrix_nll_mpi_rank_parameterized_benchmarks(&mut criterion);
        mpi_collective_overhead_isolation_benchmarks(&mut criterion);
        mpi_reduction_materialization_microbenchmarks(&mut criterion);
        criterion.final_summary();
        finalize_mpi();
    }
}

#[cfg(feature = "mpi")]
fn main() {
    mpi_benches::run();
}
