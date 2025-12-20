//! Memory sampling helper for dataset loads.
//!
//! Without the `mpi` feature, this simply records RSS before/after loading. With
//! the feature enabled, run it under `mpirun` so each rank reports its usage:
//! `mpirun -n 4 cargo run -p laddu-core --example mpi_memtrace --features mpi -- path/to/data.parquet`
//! (the dataset argument defaults to `test_data/data_f32.parquet`).

use laddu_core::data::{Dataset, DatasetReadOptions};
use std::error::Error;
use std::path::{Path, PathBuf};
use sysinfo::{get_current_pid, Process, ProcessesToUpdate, System};

#[cfg(feature = "mpi")]
use {
    ::mpi::traits::{Communicator, CommunicatorCollectives},
    laddu_core::mpi,
};

fn main() -> Result<(), Box<dyn Error>> {
    #[cfg(feature = "mpi")]
    {
        run_mpi()?;
        return Ok(());
    }

    #[cfg(not(feature = "mpi"))]
    {
        run_local()?;
        return Ok(());
    }
}

#[cfg(feature = "mpi")]
fn run_mpi() -> Result<(), Box<dyn Error>> {
    mpi::use_mpi(true);
    let world = mpi::get_world().expect("MPI world should be available");
    let rank = world.rank();

    let data_path = dataset_path_from_args();
    let data_path_str = data_path
        .to_str()
        .ok_or_else(|| "Dataset path must be valid UTF-8")?;

    if rank == 0 {
        eprintln!(
            "Loading dataset '{}' across {} ranks",
            data_path.display(),
            world.size()
        );
    }

    let mut sys = System::new();
    let before_mb = sample_rss_mb(&mut sys);
    world.barrier();

    let options = DatasetReadOptions::new();
    let dataset = load_dataset(data_path_str, &data_path, &options)?;

    world.barrier();
    let after_mb = sample_rss_mb(&mut sys);
    let delta_mb = after_mb - before_mb;
    let local_events = dataset.n_events_local();

    let local_stats = [before_mb, after_mb, delta_mb, local_events as f64];
    let mut gathered = vec![0.0f64; local_stats.len() * world.size() as usize];
    world.all_gather_into(&local_stats, &mut gathered);

    if rank == 0 {
        for r in 0..world.size() as usize {
            let base = r * local_stats.len();
            let before = gathered[base];
            let after = gathered[base + 1];
            let delta = gathered[base + 2];
            let events = gathered[base + 3] as usize;
            println!(
                "rank {:>2}: start {:>8.2} MB -> end {:>8.2} MB (Δ {:>7.2} MB), local events {}",
                r, before, after, delta, events
            );
        }
    }

    mpi::finalize_mpi();
    Ok(())
}

#[cfg(not(feature = "mpi"))]
fn run_local() -> Result<(), Box<dyn Error>> {
    let data_path = dataset_path_from_args();
    let data_path_str = data_path
        .to_str()
        .ok_or_else(|| "Dataset path must be valid UTF-8")?;

    eprintln!(
        "Loading dataset '{}' without MPI (single process)",
        data_path.display()
    );

    let mut sys = System::new();
    let before_mb = sample_rss_mb(&mut sys);

    let options = DatasetReadOptions::new();
    let dataset = load_dataset(data_path_str, &data_path, &options)?;

    let after_mb = sample_rss_mb(&mut sys);
    let delta_mb = after_mb - before_mb;

    println!(
        "start {:>8.2} MB -> end {:>8.2} MB (Δ {:>7.2} MB), events {}",
        before_mb,
        after_mb,
        delta_mb,
        dataset.n_events()
    );

    Ok(())
}

fn load_dataset(
    data_path_str: &str,
    data_path: &Path,
    options: &DatasetReadOptions,
) -> Result<std::sync::Arc<Dataset>, Box<dyn Error>> {
    match extension_of(data_path).as_str() {
        "parquet" => Ok(Dataset::from_parquet(data_path_str, options)?),
        "root" => Ok(Dataset::from_root(data_path_str, options)?),
        other => Err(format!("Unsupported dataset extension '{other}'").into()),
    }
}

fn dataset_path_from_args() -> PathBuf {
    let default = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("test_data")
        .join("data_f32.parquet");
    std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or(default)
}

fn extension_of(path: &Path) -> String {
    path.extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
}

fn sample_rss_mb(sys: &mut System) -> f64 {
    let pid = get_current_pid().expect("current PID");
    sys.refresh_processes(ProcessesToUpdate::Some(&[pid.into()]), true);
    sys.process(pid)
        .map(|proc: &Process| proc.memory() as f64 / 1024.0)
        .unwrap_or(0.0)
}
