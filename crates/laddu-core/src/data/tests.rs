use std::{
    env,
    fmt::Display,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

#[cfg(feature = "rayon")]
use accurate::{sum::Klein, traits::*};
use approx::{assert_relative_eq, assert_relative_ne};
use fastrand;
#[cfg(feature = "mpi")]
use mpi::traits::*;
#[cfg(feature = "mpi")]
use mpi_test::mpi_test;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::*;
#[cfg(feature = "mpi")]
use crate::mpi::{finalize_mpi, get_world, use_mpi, LadduMPI};
use crate::{traits::Variable, vectors::Vec3, LadduError, LadduResult, Mass, Vec4};

fn test_data_path(file: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("test_data")
        .join(file)
}

fn open_test_dataset(file: &str, options: DatasetReadOptions) -> Arc<Dataset> {
    let path = test_data_path(file);
    let path_str = path.to_str().expect("test data path should be valid UTF-8");
    let ext = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    match ext.as_str() {
        "parquet" => read_parquet(path_str, &options),
        "root" => read_root(path_str, &options),
        other => panic!("Unsupported extension in test data: {other}"),
    }
    .expect("dataset should open")
}

fn make_temp_dir() -> PathBuf {
    let dir = env::temp_dir().join(format!("laddu_test_{}", fastrand::u64(..)));
    fs::create_dir(&dir).expect("temp dir should be created");
    dir
}

#[cfg(feature = "mpi")]
fn mpi_chunk_test_dataset(n_events: usize) -> Dataset {
    let metadata = test_dataset().metadata_arc();
    let base = test_event();
    let events = (0..n_events)
        .map(|index| {
            let mut event = base.clone();
            event.p4s[0] =
                Vec3::new(index as f64 * 0.1, 0.0, 8.747 + index as f64 * 0.01).with_mass(0.0);
            event.aux[0] += index as f64;
            event.aux[1] += index as f64 * 0.5;
            event.weight = 1.0 + index as f64;
            Arc::new(event)
        })
        .collect();
    Dataset::new_with_metadata(events, metadata)
}

fn assert_events_close(left: &Event, right: &Event, p4_names: &[&str], aux_names: &[&str]) {
    for name in p4_names {
        let lp4 = left
            .p4(name)
            .unwrap_or_else(|| panic!("missing p4 '{name}' in left dataset"));
        let rp4 = right
            .p4(name)
            .unwrap_or_else(|| panic!("missing p4 '{name}' in right dataset"));
        assert_relative_eq!(lp4.px(), rp4.px(), epsilon = 1e-9);
        assert_relative_eq!(lp4.py(), rp4.py(), epsilon = 1e-9);
        assert_relative_eq!(lp4.pz(), rp4.pz(), epsilon = 1e-9);
        assert_relative_eq!(lp4.e(), rp4.e(), epsilon = 1e-9);
    }
    let left_aux = left.aux();
    let right_aux = right.aux();
    for name in aux_names {
        let laux = left_aux
            .get(name)
            .copied()
            .unwrap_or_else(|| panic!("missing aux '{name}' in left dataset"));
        let raux = right_aux
            .get(name)
            .copied()
            .unwrap_or_else(|| panic!("missing aux '{name}' in right dataset"));
        assert_relative_eq!(laux, raux, epsilon = 1e-9);
    }
    assert_relative_eq!(left.weight(), right.weight(), epsilon = 1e-9);
}

fn assert_datasets_close(
    left: &Arc<Dataset>,
    right: &Arc<Dataset>,
    p4_names: &[&str],
    aux_names: &[&str],
) {
    assert_eq!(left.n_events(), right.n_events());
    for idx in 0..left.n_events() {
        let Ok(levent) = left.event(idx) else {
            panic!("left dataset missing event at index {idx}");
        };
        let Ok(revent) = right.event(idx) else {
            panic!("right dataset missing event at index {idx}");
        };
        assert_events_close(&levent, &revent, p4_names, aux_names);
    }
}

fn assert_dataset_columnar_close(left: &DatasetStorage, right: &DatasetStorage) {
    assert_eq!(left.n_events(), right.n_events());
    assert_eq!(left.metadata().p4_names(), right.metadata().p4_names());
    assert_eq!(left.metadata().aux_names(), right.metadata().aux_names());
    for event_index in 0..left.n_events() {
        for p4_index in 0..left.metadata().p4_names().len() {
            let lp4 = left.p4(event_index, p4_index);
            let rp4 = right.p4(event_index, p4_index);
            assert_relative_eq!(lp4.px(), rp4.px(), epsilon = 1e-12);
            assert_relative_eq!(lp4.py(), rp4.py(), epsilon = 1e-12);
            assert_relative_eq!(lp4.pz(), rp4.pz(), epsilon = 1e-12);
            assert_relative_eq!(lp4.e(), rp4.e(), epsilon = 1e-12);
        }
        for aux_index in 0..left.metadata().aux_names().len() {
            let l = left.aux(event_index, aux_index);
            let r = right.aux(event_index, aux_index);
            assert_relative_eq!(l, r, epsilon = 1e-12);
        }
        let lw = left.weight(event_index);
        let rw = right.weight(event_index);
        assert_relative_eq!(lw, rw, epsilon = 1e-12);
    }
}

#[test]
fn test_from_parquet_auto_matches_explicit_names() {
    let auto = open_test_dataset("data_f32.parquet", DatasetReadOptions::new());
    let explicit_options = DatasetReadOptions::new()
        .p4_names(TEST_P4_NAMES)
        .aux_names(TEST_AUX_NAMES);
    let explicit = open_test_dataset("data_f32.parquet", explicit_options);

    let mut detected_p4: Vec<&str> = auto.p4_names().iter().map(String::as_str).collect();
    detected_p4.sort_unstable();
    let mut expected_p4 = TEST_P4_NAMES.to_vec();
    expected_p4.sort_unstable();
    assert_eq!(detected_p4, expected_p4);
    let mut detected_aux: Vec<&str> = auto.aux_names().iter().map(String::as_str).collect();
    detected_aux.sort_unstable();
    let mut expected_aux = TEST_AUX_NAMES.to_vec();
    expected_aux.sort_unstable();
    assert_eq!(detected_aux, expected_aux);
    assert_datasets_close(&auto, &explicit, TEST_P4_NAMES, TEST_AUX_NAMES);
}

#[test]
fn test_from_parquet_with_aliases() {
    let dataset = open_test_dataset(
        "data_f32.parquet",
        DatasetReadOptions::new().alias("resonance", ["kshort1", "kshort2"]),
    );
    let event = dataset.named_event(0).expect("event should exist");
    let alias_vec = event.p4("resonance").expect("alias vector");
    let expected = event.get_p4_sum(["kshort1", "kshort2"]);
    assert_relative_eq!(alias_vec.px(), expected.px(), epsilon = 1e-9);
    assert_relative_eq!(alias_vec.py(), expected.py(), epsilon = 1e-9);
    assert_relative_eq!(alias_vec.pz(), expected.pz(), epsilon = 1e-9);
    assert_relative_eq!(alias_vec.e(), expected.e(), epsilon = 1e-9);
}

#[test]
fn test_from_parquet_alias_resolution_parity_auto_vs_explicit() {
    let auto = open_test_dataset(
        "data_f32.parquet",
        DatasetReadOptions::new().alias("resonance", ["kshort1", "kshort2"]),
    );
    let explicit = open_test_dataset(
        "data_f32.parquet",
        DatasetReadOptions::new()
            .p4_names(TEST_P4_NAMES)
            .aux_names(TEST_AUX_NAMES)
            .alias("resonance", ["kshort1", "kshort2"]),
    );

    assert_datasets_close(&auto, &explicit, TEST_P4_NAMES, TEST_AUX_NAMES);
    for event_index in 0..auto.n_events() {
        let auto_event = auto
            .named_event(event_index)
            .expect("auto parquet event should exist");
        let explicit_event = explicit
            .named_event(event_index)
            .expect("explicit parquet event should exist");

        let auto_alias = auto_event
            .p4("resonance")
            .expect("auto alias should resolve");
        let explicit_alias = explicit_event
            .p4("resonance")
            .expect("explicit alias should resolve");
        let auto_expected = auto_event.get_p4_sum(["kshort1", "kshort2"]);
        let explicit_expected = explicit_event.get_p4_sum(["kshort1", "kshort2"]);

        assert_relative_eq!(auto_alias.px(), auto_expected.px(), epsilon = 1e-9);
        assert_relative_eq!(auto_alias.py(), auto_expected.py(), epsilon = 1e-9);
        assert_relative_eq!(auto_alias.pz(), auto_expected.pz(), epsilon = 1e-9);
        assert_relative_eq!(auto_alias.e(), auto_expected.e(), epsilon = 1e-9);

        assert_relative_eq!(explicit_alias.px(), explicit_expected.px(), epsilon = 1e-9);
        assert_relative_eq!(explicit_alias.py(), explicit_expected.py(), epsilon = 1e-9);
        assert_relative_eq!(explicit_alias.pz(), explicit_expected.pz(), epsilon = 1e-9);
        assert_relative_eq!(explicit_alias.e(), explicit_expected.e(), epsilon = 1e-9);
    }
}

#[test]
fn test_from_parquet_f64_matches_f32() {
    let f32_ds = open_test_dataset("data_f32.parquet", DatasetReadOptions::new());
    let f64_ds = open_test_dataset("data_f64.parquet", DatasetReadOptions::new());
    assert_datasets_close(&f64_ds, &f32_ds, TEST_P4_NAMES, TEST_AUX_NAMES);
}

#[test]
fn test_from_root_detects_columns_and_matches_parquet() {
    let parquet = open_test_dataset("data_f32.parquet", DatasetReadOptions::new());
    let root_auto = open_test_dataset("data_f32.root", DatasetReadOptions::new());
    let mut detected_p4: Vec<&str> = root_auto.p4_names().iter().map(String::as_str).collect();
    detected_p4.sort_unstable();
    let mut expected_p4 = TEST_P4_NAMES.to_vec();
    expected_p4.sort_unstable();
    assert_eq!(detected_p4, expected_p4);
    let mut detected_aux: Vec<&str> = root_auto.aux_names().iter().map(String::as_str).collect();
    detected_aux.sort_unstable();
    let mut expected_aux = TEST_AUX_NAMES.to_vec();
    expected_aux.sort_unstable();
    assert_eq!(detected_aux, expected_aux);
    let root_named_options = DatasetReadOptions::new()
        .p4_names(TEST_P4_NAMES)
        .aux_names(TEST_AUX_NAMES);
    let root_named = open_test_dataset("data_f32.root", root_named_options);
    assert_datasets_close(&root_auto, &root_named, TEST_P4_NAMES, TEST_AUX_NAMES);
    assert_datasets_close(&root_auto, &parquet, TEST_P4_NAMES, TEST_AUX_NAMES);
}

#[cfg(feature = "mpi")]
#[mpi_test(np = [2])]
fn test_from_root_metadata_matches_non_mpi_under_mpi() {
    let reference_auto = open_test_dataset("data_f32.root", DatasetReadOptions::new());
    let explicit_options = DatasetReadOptions::new()
        .p4_names(TEST_P4_NAMES)
        .aux_names(TEST_AUX_NAMES);
    let reference_explicit = open_test_dataset("data_f32.root", explicit_options.clone());

    use_mpi(true);
    let local_auto = open_test_dataset("data_f32.root", DatasetReadOptions::new());
    let local_explicit = open_test_dataset("data_f32.root", explicit_options);

    assert_eq!(local_auto.p4_names(), reference_auto.p4_names());
    assert_eq!(local_auto.aux_names(), reference_auto.aux_names());
    assert_eq!(local_explicit.p4_names(), reference_explicit.p4_names());
    assert_eq!(local_explicit.aux_names(), reference_explicit.aux_names());
    assert_eq!(local_auto.p4_names(), local_explicit.p4_names());
    assert_eq!(local_auto.aux_names(), local_explicit.aux_names());

    for name in local_auto.p4_names() {
        let local_auto_selection = local_auto
            .metadata()
            .p4_selection(name)
            .expect("local auto canonical p4 selection should exist");
        let reference_auto_selection = reference_auto
            .metadata()
            .p4_selection(name)
            .expect("reference auto canonical p4 selection should exist");
        let local_explicit_selection = local_explicit
            .metadata()
            .p4_selection(name)
            .expect("local explicit canonical p4 selection should exist");
        assert_eq!(
            local_auto_selection.names(),
            reference_auto_selection.names()
        );
        assert_eq!(
            local_auto_selection.indices(),
            reference_auto_selection.indices()
        );
        assert_eq!(
            local_explicit_selection.names(),
            reference_auto_selection.names()
        );
        assert_eq!(
            local_explicit_selection.indices(),
            reference_auto_selection.indices()
        );
    }

    finalize_mpi();
}

#[test]
fn test_from_root_f64_matches_parquet() {
    let parquet = open_test_dataset("data_f32.parquet", DatasetReadOptions::new());
    let root_f64 = open_test_dataset("data_f64.root", DatasetReadOptions::new());
    assert_datasets_close(&root_f64, &parquet, TEST_P4_NAMES, TEST_AUX_NAMES);
}

#[cfg(feature = "mpi")]
#[mpi_test(np = [2])]
fn test_from_root_alias_resolution_matches_non_mpi_under_mpi() {
    let alias_options = DatasetReadOptions::new().alias("resonance", ["kshort1", "kshort2"]);
    let explicit_alias_options = DatasetReadOptions::new()
        .p4_names(TEST_P4_NAMES)
        .aux_names(TEST_AUX_NAMES)
        .alias("resonance", ["kshort1", "kshort2"]);
    let reference_auto = open_test_dataset("data_f32.root", alias_options.clone());
    let reference_explicit = open_test_dataset("data_f32.root", explicit_alias_options.clone());

    use_mpi(true);
    let world = get_world().expect("MPI world should be initialized");
    let local_auto = open_test_dataset("data_f32.root", alias_options);
    let local_explicit = open_test_dataset("data_f32.root", explicit_alias_options);

    let local_auto_alias = local_auto
        .metadata()
        .p4_selection("resonance")
        .expect("local auto alias should exist");
    let local_explicit_alias = local_explicit
        .metadata()
        .p4_selection("resonance")
        .expect("local explicit alias should exist");
    let reference_alias = reference_auto
        .metadata()
        .p4_selection("resonance")
        .expect("reference alias should exist");
    let reference_explicit_alias = reference_explicit
        .metadata()
        .p4_selection("resonance")
        .expect("reference explicit alias should exist");
    assert_eq!(local_auto_alias.names(), reference_alias.names());
    assert_eq!(local_auto_alias.indices(), reference_alias.indices());
    assert_eq!(
        local_explicit_alias.names(),
        reference_explicit_alias.names()
    );
    assert_eq!(
        local_explicit_alias.indices(),
        reference_explicit_alias.indices()
    );
    assert_eq!(local_auto_alias.names(), local_explicit_alias.names());
    assert_eq!(local_auto_alias.indices(), local_explicit_alias.indices());

    let partition = world.partition(reference_auto.n_events());
    let local_range = partition.range_for_rank(world.rank() as usize);
    assert_eq!(local_auto.n_events_local(), local_range.len());
    assert_eq!(local_explicit.n_events_local(), local_range.len());

    for (local_index, global_index) in local_range.enumerate() {
        let local_auto_event = local_auto.event_view(local_index);
        let local_explicit_event = local_explicit.event_view(local_index);
        let reference_event = reference_auto.event_view(global_index);
        let reference_explicit_event = reference_explicit.event_view(global_index);

        let local_auto_value = local_auto_event
            .p4("resonance")
            .expect("local auto alias should resolve");
        let local_explicit_value = local_explicit_event
            .p4("resonance")
            .expect("local explicit alias should resolve");
        let reference_value = reference_event
            .p4("resonance")
            .expect("reference alias should resolve");
        let reference_explicit_value = reference_explicit_event
            .p4("resonance")
            .expect("reference explicit alias should resolve");

        assert_relative_eq!(local_auto_value.px(), reference_value.px(), epsilon = 1e-9);
        assert_relative_eq!(local_auto_value.py(), reference_value.py(), epsilon = 1e-9);
        assert_relative_eq!(local_auto_value.pz(), reference_value.pz(), epsilon = 1e-9);
        assert_relative_eq!(local_auto_value.e(), reference_value.e(), epsilon = 1e-9);

        assert_relative_eq!(
            local_explicit_value.px(),
            reference_explicit_value.px(),
            epsilon = 1e-9
        );
        assert_relative_eq!(
            local_explicit_value.py(),
            reference_explicit_value.py(),
            epsilon = 1e-9
        );
        assert_relative_eq!(
            local_explicit_value.pz(),
            reference_explicit_value.pz(),
            epsilon = 1e-9
        );
        assert_relative_eq!(
            local_explicit_value.e(),
            reference_explicit_value.e(),
            epsilon = 1e-9
        );
    }

    finalize_mpi();
}

#[test]
fn test_event_creation() {
    let event = test_event();
    assert_eq!(event.p4s.len(), 4);
    assert_eq!(event.aux.len(), 2);
    assert_relative_eq!(event.weight, 0.48)
}

#[test]
fn test_event_p4_sum() {
    let event = test_event();
    let sum = event.get_p4_sum([2, 3]);
    assert_relative_eq!(sum.px(), event.p4s[2].px() + event.p4s[3].px());
    assert_relative_eq!(sum.py(), event.p4s[2].py() + event.p4s[3].py());
    assert_relative_eq!(sum.pz(), event.p4s[2].pz() + event.p4s[3].pz());
    assert_relative_eq!(sum.e(), event.p4s[2].e() + event.p4s[3].e());
}

#[test]
fn test_event_boost() {
    let event = test_event();
    let event_boosted = event.boost_to_rest_frame_of([1, 2, 3]);
    let p4_sum = event_boosted.get_p4_sum([1, 2, 3]);
    assert_relative_eq!(p4_sum.px(), 0.0);
    assert_relative_eq!(p4_sum.py(), 0.0);
    assert_relative_eq!(p4_sum.pz(), 0.0, epsilon = f64::EPSILON.sqrt());
}

#[test]
fn test_named_event_view_evaluate() {
    let dataset = test_dataset();
    let event = dataset.event_view(0);
    let mut mass = Mass::new(["proton"]);
    mass.bind(dataset.metadata()).unwrap();
    assert_relative_eq!(event.evaluate(&mass), 1.007);
}

#[test]
fn test_dataset_size_check() {
    let dataset = Dataset::new(Vec::new());
    assert_eq!(dataset.n_events(), 0);
    let dataset = Dataset::new(vec![Arc::new(test_event())]);
    assert_eq!(dataset.n_events(), 1);
}

#[test]
fn test_dataset_sum() {
    let dataset = test_dataset();
    let metadata = dataset.metadata_arc();
    let dataset2 = Dataset::new_with_metadata(
        vec![Arc::new(EventData {
            p4s: test_event().p4s,
            aux: test_event().aux,
            weight: 0.52,
        })],
        metadata.clone(),
    );
    let dataset_sum = &dataset + &dataset2;
    assert_eq!(
        dataset_sum.event(0).expect("event should exist").weight,
        dataset.event(0).expect("event should exist").weight
    );
    assert_eq!(
        dataset_sum.event(1).expect("event should exist").weight,
        dataset2.event(0).expect("event should exist").weight
    );
}

#[test]
fn test_dataset_weights() {
    let dataset = Dataset::new(vec![
        Arc::new(test_event()),
        Arc::new(EventData {
            p4s: test_event().p4s,
            aux: test_event().aux,
            weight: 0.52,
        }),
    ]);
    let weights = dataset.weights();
    assert_eq!(weights.len(), 2);
    assert_relative_eq!(weights[0], 0.48);
    assert_relative_eq!(weights[1], 0.52);
    assert_relative_eq!(dataset.n_events_weighted(), 1.0);
}

#[test]
fn test_dataset_empty_push_event_named_matches_columns() {
    let metadata = DatasetMetadata::new(vec!["beam", "recoil"], vec!["pol_angle"])
        .expect("metadata should be valid");
    let beam = Vec3::new(0.0, 0.0, 8.0).with_mass(0.0);
    let recoil = Vec3::new(0.1, 0.2, 0.3).with_mass(0.938);
    let mut row_dataset = Dataset::empty_local(metadata.clone());
    row_dataset
        .push_event_named_local(
            [("recoil", recoil), ("beam", beam)],
            [("pol_angle", 0.25)],
            2.0,
        )
        .expect("named push should succeed");

    let column_dataset = Dataset::from_columns_local(
        metadata,
        vec![vec![beam], vec![recoil]],
        vec![vec![0.25]],
        vec![2.0],
    )
    .expect("column construction should succeed");

    let row_dataset = Arc::new(row_dataset);
    let column_dataset = Arc::new(column_dataset);
    assert_datasets_close(
        &row_dataset,
        &column_dataset,
        &["beam", "recoil"],
        &["pol_angle"],
    );
}

#[test]
fn test_dataset_push_event_validation() {
    let metadata = DatasetMetadata::new(vec!["beam", "recoil"], vec!["pol_angle"])
        .expect("metadata should be valid");
    let beam = Vec3::new(0.0, 0.0, 8.0).with_mass(0.0);
    let recoil = Vec3::new(0.1, 0.2, 0.3).with_mass(0.938);
    let mut dataset = Dataset::empty_local(metadata);

    assert!(dataset.push_event_local([beam], [0.25], 1.0).is_err());
    assert!(dataset.push_event_local([beam, recoil], [], 1.0).is_err());
    assert!(dataset
        .push_event_named_local([("beam", beam)], [("pol_angle", 0.25)], 1.0)
        .is_err());
    assert!(dataset
        .push_event_named_local(
            [("beam", beam), ("beam", recoil)],
            [("pol_angle", 0.25)],
            1.0
        )
        .is_err());
    assert!(dataset
        .push_event_named_local(
            [("beam", beam), ("unknown", recoil)],
            [("pol_angle", 0.25)],
            1.0
        )
        .is_err());
    assert!(dataset
        .push_event_named_local(
            [("beam", beam), ("recoil", recoil)],
            [("unknown", 0.25)],
            1.0
        )
        .is_err());
}

#[test]
fn test_dataset_explicit_local_global_push_non_mpi() {
    let metadata = DatasetMetadata::new(vec!["beam"], vec!["pol_angle"]).unwrap();
    let mut dataset = Dataset::empty_local(metadata);
    let beam = Vec3::new(0.0, 0.0, 8.0).with_mass(0.0);

    dataset
        .push_event_named_local([("beam", beam)], [("pol_angle", 0.25)], 2.0)
        .unwrap();
    dataset
        .push_event_named_global([("beam", beam)], [("pol_angle", 0.50)], 3.0)
        .unwrap();

    assert_eq!(dataset.n_events_local(), 2);
    assert_eq!(dataset.n_events(), 2);
    assert_relative_eq!(dataset.event_global(0).unwrap().weight(), 2.0);
    assert_relative_eq!(dataset.event_global(1).unwrap().weight(), 3.0);
}

#[cfg(feature = "mpi")]
#[mpi_test(np = [2])]
fn test_dataset_push_event_global_round_robin_mpi() {
    use_mpi(true);
    let world = get_world().expect("MPI world should be initialized");
    let metadata = DatasetMetadata::new(vec!["beam"], vec!["pol_angle"]).unwrap();
    let mut dataset = Dataset::empty_local(metadata);

    for index in 0..4 {
        let pz = index as f64 + 1.0;
        let beam = Vec3::new(0.0, 0.0, pz).with_mass(0.0);
        dataset
            .push_event_named_global([("beam", beam)], [("pol_angle", pz)], pz)
            .unwrap();
    }

    assert_eq!(dataset.n_events(), 4);
    assert_eq!(dataset.n_events_local(), 2);
    let expected_local_weights = if world.rank() == 0 {
        vec![1.0, 3.0]
    } else {
        vec![2.0, 4.0]
    };
    let local_weights = dataset
        .events_local()
        .iter()
        .map(Event::weight)
        .collect::<Vec<_>>();
    assert_eq!(local_weights, expected_local_weights);
    let global_weights = dataset
        .events_global()
        .into_iter()
        .map(|event| event.weight())
        .collect::<Vec<_>>();
    assert_eq!(global_weights, vec![1.0, 2.0, 3.0, 4.0]);
    finalize_mpi();
}

#[test]
fn test_dataset_views_local_evaluate_without_event_clone() {
    let dataset = test_dataset();
    let mut mass = Mass::new(["proton"]);
    mass.bind(dataset.metadata()).unwrap();
    let values = dataset
        .views_local()
        .map(|event| event.evaluate(&mass))
        .collect::<Vec<_>>();
    assert_eq!(values.len(), dataset.n_events_local());
    assert_relative_eq!(values[0], 1.007);
    assert!(dataset.view_local(dataset.n_events_local()).is_err());
}

#[test]
#[should_panic(
    expected = "Dataset requires rectangular p4/aux columns for canonical columnar storage"
)]
fn test_dataset_rejects_ragged_rows_at_construction() {
    let _ = Dataset::new(vec![
        Arc::new(EventData {
            p4s: vec![Vec4::new(0.0, 0.0, 1.0, 1.0)],
            aux: vec![0.1],
            weight: 1.0,
        }),
        Arc::new(EventData {
            p4s: vec![],
            aux: vec![0.2, 0.3],
            weight: 2.0,
        }),
    ]);
}

#[test]
fn test_dataset_filtering() {
    let metadata = Arc::new(
        DatasetMetadata::new(vec!["beam"], Vec::<String>::new()).expect("metadata should be valid"),
    );
    let events = vec![
        Arc::new(EventData {
            p4s: vec![Vec3::new(0.0, 0.0, 5.0).with_mass(0.0)],
            aux: vec![],
            weight: 1.0,
        }),
        Arc::new(EventData {
            p4s: vec![Vec3::new(0.0, 0.0, 5.0).with_mass(0.5)],
            aux: vec![],
            weight: 1.0,
        }),
        Arc::new(EventData {
            p4s: vec![Vec3::new(0.0, 0.0, 5.0).with_mass(1.1)],
            // HACK: using 1.0 messes with this test because the eventual computation gives a mass
            // slightly less than 1.0
            aux: vec![],
            weight: 1.0,
        }),
    ];
    let dataset = Dataset::new_with_metadata(events, metadata);

    let metadata = dataset.metadata_arc();
    let mut mass = Mass::new(["beam"]);
    mass.bind(metadata.as_ref()).unwrap();
    let expression = mass.gt(0.0).and(&mass.lt(1.0));

    let filtered = dataset.filter(&expression).unwrap();
    assert_eq!(filtered.n_events(), 1);
    assert_relative_eq!(mass.value(&filtered.event_view(0)), 0.5);
}

#[test]
fn test_dataset_boost() {
    let dataset = test_dataset();
    let dataset_boosted = dataset.boost_to_rest_frame_of(&["proton", "kshort1", "kshort2"]);
    let p4_sum = dataset_boosted
        .event(0)
        .expect("event should exist")
        .get_p4_sum(["proton", "kshort1", "kshort2"]);
    assert_relative_eq!(p4_sum.px(), 0.0);
    assert_relative_eq!(p4_sum.py(), 0.0);
    assert_relative_eq!(p4_sum.pz(), 0.0, epsilon = f64::EPSILON.sqrt());
}

#[test]
fn test_named_event_view() {
    let dataset = test_dataset();
    let view = dataset.named_event(0).expect("event should exist");
    let dataset_event = dataset.event(0).expect("event should exist");
    assert_relative_eq!(view.weight(), dataset_event.weight);
    let beam = view.p4("beam").expect("beam p4");
    assert_relative_eq!(beam.px(), dataset_event.p4s[0].px());
    assert_relative_eq!(beam.e(), dataset_event.p4s[0].e());

    let summed = view.get_p4_sum(["kshort1", "kshort2"]);
    assert_relative_eq!(
        summed.e(),
        dataset_event.p4s[2].e() + dataset_event.p4s[3].e()
    );

    let aux_angle = view.aux().get("pol_angle").copied().expect("pol angle");
    assert_relative_eq!(aux_angle, dataset_event.aux[1]);

    let metadata = dataset.metadata_arc();
    let boosted = view.boost_to_rest_frame_of(["proton", "kshort1", "kshort2"]);
    let boosted_event = Event::new(Arc::new(boosted), metadata);
    let boosted_sum = boosted_event.get_p4_sum(["proton", "kshort1", "kshort2"]);
    assert_relative_eq!(boosted_sum.px(), 0.0);
}

#[test]
fn test_dataset_evaluate() {
    let dataset = test_dataset();
    let mass = Mass::new(["proton"]);
    assert_relative_eq!(dataset.evaluate(&mass).unwrap()[0], 1.007);
}

#[test]
fn test_dataset_metadata_rejects_duplicate_names() {
    let err = DatasetMetadata::new(vec!["beam", "beam"], Vec::<String>::new());
    assert!(matches!(
        err,
        Err(LadduError::DuplicateName { category, .. }) if category == "p4"
    ));
    let err = DatasetMetadata::new(
        vec!["beam"],
        vec!["pol_angle".to_string(), "pol_angle".to_string()],
    );
    assert!(matches!(
        err,
        Err(LadduError::DuplicateName { category, .. }) if category == "aux"
    ));
}

#[test]
fn test_dataset_lookup_by_name() {
    let dataset = test_dataset();
    let proton = dataset.p4_by_name(0, "proton").expect("proton p4");
    let proton_idx = dataset.metadata().p4_index("proton").unwrap();
    assert_relative_eq!(
        proton.e(),
        dataset.event(0).expect("event should exist").p4s[proton_idx].e()
    );
    assert!(dataset.p4_by_name(0, "unknown").is_none());
    let angle = dataset.aux_by_name(0, "pol_angle").expect("pol_angle");
    assert_relative_eq!(angle, dataset.event(0).expect("event should exist").aux[1]);
    assert!(dataset.aux_by_name(0, "missing").is_none());
}

#[test]
fn test_binned_dataset() {
    let dataset = Dataset::new(vec![
        Arc::new(EventData {
            p4s: vec![Vec3::new(0.0, 0.0, 1.0).with_mass(1.0)],
            aux: vec![],
            weight: 1.0,
        }),
        Arc::new(EventData {
            p4s: vec![Vec3::new(0.0, 0.0, 2.0).with_mass(2.0)],
            aux: vec![],
            weight: 2.0,
        }),
    ]);

    #[derive(Clone, Serialize, Deserialize, Debug)]
    struct BeamEnergy;
    impl Display for BeamEnergy {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "BeamEnergy")
        }
    }
    #[typetag::serde]
    impl Variable for BeamEnergy {
        fn value(&self, event: &NamedEventView<'_>) -> f64 {
            event.p4_at(0).e()
        }
    }
    assert_eq!(BeamEnergy.to_string(), "BeamEnergy");

    // Test binning by first particle energy
    let binned = dataset.bin_by(BeamEnergy, 2, (0.0, 3.0)).unwrap();

    assert_eq!(binned.n_bins(), 2);
    assert_eq!(binned.edges().len(), 3);
    assert_relative_eq!(binned.edges()[0], 0.0);
    assert_relative_eq!(binned.edges()[2], 3.0);
    assert_eq!(binned[0].n_events(), 1);
    assert_relative_eq!(binned[0].n_events_weighted(), 1.0);
    assert_eq!(binned[1].n_events(), 1);
    assert_relative_eq!(binned[1].n_events_weighted(), 2.0);
}

#[test]
fn test_dataset_bootstrap() {
    let metadata = test_dataset().metadata_arc();
    let dataset = Dataset::new_with_metadata(
        vec![
            Arc::new(test_event()),
            Arc::new(EventData {
                p4s: test_event().p4s.clone(),
                aux: test_event().aux.clone(),
                weight: 1.0,
            }),
        ],
        metadata,
    );
    assert_relative_ne!(
        dataset.event(0).expect("event should exist").weight,
        dataset.event(1).expect("event should exist").weight
    );

    let bootstrapped = dataset.bootstrap(43);
    assert_eq!(bootstrapped.n_events(), dataset.n_events());
    assert_relative_eq!(
        bootstrapped.event(0).expect("event should exist").weight,
        bootstrapped.event(1).expect("event should exist").weight
    );

    // Test empty dataset bootstrap
    let empty_dataset = Dataset::new(Vec::new());
    let empty_bootstrap = empty_dataset.bootstrap(43);
    assert_eq!(empty_bootstrap.n_events(), 0);
}

fn assert_weight_cache_matches_local_events(dataset: &Dataset) {
    #[cfg(feature = "rayon")]
    let expected = dataset
        .events_local()
        .par_iter()
        .map(|event| event.weight)
        .parallel_sum_with_accumulator::<Klein<f64>>();
    #[cfg(not(feature = "rayon"))]
    let expected = dataset
        .events_local()
        .iter()
        .map(|event| event.weight)
        .sum_with_accumulator::<Klein<f64>>();
    assert_relative_eq!(dataset.cached_local_weighted_sum, expected);
    assert_relative_eq!(dataset.n_events_weighted_local(), expected);
}

#[test]
fn test_weight_cache_recomputed_for_dataset_transforms() {
    let metadata = Arc::new(
        DatasetMetadata::new(vec!["beam"], Vec::<String>::new()).expect("metadata should be valid"),
    );
    let dataset = Dataset::new_with_metadata(
        vec![
            Arc::new(EventData {
                p4s: vec![Vec3::new(0.0, 0.0, 1.0).with_mass(0.0)],
                aux: vec![],
                weight: 1.0,
            }),
            Arc::new(EventData {
                p4s: vec![Vec3::new(0.0, 0.0, 2.0).with_mass(0.0)],
                aux: vec![],
                weight: 2.0,
            }),
            Arc::new(EventData {
                p4s: vec![Vec3::new(0.0, 0.0, 3.0).with_mass(0.0)],
                aux: vec![],
                weight: 3.0,
            }),
        ],
        metadata,
    );
    assert_weight_cache_matches_local_events(&dataset);

    let filtered = dataset.filter(&Mass::new(["beam"]).gt(0.0)).unwrap();
    assert_weight_cache_matches_local_events(&filtered);

    let bootstrapped = dataset.bootstrap(7);
    assert_weight_cache_matches_local_events(&bootstrapped);

    let boosted = dataset.boost_to_rest_frame_of(&["beam"]);
    assert_weight_cache_matches_local_events(&boosted);

    let combined = &dataset + &dataset;
    assert_weight_cache_matches_local_events(&combined);
}

#[test]
fn test_dataset_iteration_returns_events() {
    let dataset = test_dataset();
    let mut weights = Vec::new();
    for event in dataset.events_global() {
        weights.push(event.weight());
    }
    assert_eq!(weights.len(), dataset.n_events());
    assert_relative_eq!(
        weights[0],
        dataset.event(0).expect("event should exist").weight
    );
}

#[test]
fn test_dataset_events_global_returns_events() {
    let dataset = test_dataset();
    let weights: Vec<f64> = dataset
        .events_global()
        .into_iter()
        .map(|event| event.weight())
        .collect();
    assert_eq!(weights.len(), 1);
    assert_relative_eq!(weights[0], test_event().weight);
}

#[test]
fn test_dataset_arc_into_iter_returns_events() {
    let dataset = Arc::new(test_dataset());
    let weights: Vec<f64> = dataset.shared_iter().map(|event| event.weight()).collect();
    assert_eq!(weights.len(), 1);
    assert_relative_eq!(weights[0], test_event().weight);
}

#[test]
fn test_dataset_get_event_local_reuses_underlying_data() {
    let dataset = test_dataset();
    let first = dataset.get_event(0).expect("event should exist");
    let second = dataset.get_event(0).expect("event should exist");
    assert!(Arc::ptr_eq(&first.data_arc(), &second.data_arc()));
}

#[test]
fn test_dataset_event_out_of_bounds_is_error() {
    let dataset = test_dataset();
    assert!(dataset.event(99).is_err());
    assert!(dataset.get_event(99).is_none());
}

#[cfg(feature = "mpi")]
fn event_iteration_signature<I>(iter: I) -> (usize, f64, f64, f64)
where
    I: IntoIterator<Item = Event>,
{
    let mut count = 0usize;
    let mut weight_signature = 0.0;
    let mut beam_signature = 0.0;
    let mut aux_signature = 0.0;

    for (index, event) in iter.into_iter().enumerate() {
        let position = (index + 1) as f64;
        count += 1;
        weight_signature += position * event.weight();
        beam_signature += position * event.p4("beam").expect("beam should exist").e();
        aux_signature += position
            * event
                .aux()
                .get("pol_angle")
                .copied()
                .expect("pol_angle should exist");
    }

    (count, weight_signature, beam_signature, aux_signature)
}

#[cfg(feature = "mpi")]
fn read_resident_rss_kb() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let status = fs::read_to_string("/proc/self/status").ok()?;
        let vm_rss = status
            .lines()
            .find(|line| line.starts_with("VmRSS:"))?
            .split_whitespace()
            .nth(1)?;
        vm_rss.parse::<u64>().ok()
    }

    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

#[test]
fn test_dataset_event_stress_local_repeated_access() {
    let metadata = test_dataset().metadata_arc();
    let base = test_event();
    let mut events = Vec::new();
    for idx in 0..8 {
        events.push(Arc::new(EventData {
            p4s: base.p4s.clone(),
            aux: base.aux.clone(),
            weight: 1.0 + idx as f64,
        }));
    }
    let dataset = Dataset::new_with_metadata(events, metadata);
    let baseline: Vec<f64> = (0..dataset.n_events())
        .map(|index| dataset.event(index).expect("event should exist").weight())
        .collect();

    for _ in 0..250 {
        for (index, expected_weight) in baseline.iter().enumerate() {
            let event = dataset.event(index).expect("event should exist");
            assert_relative_eq!(event.weight(), *expected_weight);
        }
    }
}

#[cfg(feature = "mpi")]
#[mpi_test(np = [2])]
fn test_dataset_event_mpi_repeated_access_is_stable() {
    use_mpi(true);
    assert!(get_world().is_some(), "MPI world should be initialized");

    let dataset = test_dataset();
    for _ in 0..32 {
        let first = dataset.event(0).expect("event should exist");
        let second = dataset.event(0).expect("event should exist");
        assert_relative_eq!(first.weight(), second.weight());
    }
    finalize_mpi();
}

#[cfg(feature = "mpi")]
#[mpi_test(np = [2])]
fn test_dataset_event_stress_mpi_repeated_access() {
    use_mpi(true);
    assert!(get_world().is_some(), "MPI world should be initialized");

    let metadata = test_dataset().metadata_arc();
    let base = test_event();
    let mut events = Vec::new();
    for idx in 0..8 {
        events.push(Arc::new(EventData {
            p4s: base.p4s.clone(),
            aux: base.aux.clone(),
            weight: 1.0 + idx as f64,
        }));
    }
    let dataset = Dataset::new_with_metadata(events, metadata);

    let baseline: Vec<f64> = (0..dataset.n_events())
        .map(|index| dataset.event(index).expect("event should exist").weight())
        .collect();

    for _ in 0..120 {
        for (index, expected_weight) in baseline.iter().enumerate() {
            let event = dataset.event(index).expect("event should exist");
            assert_relative_eq!(event.weight(), *expected_weight);
        }
    }
    finalize_mpi();
}

#[cfg(feature = "mpi")]
#[mpi_test(np = [2])]
fn test_dataset_iter_stress_mpi_repeated_passes() {
    use_mpi(true);
    assert!(get_world().is_some(), "MPI world should be initialized");

    let metadata = test_dataset().metadata_arc();
    let base = test_event();
    let mut events = Vec::new();
    for idx in 0..8 {
        events.push(Arc::new(EventData {
            p4s: base.p4s.clone(),
            aux: base.aux.clone(),
            weight: 1.0 + idx as f64,
        }));
    }
    let dataset = Dataset::new_with_metadata(events, metadata);
    let baseline: Vec<f64> = dataset
        .events_global()
        .into_iter()
        .map(|event| event.weight())
        .collect();

    for _ in 0..80 {
        let current: Vec<f64> = dataset
            .events_global()
            .into_iter()
            .map(|event| event.weight())
            .collect();
        assert_eq!(current.len(), baseline.len());
        for (current_weight, expected_weight) in current.iter().zip(baseline.iter()) {
            assert_relative_eq!(*current_weight, *expected_weight);
        }
    }
    finalize_mpi();
}

#[cfg(feature = "mpi")]
#[mpi_test(np = [2])]
fn test_dataset_arc_into_iter_stress_mpi_repeated_passes() {
    use_mpi(true);
    assert!(get_world().is_some(), "MPI world should be initialized");

    let metadata = test_dataset().metadata_arc();
    let base = test_event();
    let mut events = Vec::new();
    for idx in 0..8 {
        events.push(Arc::new(EventData {
            p4s: base.p4s.clone(),
            aux: base.aux.clone(),
            weight: 1.0 + idx as f64,
        }));
    }
    let dataset = Arc::new(Dataset::new_with_metadata(events, metadata));
    let baseline: Vec<f64> = dataset.shared_iter().map(|event| event.weight()).collect();

    for _ in 0..80 {
        let current: Vec<f64> = dataset.shared_iter().map(|event| event.weight()).collect();
        assert_eq!(current.len(), baseline.len());
        for (current_weight, expected_weight) in current.iter().zip(baseline.iter()) {
            assert_relative_eq!(*current_weight, *expected_weight);
        }
    }
    finalize_mpi();
}

#[cfg(feature = "mpi")]
#[mpi_test(np = [2])]
fn test_dataset_iteration_long_running_mpi_repeated_passes() {
    use_mpi(true);
    assert!(get_world().is_some(), "MPI world should be initialized");

    let dataset = Arc::new(mpi_chunk_test_dataset(8_192));
    let baseline_iter = event_iteration_signature(dataset.events_global().into_iter());
    let baseline_shared = event_iteration_signature(dataset.shared_iter());
    assert_eq!(baseline_iter, baseline_shared);
    let mut post_warmup_rss_kb = Vec::new();

    for pass_index in 0..48 {
        let current_iter = event_iteration_signature(dataset.events_global().into_iter());
        let current_shared = event_iteration_signature(dataset.shared_iter());
        assert_eq!(current_iter, baseline_iter);
        assert_eq!(current_shared, baseline_shared);
        if pass_index >= 7 {
            if let Some(rss_kb) = read_resident_rss_kb() {
                post_warmup_rss_kb.push(rss_kb);
            }
        }
    }

    if let Some((&first_rss_kb, rest_rss_kb)) = post_warmup_rss_kb.split_first() {
        let last_rss_kb = *rest_rss_kb.last().unwrap_or(&first_rss_kb);
        let min_rss_kb = post_warmup_rss_kb
            .iter()
            .copied()
            .min()
            .expect("post-warmup RSS sample should exist");
        let max_rss_kb = post_warmup_rss_kb
            .iter()
            .copied()
            .max()
            .expect("post-warmup RSS sample should exist");
        const MAX_POST_WARMUP_RSS_GROWTH_KB: u64 = 32 * 1024;
        const MAX_POST_WARMUP_RSS_SPREAD_KB: u64 = 32 * 1024;
        assert!(
            last_rss_kb.saturating_sub(first_rss_kb) <= MAX_POST_WARMUP_RSS_GROWTH_KB,
            "post-warmup RSS grew by {} KiB (first={} KiB, last={} KiB)",
            last_rss_kb.saturating_sub(first_rss_kb),
            first_rss_kb,
            last_rss_kb
        );
        assert!(
            max_rss_kb.saturating_sub(min_rss_kb) <= MAX_POST_WARMUP_RSS_SPREAD_KB,
            "post-warmup RSS spread was {} KiB (min={} KiB, max={} KiB)",
            max_rss_kb.saturating_sub(min_rss_kb),
            min_rss_kb,
            max_rss_kb
        );
    }

    finalize_mpi();
}

#[cfg(feature = "mpi")]
#[mpi_test(np = [2])]
fn test_fetch_event_chunk_mpi_matches_single_event_fetches() {
    use_mpi(true);
    let world = get_world().expect("MPI world should be initialized");

    let dataset = mpi_chunk_test_dataset(8);
    let chunk = fetch_event_chunk_mpi(
        &dataset,
        1,
        5,
        &world,
        dataset.n_events(),
        MpiDatasetLayout::Canonical,
    );

    assert_eq!(chunk.len(), 5);
    for (offset, event) in chunk.iter().enumerate() {
        let baseline = dataset
            .event(1 + offset)
            .expect("chunk baseline event should exist");
        assert_events_close(event, &baseline, TEST_P4_NAMES, TEST_AUX_NAMES);
    }

    assert!(fetch_event_chunk_mpi(
        &dataset,
        dataset.n_events(),
        4,
        &world,
        dataset.n_events(),
        MpiDatasetLayout::Canonical,
    )
    .is_empty());
    finalize_mpi();
}

#[cfg(feature = "mpi")]
#[mpi_test(np = [2])]
fn test_fetch_event_chunk_mpi_truncates_at_dataset_end() {
    use_mpi(true);
    let world = get_world().expect("MPI world should be initialized");

    let dataset = mpi_chunk_test_dataset(8);
    let chunk = fetch_event_chunk_mpi(
        &dataset,
        6,
        10,
        &world,
        dataset.n_events(),
        MpiDatasetLayout::Canonical,
    );

    assert_eq!(chunk.len(), 2);
    for (offset, event) in chunk.iter().enumerate() {
        let baseline = dataset
            .event(6 + offset)
            .expect("truncated chunk baseline event should exist");
        assert_events_close(event, &baseline, TEST_P4_NAMES, TEST_AUX_NAMES);
    }
    finalize_mpi();
}

#[cfg(feature = "mpi")]
#[mpi_test(np = [2])]
fn test_mpi_event_chunk_cursor_reuses_cached_chunk_for_dataset_and_events() {
    use_mpi(true);
    let world = get_world().expect("MPI world should be initialized");

    let dataset = mpi_chunk_test_dataset(9);
    let total = dataset.n_events();

    let mut dataset_cursor = MpiEventChunkCursor::new(3);
    for index in 0..total {
        let actual = dataset_cursor
            .event_for_dataset(&dataset, index, &world, total, MpiDatasetLayout::Canonical)
            .expect("dataset cursor event should exist");
        let expected = dataset.event(index).expect("baseline event should exist");
        assert_events_close(&actual, &expected, TEST_P4_NAMES, TEST_AUX_NAMES);
    }
    assert!(dataset_cursor
        .event_for_dataset(&dataset, total, &world, total, MpiDatasetLayout::Canonical)
        .is_none());

    let mut events_cursor = MpiEventChunkCursor::new(4);
    for index in 0..total {
        let actual = events_cursor
            .event_for_dataset(&dataset, index, &world, total, MpiDatasetLayout::Canonical)
            .expect("events cursor event should exist");
        let expected = dataset.event(index).expect("baseline event should exist");
        assert_events_close(&actual, &expected, TEST_P4_NAMES, TEST_AUX_NAMES);
    }
    finalize_mpi();
}

#[cfg(feature = "mpi")]
#[test]
#[ignore = "developer probe for MPI iteration chunk-size tuning"]
fn probe_mpi_iteration_chunk_size() {
    use std::time::Instant;

    use_mpi(true);
    let Some(world) = get_world() else {
        finalize_mpi();
        return;
    };

    let dataset = mpi_chunk_test_dataset(32_768);
    let total = dataset.n_events();
    let chunk_sizes = [64_usize, 128, 256, 512, 1024];
    if world.rank() == 0 {
        println!("probe=iteration");
    }
    for chunk_size in chunk_sizes {
        let started = Instant::now();
        let mut checksum = 0.0;
        for _ in 0..8 {
            let mut cursor = MpiEventChunkCursor::new(chunk_size);
            for index in 0..total {
                let event = cursor
                    .event_for_dataset(&dataset, index, &world, total, MpiDatasetLayout::Canonical)
                    .expect("cursor event should exist");
                checksum += event.weight() + event.p4("beam").expect("beam should exist").e();
            }
        }
        if world.rank() == 0 {
            println!(
                "probe=iteration chunk_size={} elapsed_sec={:.6} checksum={:.6}",
                chunk_size,
                started.elapsed().as_secs_f64(),
                checksum,
            );
        }
    }
    finalize_mpi();
}

#[cfg(feature = "mpi")]
#[test]
#[ignore = "developer probe for MPI ROOT write chunk-size tuning"]
fn probe_mpi_root_write_chunk_size() {
    use std::time::Instant;

    use_mpi(true);
    let Some(world) = get_world() else {
        finalize_mpi();
        return;
    };

    let dataset = Arc::new(mpi_chunk_test_dataset(32_768));
    let chunk_sizes = [64_usize, 128, 256, 512, 1024];
    if world.rank() == 0 {
        println!("probe=root_write");
    }
    for chunk_size in chunk_sizes {
        let dir = make_temp_dir();
        let path = dir.join(format!("mpi_chunk_probe_{chunk_size}.root"));
        let path_str = path.to_str().expect("probe path should be valid UTF-8");
        let started = Instant::now();
        for _ in 0..4 {
            io::write_root_with_chunk_size_for_test(
                &dataset,
                path_str,
                &DatasetWriteOptions::default(),
                chunk_size,
            )
            .expect("probe root write should succeed");
        }

        if world.rank() == 0 {
            println!(
                "probe=root_write chunk_size={} elapsed_sec={:.6}",
                chunk_size,
                started.elapsed().as_secs_f64(),
            );
            fs::remove_dir_all(&dir).expect("probe temp dir cleanup should succeed");
        }
    }
    finalize_mpi();
}

#[test]
fn test_event_display() {
    let event = test_event();
    let display_string = format!("{}", event);
    assert!(display_string.contains("Event:"));
    assert!(display_string.contains("p4s:"));
    assert!(display_string.contains("aux:"));
    assert!(display_string.contains("aux[0]: 0.38562805"));
    assert!(display_string.contains("aux[1]: 0.05708078"));
    assert!(display_string.contains("weight:"));
}

#[test]
fn test_name_based_access() {
    let metadata =
        Arc::new(DatasetMetadata::new(vec!["beam", "target"], vec!["pol_angle"]).unwrap());
    let event = Arc::new(EventData {
        p4s: vec![Vec4::new(0.0, 0.0, 1.0, 1.0), Vec4::new(0.1, 0.2, 0.3, 0.5)],
        aux: vec![0.42],
        weight: 1.0,
    });
    let dataset = Dataset::new_with_metadata(vec![event], metadata);
    let beam = dataset.p4_by_name(0, "beam").unwrap();
    assert_relative_eq!(beam.px(), 0.0);
    assert_relative_eq!(beam.py(), 0.0);
    assert_relative_eq!(beam.pz(), 1.0);
    assert_relative_eq!(beam.e(), 1.0);
    assert_relative_eq!(dataset.aux_by_name(0, "pol_angle").unwrap(), 0.42);
    assert!(dataset.p4_by_name(0, "missing").is_none());
    assert!(dataset.aux_by_name(0, "missing").is_none());
}

#[test]
fn test_parquet_roundtrip_to_tempfile() {
    let dataset = open_test_dataset("data_f32.parquet", DatasetReadOptions::new());
    let dir = make_temp_dir();
    let path = dir.join("roundtrip.parquet");
    let path_str = path.to_str().expect("path should be valid UTF-8");

    write_parquet(&dataset, path_str, &DatasetWriteOptions::default())
        .expect("writing parquet should succeed");
    let reopened = read_parquet(path_str, &DatasetReadOptions::new())
        .expect("parquet roundtrip should reopen");

    assert_datasets_close(&dataset, &reopened, TEST_P4_NAMES, TEST_AUX_NAMES);
    fs::remove_dir_all(&dir).expect("temp dir cleanup should succeed");
}

#[test]
fn test_parquet_roundtrip_incremental_small_batches() {
    let dataset = open_test_dataset("data_f32.parquet", DatasetReadOptions::new());
    let dir = make_temp_dir();
    let path = dir.join("roundtrip_small_batches.parquet");
    let path_str = path.to_str().expect("path should be valid UTF-8");

    let write_options = DatasetWriteOptions::default().batch_size(1);
    write_parquet(&dataset, path_str, &write_options)
        .expect("writing parquet in small batches should succeed");
    let reopened = read_parquet(path_str, &DatasetReadOptions::new())
        .expect("parquet roundtrip should reopen");

    assert_datasets_close(&dataset, &reopened, TEST_P4_NAMES, TEST_AUX_NAMES);
    fs::remove_dir_all(&dir).expect("temp dir cleanup should succeed");
}

#[test]
fn test_parquet_read_order_is_deterministic_across_repeated_reads() {
    let dataset = open_test_dataset("data_f32.parquet", DatasetReadOptions::new());
    let dir = make_temp_dir();
    let path = dir.join("deterministic_order.parquet");
    let path_str = path.to_str().expect("path should be valid UTF-8");

    // Force many parquet batches so order stability is verified under incremental reads.
    let write_options = DatasetWriteOptions::default().batch_size(1);
    write_parquet(&dataset, path_str, &write_options)
        .expect("writing parquet in small batches should succeed");

    let first = read_parquet(path_str, &DatasetReadOptions::new())
        .expect("first parquet read should succeed");
    let second = read_parquet(path_str, &DatasetReadOptions::new())
        .expect("second parquet read should succeed");

    assert_eq!(first.n_events(), second.n_events());
    assert_eq!(first.n_events(), dataset.n_events());
    for event_index in 0..dataset.n_events() {
        let source = dataset
            .event(event_index)
            .expect("source event should exist");
        let first_event = first
            .event(event_index)
            .expect("first read event should exist");
        let second_event = second
            .event(event_index)
            .expect("second read event should exist");
        assert_events_close(&source, &first_event, TEST_P4_NAMES, TEST_AUX_NAMES);
        assert_events_close(&source, &second_event, TEST_P4_NAMES, TEST_AUX_NAMES);
    }

    fs::remove_dir_all(&dir).expect("temp dir cleanup should succeed");
}

#[test]
fn test_parquet_storage_roundtrip_to_tempfile() {
    let source_path = test_data_path("data_f32.parquet");
    let source_path_str = source_path.to_str().expect("path should be valid UTF-8");
    let dataset_columnar =
        read_parquet_storage(source_path_str, &DatasetReadOptions::new()).expect("columnar load");
    let dir = make_temp_dir();
    let path = dir.join("roundtrip_columnar.parquet");
    let path_str = path.to_str().expect("path should be valid UTF-8");

    write_parquet_storage(&dataset_columnar, path_str, &DatasetWriteOptions::default())
        .expect("writing columnar parquet should succeed");
    let reopened = read_parquet_storage(path_str, &DatasetReadOptions::new())
        .expect("columnar roundtrip reopen");

    assert_dataset_columnar_close(&dataset_columnar, &reopened);
    fs::remove_dir_all(&dir).expect("temp dir cleanup should succeed");
}

#[test]
fn test_root_storage_matches_parquet_storage() {
    let root_path = test_data_path("data_f32.root");
    let root_path_str = root_path.to_str().expect("path should be valid UTF-8");
    let parquet_path = test_data_path("data_f32.parquet");
    let parquet_path_str = parquet_path.to_str().expect("path should be valid UTF-8");

    let from_root = read_root_storage(root_path_str, &DatasetReadOptions::new())
        .expect("root columnar load should work");
    let from_parquet = read_parquet_storage(parquet_path_str, &DatasetReadOptions::new())
        .expect("parquet columnar load should work");
    assert_dataset_columnar_close(&from_root, &from_parquet);
}

#[test]
fn test_root_storage_repeated_reads_are_stable() {
    let root_path = test_data_path("data_f32.root");
    let root_path_str = root_path.to_str().expect("path should be valid UTF-8");
    let first = read_root_storage(root_path_str, &DatasetReadOptions::new())
        .expect("first root columnar load should work");
    let second = read_root_storage(root_path_str, &DatasetReadOptions::new())
        .expect("second root columnar load should work");
    assert_dataset_columnar_close(&first, &second);
}

#[cfg(feature = "mpi")]
#[mpi_test(np = [2])]
fn test_root_storage_reads_rank_local_entry_ranges_under_mpi() {
    let root_path = test_data_path("data_f32.root");
    let root_path_str = root_path.to_str().expect("path should be valid UTF-8");
    let full = read_root_storage(root_path_str, &DatasetReadOptions::new())
        .expect("full root columnar load should work");
    let total = full.n_events();

    use_mpi(true);
    let world = get_world().expect("MPI world should be initialized");
    let partition = world.partition(total);
    let local_range = partition.range_for_rank(world.rank() as usize);

    let local = read_root_storage(root_path_str, &DatasetReadOptions::new())
        .expect("rank-local root columnar load should work");
    assert_eq!(local.n_events(), local_range.len());

    for (local_index, global_index) in local_range.clone().enumerate() {
        for p4_index in 0..full.metadata().p4_names().len() {
            let expected = full.p4(global_index, p4_index);
            let actual = local.p4(local_index, p4_index);
            assert_relative_eq!(actual.px(), expected.px(), epsilon = 1e-12);
            assert_relative_eq!(actual.py(), expected.py(), epsilon = 1e-12);
            assert_relative_eq!(actual.pz(), expected.pz(), epsilon = 1e-12);
            assert_relative_eq!(actual.e(), expected.e(), epsilon = 1e-12);
        }
        for aux_index in 0..full.metadata().aux_names().len() {
            assert_relative_eq!(
                local.aux(local_index, aux_index),
                full.aux(global_index, aux_index),
                epsilon = 1e-12
            );
        }
        assert_relative_eq!(
            local.weight(local_index),
            full.weight(global_index),
            epsilon = 1e-12
        );
    }

    let local_dataset = local.to_dataset();
    assert_eq!(local_dataset.n_events_local(), local_range.len());
    assert_eq!(local_dataset.n_events(), total);
    finalize_mpi();
}

#[test]
fn test_root_roundtrip_to_tempfile() {
    let dataset = open_test_dataset("data_f32.parquet", DatasetReadOptions::new());
    let dir = make_temp_dir();
    let path = dir.join("roundtrip.root");
    let path_str = path.to_str().expect("path should be valid UTF-8");

    write_root(&dataset, path_str, &DatasetWriteOptions::default())
        .expect("writing root should succeed");
    let reopened =
        read_root(path_str, &DatasetReadOptions::new()).expect("root roundtrip should reopen");

    assert_datasets_close(&dataset, &reopened, TEST_P4_NAMES, TEST_AUX_NAMES);
    fs::remove_dir_all(&dir).expect("temp dir cleanup should succeed");
}

#[cfg(feature = "mpi")]
#[mpi_test(np = [2])]
fn test_root_roundtrip_to_tempfile_mpi() {
    let reference = open_test_dataset("data_f32.parquet", DatasetReadOptions::new());
    use_mpi(true);
    let world = get_world().expect("MPI world should be initialized");
    let is_root = world.is_root();

    let dataset = open_test_dataset("data_f32.parquet", DatasetReadOptions::new());
    let path = env::temp_dir().join("laddu_mpi_root_roundtrip.root");
    let path_str = path.to_str().expect("path should be valid UTF-8");

    if world.is_root() && path.exists() {
        fs::remove_file(&path).expect("stale mpi root file cleanup should succeed");
    }
    world.barrier();

    write_root(&dataset, path_str, &DatasetWriteOptions::default())
        .expect("writing root with mpi should succeed");
    world.barrier();
    world.barrier();
    finalize_mpi();

    if is_root {
        let reopened =
            read_root(path_str, &DatasetReadOptions::new()).expect("root roundtrip should reopen");
        assert_datasets_close(&reference, &reopened, TEST_P4_NAMES, TEST_AUX_NAMES);
        if path.exists() {
            fs::remove_file(&path).expect("mpi root roundtrip cleanup should succeed");
        }
    }
}

#[cfg(feature = "mpi")]
#[mpi_test(np = [2])]
fn test_root_output_is_deterministic_under_mpi() {
    use_mpi(true);
    let world = get_world().expect("MPI world should be initialized");

    let dataset = open_test_dataset("data_f32.parquet", DatasetReadOptions::new());
    let first_path = env::temp_dir().join("laddu_mpi_root_determinism_first.root");
    let second_path = env::temp_dir().join("laddu_mpi_root_determinism_second.root");
    let first_path_str = first_path.to_str().expect("path should be valid UTF-8");
    let second_path_str = second_path.to_str().expect("path should be valid UTF-8");

    if world.is_root() {
        for path in [&first_path, &second_path] {
            if path.exists() {
                fs::remove_file(path).expect("stale mpi root file cleanup should succeed");
            }
        }
    }
    world.barrier();

    write_root(&dataset, first_path_str, &DatasetWriteOptions::default())
        .expect("first mpi root write should succeed");
    world.barrier();
    write_root(&dataset, second_path_str, &DatasetWriteOptions::default())
        .expect("second mpi root write should succeed");
    world.barrier();

    let first = read_root_storage(first_path_str, &DatasetReadOptions::new())
        .expect("first mpi root output should reopen");
    let second = read_root_storage(second_path_str, &DatasetReadOptions::new())
        .expect("second mpi root output should reopen");
    assert_dataset_columnar_close(&first, &second);

    world.barrier();
    if world.is_root() {
        for path in [&first_path, &second_path] {
            if path.exists() {
                fs::remove_file(path).expect("mpi root determinism cleanup should succeed");
            }
        }
    }
    finalize_mpi();
}

#[cfg(feature = "mpi")]
#[mpi_test(np = [2])]
fn test_root_output_matches_between_mpi_and_non_mpi_writes() {
    let cpu_dataset = open_test_dataset("data_f32.parquet", DatasetReadOptions::new());
    let mpi_path = env::temp_dir().join("laddu_root_mpi_reference.root");
    let mpi_path_str = mpi_path.to_str().expect("path should be valid UTF-8");

    use_mpi(true);
    let world = get_world().expect("MPI world should be initialized");
    let is_root = world.is_root();
    let mpi_dataset = open_test_dataset("data_f32.parquet", DatasetReadOptions::new());

    if is_root && mpi_path.exists() {
        fs::remove_file(&mpi_path).expect("stale root file cleanup should succeed");
    }
    world.barrier();
    write_root(&mpi_dataset, mpi_path_str, &DatasetWriteOptions::default())
        .expect("mpi root write should succeed");
    world.barrier();
    world.barrier();
    finalize_mpi();

    if is_root {
        let cpu_dir = make_temp_dir();
        let cpu_path = cpu_dir.join("laddu_root_cpu_reference.root");
        let cpu_path_str = cpu_path.to_str().expect("path should be valid UTF-8");
        write_root(&cpu_dataset, cpu_path_str, &DatasetWriteOptions::default())
            .expect("non-mpi root write should succeed");

        let cpu_output = read_root_storage(cpu_path_str, &DatasetReadOptions::new())
            .expect("non-mpi root output should reopen");
        let mpi_output = read_root_storage(mpi_path_str, &DatasetReadOptions::new())
            .expect("mpi root output should reopen");
        assert_dataset_columnar_close(&cpu_output, &mpi_output);

        fs::remove_dir_all(&cpu_dir).expect("root comparison temp dir cleanup should succeed");
        if mpi_path.exists() {
            fs::remove_file(&mpi_path).expect("root comparison cleanup should succeed");
        }
    }
}

#[test]
fn test_root_local_column_buffers_match_columnar_storage() {
    let dataset = open_test_dataset("data_f32.parquet", DatasetReadOptions::new());
    let buffers = io::build_root_local_column_buffers::<f64>(&dataset.columnar);
    let expected_names = dataset
        .p4_names()
        .iter()
        .flat_map(|name| {
            io::P4_COMPONENT_SUFFIXES
                .iter()
                .map(move |suffix| format!("{name}{suffix}"))
        })
        .chain(dataset.aux_names().iter().cloned())
        .chain(std::iter::once("weight".to_string()))
        .collect::<Vec<_>>();
    let expected_values = dataset
        .columnar
        .p4
        .iter()
        .flat_map(|p4| [p4.px.clone(), p4.py.clone(), p4.pz.clone(), p4.e.clone()])
        .chain(dataset.columnar.aux.clone())
        .chain(std::iter::once(dataset.columnar.weights.clone()))
        .collect::<Vec<_>>();
    assert_eq!(
        buffers
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
        expected_names
    );
    assert_eq!(
        buffers
            .into_iter()
            .map(|(_, values)| values)
            .collect::<Vec<_>>(),
        expected_values
    );
}

#[test]
fn test_root_local_column_buffers_convert_precision() {
    let dataset = open_test_dataset("data_f32.parquet", DatasetReadOptions::new());
    let buffers = io::build_root_local_column_buffers::<f32>(&dataset.columnar);
    let expected_values = dataset
        .columnar
        .p4
        .iter()
        .flat_map(|p4| {
            [
                p4.px.iter().map(|value| *value as f32).collect::<Vec<_>>(),
                p4.py.iter().map(|value| *value as f32).collect::<Vec<_>>(),
                p4.pz.iter().map(|value| *value as f32).collect::<Vec<_>>(),
                p4.e.iter().map(|value| *value as f32).collect::<Vec<_>>(),
            ]
        })
        .chain(
            dataset
                .columnar
                .aux
                .iter()
                .map(|aux| aux.iter().map(|value| *value as f32).collect::<Vec<_>>()),
        )
        .chain(std::iter::once(
            dataset
                .columnar
                .weights
                .iter()
                .map(|value| *value as f32)
                .collect::<Vec<_>>(),
        ))
        .collect::<Vec<_>>();

    assert_eq!(
        buffers
            .into_iter()
            .map(|(_, values)| values)
            .collect::<Vec<_>>(),
        expected_values
    );
}

#[test]
fn test_parquet_chunk_iterator_matches_full_read() {
    let path = test_data_path("data_f32.parquet");
    let path_str = path.to_str().expect("path should be valid UTF-8");
    let options = DatasetReadOptions::new();
    let full = read_parquet(path_str, &options).expect("full parquet read should work");
    let chunks = read_parquet_chunks(path_str, &options, 17).expect("chunk iterator should open");

    let mut global_idx = 0usize;
    for chunk in chunks {
        let chunk = chunk.expect("chunk read should succeed");
        for local_idx in 0..chunk.n_events_local() {
            let left = full
                .event(global_idx)
                .expect("full dataset event should exist");
            let right = chunk
                .event(local_idx)
                .expect("chunk dataset event should exist");
            assert_events_close(&left, &right, TEST_P4_NAMES, TEST_AUX_NAMES);
            global_idx += 1;
        }
    }

    assert_eq!(global_idx, full.n_events());
}

#[cfg(feature = "mpi")]
#[mpi_test(np = [2])]
fn test_parquet_chunk_iterator_respects_mpi_partition() {
    let path = test_data_path("data_f32.parquet");
    let path_str = path.to_str().expect("path should be valid UTF-8");
    let options = DatasetReadOptions::new();
    let reference = read_parquet(path_str, &options).expect("reference parquet read should work");

    use_mpi(true);
    let world = get_world().expect("MPI world should be initialized");
    let partition = world.partition(reference.n_events());
    let local_range = partition.range_for_rank(world.rank() as usize);
    let chunks = read_parquet_chunks(path_str, &options, 17).expect("chunk iterator should open");

    let mut local_idx = 0usize;
    for chunk in chunks {
        let chunk = chunk.expect("chunk read should succeed");
        assert!(chunk.n_events_local() <= 17);
        for chunk_idx in 0..chunk.n_events_local() {
            let expected = reference
                .event(local_range.start + local_idx)
                .expect("reference event should exist");
            let actual = chunk.event(chunk_idx).expect("chunk event should exist");
            assert_events_close(&expected, &actual, TEST_P4_NAMES, TEST_AUX_NAMES);
            local_idx += 1;
        }
    }

    assert_eq!(local_idx, local_range.len());
    let mut gathered_counts = vec![0usize; world.size() as usize];
    world.all_gather_into(&local_idx, &mut gathered_counts);
    assert_eq!(
        gathered_counts.into_iter().sum::<usize>(),
        reference.n_events()
    );
    finalize_mpi();
}

#[test]
fn test_parquet_chunk_iterator_with_options_chunk_size_one() {
    let path = test_data_path("data_f32.parquet");
    let path_str = path.to_str().expect("path should be valid UTF-8");
    let options = DatasetReadOptions::new().chunk_size(1);
    let full =
        read_parquet(path_str, &DatasetReadOptions::new()).expect("full parquet read should work");
    let chunks =
        read_parquet_chunks_with_options(path_str, &options).expect("chunk iterator should open");
    let mut event_count = 0usize;
    let mut chunk_count = 0usize;

    for chunk in chunks {
        let chunk = chunk.expect("chunk read should succeed");
        chunk_count += 1;
        assert_eq!(chunk.n_events_local(), 1);
        event_count += chunk.n_events_local();
    }

    assert_eq!(event_count, full.n_events());
    assert_eq!(chunk_count, full.n_events());
}

#[test]
fn test_parquet_chunk_iterator_with_options_large_chunk_size() {
    let path = test_data_path("data_f32.parquet");
    let path_str = path.to_str().expect("path should be valid UTF-8");
    let full =
        read_parquet(path_str, &DatasetReadOptions::new()).expect("full parquet read should work");
    let options = DatasetReadOptions::new().chunk_size(full.n_events() + 100);
    let chunks =
        read_parquet_chunks_with_options(path_str, &options).expect("chunk iterator should open");
    let chunk_vec = chunks
        .collect::<LadduResult<Vec<_>>>()
        .expect("all chunk reads should succeed");

    assert_eq!(chunk_vec.len(), 1);
    assert_eq!(chunk_vec[0].n_events_local(), full.n_events());
}

#[test]
fn test_dataset_chunk_builder_matches_full_parquet_read() {
    let path = test_data_path("data_f32.parquet");
    let path_str = path.to_str().expect("path should be valid UTF-8");
    let options = DatasetReadOptions::new().chunk_size(13);
    let full =
        read_parquet(path_str, &DatasetReadOptions::new()).expect("full parquet read should work");
    let chunks =
        read_parquet_chunks_with_options(path_str, &options).expect("chunk iterator should open");

    let mut builder = DatasetChunkBuilder::new();
    for chunk in chunks {
        let chunk = chunk.expect("chunk read should succeed");
        builder.push_chunk(&chunk).expect("chunk should append");
    }
    let rebuilt = builder.finish();

    assert_datasets_close(&full, &rebuilt, TEST_P4_NAMES, TEST_AUX_NAMES);
}

#[test]
fn test_try_fold_dataset_chunks_matches_full_weight_sum() {
    let path = test_data_path("data_f32.parquet");
    let path_str = path.to_str().expect("path should be valid UTF-8");
    let full =
        read_parquet(path_str, &DatasetReadOptions::new()).expect("full parquet read should work");
    let chunks = read_parquet_chunks(path_str, &DatasetReadOptions::new(), 11)
        .expect("chunk iterator should open");

    let folded = try_fold_dataset_chunks(chunks, 0.0_f64, |acc, chunk| {
        Ok(acc + chunk.n_events_weighted_local())
    })
    .expect("chunk fold should succeed");

    assert_relative_eq!(folded, full.n_events_weighted_local(), epsilon = 1e-9);
}
