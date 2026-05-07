use nalgebra::{Matrix2, Vector2};
use num::complex::Complex64;

use super::*;
use crate::Parameter;

#[test]
fn test_parameters() {
    let parameters = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let params = Parameters::new(parameters, 3, vec![0, 1, 2, 3, 4, 5]);

    assert_eq!(params.get(ParameterID::Parameter(0)), 1.0);
    assert_eq!(params.get(ParameterID::Parameter(1)), 2.0);
    assert_eq!(params.get(ParameterID::Parameter(2)), 3.0);
    assert_eq!(params.get(ParameterID::Constant(3)), 4.0);
    assert_eq!(params.get(ParameterID::Constant(4)), 5.0);
    assert_eq!(params.get(ParameterID::Constant(5)), 6.0);
    assert_eq!(params.free_index(ParameterID::Parameter(0)), Some(0));
    assert_eq!(params.free_index(ParameterID::Constant(3)), None);
    assert_eq!(params.len(), 3);
}

#[test]
fn test_uninit_parameter_returns_nan() {
    let parameters = vec![1.0, 1.0];
    let params = Parameters::new(parameters, 1, vec![0, 1]);
    assert!(params.get(ParameterID::Uninit).is_nan());
    assert!(params.get(ParameterID::Parameter(3)).is_nan());
    assert!(params.get(ParameterID::Constant(3)).is_nan());
}

#[test]
fn test_resources_amplitude_management() {
    let mut resources = Resources::default();

    let amp1 = resources.register_amplitude("amp1").unwrap();
    let amp2 = resources.register_amplitude("amp2").unwrap();

    assert!(resources.active[amp1.1]);
    assert!(resources.active[amp2.1]);

    resources.deactivate_strict("amp1").unwrap();
    assert!(!resources.active[amp1.1]);
    assert!(resources.active[amp2.1]);

    resources.activate_strict("amp1").unwrap();
    assert!(resources.active[amp1.1]);

    resources.deactivate_all();
    assert!(!resources.active[amp1.1]);
    assert!(!resources.active[amp2.1]);

    resources.activate_all();
    assert!(resources.active[amp1.1]);
    assert!(resources.active[amp2.1]);

    resources.isolate_strict("amp1").unwrap();
    assert!(resources.active[amp1.1]);
    assert!(!resources.active[amp2.1]);
}

#[test]
fn test_resources_amplitude_glob_management() {
    let mut resources = Resources::default();

    let signal_s = resources.register_amplitude("signal.s").unwrap();
    let signal_d = resources.register_amplitude("signal.d").unwrap();
    let background = resources.register_amplitude("background").unwrap();

    resources.deactivate_strict("signal.*").unwrap();
    assert!(!resources.active[signal_s.1]);
    assert!(!resources.active[signal_d.1]);
    assert!(resources.active[background.1]);

    resources.activate_strict("signal.?").unwrap();
    assert!(resources.active[signal_s.1]);
    assert!(resources.active[signal_d.1]);
    assert!(resources.active[background.1]);

    resources.isolate_strict("signal.*").unwrap();
    assert!(resources.active[signal_s.1]);
    assert!(resources.active[signal_d.1]);
    assert!(!resources.active[background.1]);

    resources.activate_all();
    resources
        .isolate_many_strict(&["signal.s", "back*"])
        .unwrap();
    assert!(resources.active[signal_s.1]);
    assert!(!resources.active[signal_d.1]);
    assert!(resources.active[background.1]);

    assert!(resources.activate_strict("missing*").is_err());
    assert!(resources.deactivate_strict("missing?").is_err());
    assert!(resources.isolate_strict("missing*").is_err());
}

#[test]
fn test_resources_non_strict_zero_match_glob_is_noop() {
    let mut resources = Resources::default();

    let signal = resources.register_amplitude("signal").unwrap();
    let background = resources.register_amplitude("background").unwrap();

    resources.deactivate("missing*");
    assert!(resources.active[signal.1]);
    assert!(resources.active[background.1]);

    resources.isolate("missing*");
    assert!(resources.active[signal.1]);
    assert!(resources.active[background.1]);

    resources.isolate_many(&["missing*"]);
    assert!(!resources.active[signal.1]);
    assert!(!resources.active[background.1]);
}

#[test]
fn test_untagged_amplitudes_remain_active_without_selectors() {
    let mut resources = Resources::default();

    let untagged = resources.register_amplitude([""]).unwrap();
    let tagged = resources.register_amplitude("tagged").unwrap();

    assert!(resources.active[untagged.1]);
    assert!(resources.active[tagged.1]);

    resources.deactivate_all();
    assert!(resources.active[untagged.1]);
    assert!(!resources.active[tagged.1]);

    resources.activate_all();
    resources.isolate_strict("tagged").unwrap();
    assert!(resources.active[untagged.1]);
    assert!(resources.active[tagged.1]);

    assert!(resources.deactivate_strict("").is_err());
    assert!(resources.apply_active_mask(&[false, true]).is_err());
}

#[test]
fn test_resources_parameter_registration() {
    let mut resources = Resources::default();

    let param1 = resources
        .register_parameter(&Parameter::new("param1"))
        .unwrap();
    let const1 = resources
        .register_parameter(&Parameter::new_fixed("const1", 1.0))
        .unwrap();

    match param1 {
        ParameterID::Parameter(idx) => assert_eq!(idx, 0),
        _ => panic!("Expected Parameter variant"),
    }

    match const1 {
        ParameterID::Constant(idx) => assert_eq!(idx, 1),
        _ => panic!("Expected Constant variant"),
    }

    let parameter_map = resources.parameters();
    assert_eq!(parameter_map.names(), vec!["param1", "const1"]);
    assert_eq!(parameter_map.free().names(), vec!["param1"]);
    assert_eq!(parameter_map.fixed().names(), vec!["const1"]);
    let display = parameter_map.to_string();
    assert!(display.contains("free:"));
    assert!(display.contains("param1"));
    assert!(display.contains("const1 = 1"));
}

#[test]
fn test_cache_scalar_operations() {
    let mut resources = Resources::default();

    let scalar1 = resources.register_scalar(Some("test_scalar"));
    let scalar2 = resources.register_scalar(None);
    let scalar3 = resources.register_scalar(Some("test_scalar"));

    resources.reserve_cache(1);
    let cache = &mut resources.caches[0];

    cache.store_scalar(scalar1, 1.0);
    cache.store_scalar(scalar2, 2.0);

    assert_eq!(cache.get_scalar(scalar1), 1.0);
    assert_eq!(cache.get_scalar(scalar2), 2.0);
    assert_eq!(cache.get_scalar(scalar3), 1.0);
}

#[test]
fn test_cache_complex_operations() {
    let mut resources = Resources::default();

    let complex1 = resources.register_complex_scalar(Some("test_complex"));
    let complex2 = resources.register_complex_scalar(None);
    let complex3 = resources.register_complex_scalar(Some("test_complex"));

    resources.reserve_cache(1);
    let cache = &mut resources.caches[0];

    let value1 = Complex64::new(1.0, 2.0);
    let value2 = Complex64::new(3.0, 4.0);
    cache.store_complex_scalar(complex1, value1);
    cache.store_complex_scalar(complex2, value2);

    assert_eq!(cache.get_complex_scalar(complex1), value1);
    assert_eq!(cache.get_complex_scalar(complex2), value2);
    assert_eq!(cache.get_complex_scalar(complex3), value1);
}

#[test]
fn test_cache_vector_operations() {
    let mut resources = Resources::default();

    let vector_id1: VectorID<2> = resources.register_vector(Some("test_vector"));
    let vector_id2: VectorID<2> = resources.register_vector(None);
    let vector_id3: VectorID<2> = resources.register_vector(Some("test_vector"));

    resources.reserve_cache(1);
    let cache = &mut resources.caches[0];

    let value1 = Vector2::new(1.0, 2.0);
    let value2 = Vector2::new(3.0, 4.0);
    cache.store_vector(vector_id1, value1);
    cache.store_vector(vector_id2, value2);

    assert_eq!(cache.get_vector(vector_id1), value1);
    assert_eq!(cache.get_vector(vector_id2), value2);
    assert_eq!(cache.get_vector(vector_id3), value1);
}

#[test]
fn test_cache_complex_vector_operations() {
    let mut resources = Resources::default();

    let complex_vector_id1: ComplexVectorID<2> =
        resources.register_complex_vector(Some("test_complex_vector"));
    let complex_vector_id2: ComplexVectorID<2> = resources.register_complex_vector(None);
    let complex_vector_id3: ComplexVectorID<2> =
        resources.register_complex_vector(Some("test_complex_vector"));

    resources.reserve_cache(1);
    let cache = &mut resources.caches[0];

    let value1 = Vector2::new(Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0));
    let value2 = Vector2::new(Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0));
    cache.store_complex_vector(complex_vector_id1, value1);
    cache.store_complex_vector(complex_vector_id2, value2);

    assert_eq!(cache.get_complex_vector(complex_vector_id1), value1);
    assert_eq!(cache.get_complex_vector(complex_vector_id2), value2);
    assert_eq!(cache.get_complex_vector(complex_vector_id3), value1);
}

#[test]
fn test_cache_matrix_operations() {
    let mut resources = Resources::default();

    let matrix_id1: MatrixID<2, 2> = resources.register_matrix(Some("test_matrix"));
    let matrix_id2: MatrixID<2, 2> = resources.register_matrix(None);
    let matrix_id3: MatrixID<2, 2> = resources.register_matrix(Some("test_matrix"));

    resources.reserve_cache(1);
    let cache = &mut resources.caches[0];

    let value1 = Matrix2::new(1.0, 2.0, 3.0, 4.0);
    let value2 = Matrix2::new(5.0, 6.0, 7.0, 8.0);
    cache.store_matrix(matrix_id1, value1);
    cache.store_matrix(matrix_id2, value2);

    assert_eq!(cache.get_matrix(matrix_id1), value1);
    assert_eq!(cache.get_matrix(matrix_id2), value2);
    assert_eq!(cache.get_matrix(matrix_id3), value1);
}

#[test]
fn test_cache_complex_matrix_operations() {
    let mut resources = Resources::default();

    let complex_matrix_id1: ComplexMatrixID<2, 2> =
        resources.register_complex_matrix(Some("test_complex_matrix"));
    let complex_matrix_id2: ComplexMatrixID<2, 2> = resources.register_complex_matrix(None);
    let complex_matrix_id3: ComplexMatrixID<2, 2> =
        resources.register_complex_matrix(Some("test_complex_matrix"));

    resources.reserve_cache(1);
    let cache = &mut resources.caches[0];

    let value1 = Matrix2::new(
        Complex64::new(1.0, 2.0),
        Complex64::new(3.0, 4.0),
        Complex64::new(5.0, 6.0),
        Complex64::new(7.0, 8.0),
    );
    let value2 = Matrix2::new(
        Complex64::new(9.0, 10.0),
        Complex64::new(11.0, 12.0),
        Complex64::new(13.0, 14.0),
        Complex64::new(15.0, 16.0),
    );
    cache.store_complex_matrix(complex_matrix_id1, value1);
    cache.store_complex_matrix(complex_matrix_id2, value2);

    assert_eq!(cache.get_complex_matrix(complex_matrix_id1), value1);
    assert_eq!(cache.get_complex_matrix(complex_matrix_id2), value2);
    assert_eq!(cache.get_complex_matrix(complex_matrix_id3), value1);
}

#[test]
fn test_uninit_parameter_registration() {
    let mut resources = Resources::default();
    let result = resources.register_parameter(&Parameter::default());
    assert!(result.is_err());
}

#[test]
fn test_duplicate_tag_registration_controls_all_use_sites() {
    let mut resources = Resources::default();
    let amp1 = resources.register_amplitude("test_amp").unwrap();
    let amp2 = resources.register_amplitude("test_amp").unwrap();
    assert_ne!(amp1.1, amp2.1);

    resources.deactivate_strict("test_amp").unwrap();
    assert!(!resources.active[amp1.1]);
    assert!(!resources.active[amp2.1]);
}
