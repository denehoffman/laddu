//! Fictitious coupled-channel data used by the K-matrix benchmarks and tests.

use laddu::{
    amplitudes::{blatt_weisskopf_barriers, f_vector, k_matrix_with_background, p_vector},
    complex,
    expr::{Expr, matrix_from_flat},
    parameter,
    physics::{
        math::{QR_DEFAULT, chew_mandelstam},
        quantum::L,
    },
    vector,
};

struct CoupledChannelData<const CHANNELS: usize, const POLES: usize> {
    channel_mass_1: [f64; CHANNELS],
    channel_mass_2: [f64; CHANNELS],
    pole_masses: [f64; POLES],
    couplings: [[f64; POLES]; CHANNELS],
    background: [[f64; CHANNELS]; CHANNELS],
    l: L,
    adler_zero: Option<(f64, f64)>,
    output_channel: usize,
}

/// Builds one coupled-channel amplitude from benchmark-local constants.
fn coupled_channel_amplitude<const CHANNELS: usize, const POLES: usize>(
    s: &Expr,
    data: &CoupledChannelData<CHANNELS, POLES>,
    production: [Expr; POLES],
) -> Expr {
    let masses_1 = vector(data.channel_mass_1);
    let masses_2 = vector(data.channel_mass_2);
    let poles = vector(data.pole_masses);
    let couplings = matrix_from_flat(
        CHANNELS,
        POLES,
        data.couplings.iter().flat_map(|row| row.iter().copied()),
    )
    .unwrap();
    let barriers =
        blatt_weisskopf_barriers(s, masses_1, masses_2, &poles, data.l, QR_DEFAULT).unwrap();
    let background = matrix_from_flat(
        CHANNELS,
        CHANNELS,
        (0..CHANNELS).flat_map(|row| {
            let barriers = barriers.clone();
            (0..CHANNELS).map(move |col| {
                (0..POLES).fold(Expr::from(0.0), |sum, pole| {
                    sum + barriers.matrix_element(row, pole)
                        * data.background[row][col]
                        * barriers.matrix_element(col, pole)
                })
            })
        }),
    )
    .unwrap();
    let mut k = k_matrix_with_background(s, &poles, &couplings, &barriers, background).unwrap();
    if let Some((s_0, normalization)) = data.adler_zero {
        let adler_factor = (s - s_0) / normalization;
        k = matrix_from_flat(
            CHANNELS,
            CHANNELS,
            (0..CHANNELS).flat_map(|row| {
                let adler_factor = adler_factor.clone();
                let k = k.clone();
                (0..CHANNELS).map(move |col| &adler_factor * k.matrix_element(row, col))
            }),
        )
        .unwrap();
    }
    let p = p_vector(s, &poles, vector(production), &couplings, &barriers).unwrap();
    let phase_space = matrix_from_flat(
        CHANNELS,
        CHANNELS,
        (0..CHANNELS).flat_map(|row| {
            (0..CHANNELS).map(move |col| {
                if row == col {
                    chew_mandelstam(s, data.channel_mass_1[row], data.channel_mass_2[row])
                } else {
                    Expr::from(0.0)
                }
            })
        }),
    )
    .unwrap();
    f_vector(s, poles, k, p, phase_space)
        .unwrap()
        .component(data.output_channel)
}

/// Returns the positive scalar, negative scalar, and positive tensor
/// amplitudes with neutral, fictitious benchmark data.
pub fn fictitious_kmatrix_components(s: &Expr) -> (Expr, Expr, Expr) {
    let scalar_data = CoupledChannelData {
        channel_mass_1: [0.11, 0.22, 0.33, 0.44, 0.55],
        channel_mass_2: [0.13, 0.24, 0.35, 0.46, 0.57],
        pole_masses: [0.72, 0.96, 1.20, 1.44, 1.68],
        couplings: [
            [0.72, 0.08, -0.18, 0.05, -0.11],
            [-0.06, 0.14, 0.02, 0.21, 0.09],
            [0.19, 0.33, 0.41, 0.12, 0.17],
            [-0.12, 0.27, 0.09, 0.16, 0.24],
            [0.23, 0.11, 0.28, -0.07, -0.15],
        ],
        background: [
            [0.04, 0.00, -0.02, -0.01, 0.01],
            [0.00, 0.02, 0.01, 0.00, -0.01],
            [-0.02, 0.01, 0.03, 0.02, -0.02],
            [-0.01, 0.00, 0.02, -0.04, -0.03],
            [0.01, -0.01, -0.02, -0.03, -0.05],
        ],
        l: L::int(0),
        adler_zero: Some((0.04, 1.2)),
        output_channel: 2,
    };
    let auxiliary_scalar_data = CoupledChannelData {
        channel_mass_1: [0.11, 0.33],
        channel_mass_2: [0.55, 0.37],
        pole_masses: [0.88, 1.36],
        couplings: [[0.38, 0.12], [0.09, 0.31]],
        background: [[0.03, 0.01], [0.01, -0.02]],
        l: L::int(0),
        adler_zero: None,
        output_channel: 1,
    };
    let tensor_data = CoupledChannelData {
        channel_mass_1: [0.11, 0.22, 0.33, 0.44],
        channel_mass_2: [0.13, 0.24, 0.35, 0.46],
        pole_masses: [0.81, 1.13, 1.45, 1.77],
        couplings: [
            [0.31, 0.08, -0.12, 0.04],
            [0.05, 0.26, 0.11, 0.18],
            [0.17, 0.22, 0.34, 0.09],
            [-0.09, 0.15, 0.07, 0.21],
        ],
        background: [
            [0.02, 0.00, 0.01, -0.01],
            [0.00, 0.03, 0.01, 0.00],
            [0.01, 0.01, -0.02, 0.02],
            [-0.01, 0.00, 0.02, 0.01],
        ],
        l: L::int(2),
        adler_zero: None,
        output_channel: 2,
    };
    let auxiliary_tensor_data = CoupledChannelData {
        channel_mass_1: [0.11, 0.33, 0.22],
        channel_mass_2: [0.55, 0.37, 0.67],
        pole_masses: [0.94, 1.52],
        couplings: [[0.42, 0.13], [0.07, 0.34], [0.19, -0.08]],
        background: [[0.03, 0.01, 0.00], [0.01, -0.02, 0.01], [0.00, 0.01, 0.02]],
        l: L::int(2),
        adler_zero: None,
        output_channel: 1,
    };

    let scalar_positive = coupled_channel_amplitude(
        s,
        &scalar_data,
        [
            complex(
                parameter!("scalar+ pole0 re", 0.0),
                parameter!("scalar+ pole0 im", 0.0),
            ),
            complex(
                parameter!("scalar+ pole1 re"),
                parameter!("scalar+ pole1 im_fixed", 0.0),
            ),
            complex(
                parameter!("scalar+ pole2 re"),
                parameter!("scalar+ pole2 im"),
            ),
            complex(
                parameter!("scalar+ pole3 re"),
                parameter!("scalar+ pole3 im"),
            ),
            complex(
                parameter!("scalar+ pole4 re"),
                parameter!("scalar+ pole4 im"),
            ),
        ],
    );
    let scalar_negative = coupled_channel_amplitude(
        s,
        &scalar_data,
        [
            complex(
                parameter!("scalar- pole0 re", 0.0),
                parameter!("scalar- pole0 im", 0.0),
            ),
            complex(
                parameter!("scalar- pole1 re"),
                parameter!("scalar- pole1 im_fixed", 0.0),
            ),
            complex(
                parameter!("scalar- pole2 re"),
                parameter!("scalar- pole2 im"),
            ),
            complex(
                parameter!("scalar- pole3 re"),
                parameter!("scalar- pole3 im"),
            ),
            complex(
                parameter!("scalar- pole4 re"),
                parameter!("scalar- pole4 im"),
            ),
        ],
    );
    let auxiliary_scalar_positive = coupled_channel_amplitude(
        s,
        &auxiliary_scalar_data,
        [
            complex(
                parameter!("auxiliary scalar+ pole0 re"),
                parameter!("auxiliary scalar+ pole0 im"),
            ),
            complex(
                parameter!("auxiliary scalar+ pole1 re"),
                parameter!("auxiliary scalar+ pole1 im"),
            ),
        ],
    );
    let auxiliary_scalar_negative = coupled_channel_amplitude(
        s,
        &auxiliary_scalar_data,
        [
            complex(
                parameter!("auxiliary scalar- pole0 re"),
                parameter!("auxiliary scalar- pole0 im"),
            ),
            complex(
                parameter!("auxiliary scalar- pole1 re"),
                parameter!("auxiliary scalar- pole1 im"),
            ),
        ],
    );
    let tensor_positive = coupled_channel_amplitude(
        s,
        &tensor_data,
        [
            complex(parameter!("tensor pole0 re"), parameter!("tensor pole0 im")),
            complex(parameter!("tensor pole1 re"), parameter!("tensor pole1 im")),
            complex(parameter!("tensor pole2 re"), parameter!("tensor pole2 im")),
            complex(parameter!("tensor pole3 re"), parameter!("tensor pole3 im")),
        ],
    );
    let auxiliary_tensor = coupled_channel_amplitude(
        s,
        &auxiliary_tensor_data,
        [
            complex(
                parameter!("auxiliary tensor pole0 re"),
                parameter!("auxiliary tensor pole0 im"),
            ),
            complex(
                parameter!("auxiliary tensor pole1 re"),
                parameter!("auxiliary tensor pole1 im"),
            ),
        ],
    );

    (
        scalar_positive + auxiliary_scalar_positive,
        scalar_negative + auxiliary_scalar_negative,
        tensor_positive + auxiliary_tensor,
    )
}
