use laddu::{
    amplitudes::{parameter, zlm::Zlm, Manager},
    data::open,
    utils::{
        enums::{Frame, Sign},
        variables::{Angles, Polarization},
    },
    ComplexScalar, Float, Scalar,
};

fn main() {
    let ds_data = open("data.parquet").unwrap();

    let angles = Angles::new(0, [1], [2], [2, 3], Frame::Helicity);
    let polarization = Polarization::new(0, [1], 0);
    let mut manager = Manager::default();
    let z00p = manager
        .register(Zlm::new(
            "Z00+",
            0,
            0,
            Sign::Positive,
            &angles,
            &polarization,
        ))
        .unwrap();
    let z00n = manager
        .register(Zlm::new(
            "Z00-",
            0,
            0,
            Sign::Negative,
            &angles,
            &polarization,
        ))
        .unwrap();
    let z22p = manager
        .register(Zlm::new(
            "Z22+",
            2,
            2,
            Sign::Positive,
            &angles,
            &polarization,
        ))
        .unwrap();
    let s0p = manager
        .register(Scalar::new("c00+", parameter("c00+ re")))
        .unwrap();
    let s0n = manager
        .register(Scalar::new("c00-", parameter("c00- re")))
        .unwrap();
    let d2p = manager
        .register(ComplexScalar::new(
            "c22+",
            parameter("c22+ re"),
            parameter("c22+ im"),
        ))
        .unwrap();
    let pos_re = (&s0p * z00p.real() + &d2p * z22p.real()).norm_sqr();
    let pos_im = (&s0p * z00p.imag() + &d2p * z22p.imag()).norm_sqr();
    let neg_re = (&s0n * z00n.real()).norm_sqr();
    let neg_im = (&s0n * z00n.imag()).norm_sqr();
    let expr = pos_re + pos_im + neg_re + neg_im;
    let model = manager.model(&expr);
    let evaluator = model.load(&ds_data);
    let p: Vec<Float> = vec![100.0; evaluator.parameters().len()];
    std::hint::black_box(evaluator.evaluate(&p));
}
