use std::fmt::Display;

#[cfg(feature = "python")]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

macro_rules! kmatrix_channel {
    (
        $(#[$enum_meta:meta])*
        $name:ident {
            $(
                $(#[$variant_meta:meta])*
                $variant:ident => ($index:expr, $display:expr)
            ),+ $(,)?
        }
    ) => {
        $(#[$enum_meta])*
        #[cfg_attr(feature = "python", pyclass(module = "laddu", eq, from_py_object))]
        #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub enum $name {
            $(
                $(#[$variant_meta])*
                $variant,
            )+
        }

        impl $name {
            /// Return the internal zero-based fixed K-matrix channel index.
            pub const fn index(self) -> usize {
                match self {
                    $(Self::$variant => $index,)+
                }
            }
        }

        impl Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    $(Self::$variant => write!(f, $display),)+
                }
            }
        }
    };
}

kmatrix_channel! {
    /// Channels available for the fixed Kopf $`f_0`$ K-matrix.
    KopfKMatrixF0Channel {
        /// The $`\pi\pi`$ channel.
        PiPi => (0, "pi_pi"),
        /// The $`2\pi2\pi`$ channel.
        FourPi => (1, "four_pi"),
        /// The $`K\bar{K}`$ channel.
        KKbar => (2, "k_kbar"),
        /// The $`\eta\eta`$ channel.
        EtaEta => (3, "eta_eta"),
        /// The $`\eta\eta'`$ channel.
        EtaEtaPrime => (4, "eta_eta_prime"),
    }
}

kmatrix_channel! {
    /// Channels available for the fixed Kopf $`f_2`$ K-matrix.
    KopfKMatrixF2Channel {
        /// The $`\pi\pi`$ channel.
        PiPi => (0, "pi_pi"),
        /// The $`2\pi2\pi`$ channel.
        FourPi => (1, "four_pi"),
        /// The $`K\bar{K}`$ channel.
        KKbar => (2, "k_kbar"),
        /// The $`\eta\eta`$ channel.
        EtaEta => (3, "eta_eta"),
    }
}

kmatrix_channel! {
    /// Channels available for the fixed Kopf $`a_0`$ K-matrix.
    KopfKMatrixA0Channel {
        /// The $`\pi\eta`$ channel.
        PiEta => (0, "pi_eta"),
        /// The $`K\bar{K}`$ channel.
        KKbar => (1, "k_kbar"),
    }
}

kmatrix_channel! {
    /// Channels available for the fixed Kopf $`a_2`$ K-matrix.
    KopfKMatrixA2Channel {
        /// The $`\pi\eta`$ channel.
        PiEta => (0, "pi_eta"),
        /// The $`K\bar{K}`$ channel.
        KKbar => (1, "k_kbar"),
        /// The $`\pi\eta'`$ channel.
        PiEtaPrime => (2, "pi_eta_prime"),
    }
}

kmatrix_channel! {
    /// Channels available for the fixed Kopf $`\rho`$ K-matrix.
    KopfKMatrixRhoChannel {
        /// The $`\pi\pi`$ channel.
        PiPi => (0, "pi_pi"),
        /// The $`2\pi2\pi`$ channel.
        FourPi => (1, "four_pi"),
        /// The $`K\bar{K}`$ channel.
        KKbar => (2, "k_kbar"),
    }
}

kmatrix_channel! {
    /// Channels available for the fixed Kopf $`\pi_1`$ K-matrix.
    KopfKMatrixPi1Channel {
        /// The $`\pi\eta`$ channel.
        PiEta => (0, "pi_eta"),
        /// The $`\pi\eta'`$ channel.
        PiEtaPrime => (1, "pi_eta_prime"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_indices_are_pinned() {
        assert_eq!(KopfKMatrixF0Channel::PiPi.index(), 0);
        assert_eq!(KopfKMatrixF0Channel::FourPi.index(), 1);
        assert_eq!(KopfKMatrixF0Channel::KKbar.index(), 2);
        assert_eq!(KopfKMatrixF0Channel::EtaEta.index(), 3);
        assert_eq!(KopfKMatrixF0Channel::EtaEtaPrime.index(), 4);

        assert_eq!(KopfKMatrixF2Channel::PiPi.index(), 0);
        assert_eq!(KopfKMatrixF2Channel::FourPi.index(), 1);
        assert_eq!(KopfKMatrixF2Channel::KKbar.index(), 2);
        assert_eq!(KopfKMatrixF2Channel::EtaEta.index(), 3);

        assert_eq!(KopfKMatrixA0Channel::PiEta.index(), 0);
        assert_eq!(KopfKMatrixA0Channel::KKbar.index(), 1);

        assert_eq!(KopfKMatrixA2Channel::PiEta.index(), 0);
        assert_eq!(KopfKMatrixA2Channel::KKbar.index(), 1);
        assert_eq!(KopfKMatrixA2Channel::PiEtaPrime.index(), 2);

        assert_eq!(KopfKMatrixRhoChannel::PiPi.index(), 0);
        assert_eq!(KopfKMatrixRhoChannel::FourPi.index(), 1);
        assert_eq!(KopfKMatrixRhoChannel::KKbar.index(), 2);

        assert_eq!(KopfKMatrixPi1Channel::PiEta.index(), 0);
        assert_eq!(KopfKMatrixPi1Channel::PiEtaPrime.index(), 1);
    }

    #[test]
    fn channel_displays_are_stable() {
        assert_eq!(KopfKMatrixF0Channel::KKbar.to_string(), "k_kbar");
        assert_eq!(
            KopfKMatrixF0Channel::EtaEtaPrime.to_string(),
            "eta_eta_prime"
        );
        assert_eq!(KopfKMatrixA2Channel::PiEtaPrime.to_string(), "pi_eta_prime");
    }
}
