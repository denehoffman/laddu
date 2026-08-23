use std::sync::mpsc;

use crate::{WgpuContext, WgpuError, WgpuResult};

pub(crate) const STATUS_WORD_BYTES: usize = std::mem::size_of::<u32>();
pub(crate) const STATUS_SENTINEL: u32 = u32::MAX;

struct UnmapGuard<'a>(&'a wgpu::Buffer);

impl Drop for UnmapGuard<'_> {
    fn drop(&mut self) {
        self.0.unmap();
    }
}

/// The event-indexed failures written by the GPU reduction and solve kernels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GpuStatus {
    pub(crate) invalid_event: Option<usize>,
    pub(crate) singular_event: Option<usize>,
}

impl GpuStatus {
    /// Converts status slots to the public error type in contract order.
    ///
    /// The positive-reduction failure takes precedence when both slots are
    /// populated. This is the ordering used by all GPU execution paths.
    pub(crate) fn error(self) -> Option<WgpuError> {
        self.invalid_event
            .map(WgpuError::NonPositiveEvent)
            .or_else(|| self.singular_event.map(WgpuError::SingularMatrixEvent))
    }
}

/// Decodes one or two event-status words from a readback payload.
pub(crate) fn decode_status(bytes: &[u8], words: usize) -> WgpuResult<GpuStatus> {
    let required = words
        .checked_mul(STATUS_WORD_BYTES)
        .ok_or_else(|| WgpuError::BufferMap("status payload size overflow".to_string()))?;
    if !matches!(words, 1 | 2) || bytes.len() < required {
        return Err(WgpuError::BufferMap(format!(
            "status readback is truncated: expected at least {required} bytes, got {}",
            bytes.len()
        )));
    }
    let read_word = |offset: usize| {
        u32::from_ne_bytes(
            bytes[offset..offset + STATUS_WORD_BYTES]
                .try_into()
                .expect("status word width was checked"),
        )
    };
    let invalid = read_word(0);
    let invalid_event = (invalid != STATUS_SENTINEL).then_some(invalid as usize);
    let singular_event = if words == 2 {
        let event = read_word(STATUS_WORD_BYTES);
        (event != STATUS_SENTINEL).then_some(event as usize)
    } else {
        None
    };
    Ok(GpuStatus {
        invalid_event,
        singular_event,
    })
}

/// Decodes a solve-only status buffer, whose sole word is a singular-event
/// index rather than the invalid-event slot used by reductions.
pub(crate) fn decode_singular_status(bytes: &[u8]) -> WgpuResult<GpuStatus> {
    let status = decode_status(bytes, 1)?;
    Ok(GpuStatus {
        invalid_event: None,
        singular_event: status.invalid_event,
    })
}

/// Submit an encoder and synchronously decode its mapped readback range.
///
/// Once mapping succeeds, the staging buffer is unmapped on every path,
/// including errors returned by the decoder. Keeping the mapping lifecycle in
/// one place prevents a failed status/payload decode from leaving a buffer
/// mapped for a later execution.
pub(crate) fn submit_and_readback<T>(
    context: &WgpuContext,
    encoder: wgpu::CommandEncoder,
    staging: &wgpu::Buffer,
    readback_bytes: u64,
    decode: impl FnOnce(&[u8]) -> WgpuResult<T>,
) -> WgpuResult<T> {
    context.queue().submit([encoder.finish()]);
    let slice = staging.slice(..readback_bytes);
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

    let cleanup = UnmapGuard(staging);
    let mapped = slice
        .get_mapped_range()
        .map_err(|error| WgpuError::BufferMap(error.to_string()))?;
    let result = decode(&mapped);
    drop(mapped);
    drop(cleanup);
    result
}

#[cfg(test)]
mod tests {
    use super::{GpuStatus, STATUS_SENTINEL, STATUS_WORD_BYTES, decode_status};
    use crate::WgpuError;

    fn words(invalid: u32, singular: Option<u32>) -> Vec<u8> {
        let mut bytes = invalid.to_ne_bytes().to_vec();
        if let Some(singular) = singular {
            bytes.extend(singular.to_ne_bytes());
        }
        bytes
    }

    #[test]
    fn status_sentinel_means_no_error() {
        let status = decode_status(&words(STATUS_SENTINEL, Some(STATUS_SENTINEL)), 2).unwrap();
        assert_eq!(
            status,
            GpuStatus {
                invalid_event: None,
                singular_event: None
            }
        );
        assert!(status.error().is_none());
    }

    #[test]
    fn status_rejects_truncated_payloads() {
        let error = decode_status(&[0; STATUS_WORD_BYTES], 2).unwrap_err();
        assert!(matches!(error, WgpuError::BufferMap(message) if message.contains("truncated")));
    }

    #[test]
    fn invalid_status_precedes_singular_status() {
        let status = decode_status(&words(7, Some(11)), 2).unwrap();
        assert_eq!(status.invalid_event, Some(7));
        assert!(matches!(
            status.error(),
            Some(WgpuError::NonPositiveEvent(7))
        ));
    }

    #[test]
    fn one_status_word_decodes_without_singular_slot() {
        let status = decode_status(&words(3, None), 1).unwrap();
        assert_eq!(status.invalid_event, Some(3));
        assert!(matches!(
            status.error(),
            Some(WgpuError::NonPositiveEvent(3))
        ));
    }

    #[test]
    fn solve_status_word_is_decoded_as_singular() {
        let status = super::decode_singular_status(&words(3, None)).unwrap();
        assert!(matches!(
            status.error(),
            Some(WgpuError::SingularMatrixEvent(3))
        ));
    }
}
