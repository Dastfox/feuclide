//! How long the GPU takes over each pass of a frame, from timestamps written between them.

use std::sync::{Arc, OnceLock};

use bevy_ecs::prelude::Resource;
use fk_math::Real;

use crate::Gpu;

/// How long the GPU took over the passes of a frame, seconds, as a resource the renderer
/// writes when it times its frames (only while a [`perf`](crate::perf) recording runs, on a
/// device with timestamp queries inside encoders).
///
/// Read back without waiting, so it is of a frame one to three frames old; [`serial`] grows by
/// one with each frame read. The passes are, in order: `elsewhere` (the other eye's image, when
/// there is one), `shadows` (the shadow map and the renderers' offscreen passes, the ray
/// marcher's half-resolution image among them), `opaque` (the sky and every renderer), `layer`,
/// `ink`, `mosh`, `post` and `overlay`; a pass that did not run takes next to nothing. The
/// time between the other eye's submission and the frame's is not counted.
///
/// [`serial`]: Self::serial
#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct GpuTimings {
    /// How many frames were read before this one, and this one.
    pub serial: u64,
    /// Each pass's name and duration, seconds, in the order they ran.
    pub passes: Vec<(&'static str, Real)>,
}

impl GpuTimings {
    /// The duration of every pass together, seconds.
    pub fn total(&self) -> Real {
        self.passes.iter().map(|(_, seconds)| seconds).sum()
    }
}

/// The most timestamps a frame writes.
const MARKS: u32 = 32;
/// Frames that may be in flight, each read back from a buffer of its own.
const SLOTS: usize = 4;

/// A frame's timestamps on their way back.
struct Slot {
    buffer: wgpu::Buffer,
    /// The pass ending at each mark, `None` where the GPU starts on something new.
    names: Vec<Option<&'static str>>,
    /// Set by the GPU once the buffer is mapped, or failed to be; `None` while free.
    mapped: Option<Arc<OnceLock<Result<(), wgpu::BufferAsyncError>>>>,
}

struct Timer {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    slots: Vec<Slot>,
    /// The slot this frame writes into, if one was free when it began.
    current: Option<usize>,
    next: usize,
    /// Nanoseconds per timestamp tick.
    period: Real,
    serial: u64,
}

/// Writes timestamps between a frame's passes and reads them back, when the device can.
pub(crate) struct GpuTimer(Option<Timer>);

impl GpuTimer {
    /// What a device must have for the timer to run.
    pub(crate) const FEATURES: wgpu::Features =
        wgpu::Features::TIMESTAMP_QUERY.union(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);

    /// A timer on `gpu` if `on` and the device has [`Self::FEATURES`]; one that does nothing
    /// otherwise.
    pub(crate) fn new(gpu: &Gpu, on: bool) -> Self {
        if !on {
            return Self(None);
        }
        if !gpu.device.features().contains(Self::FEATURES) {
            tracing::warn!("this device has no timestamp queries: the GPU's passes go untimed");
            return Self(None);
        }
        let bytes = u64::from(MARKS) * u64::from(wgpu::QUERY_SIZE);
        let queries = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("fk timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: MARKS,
        });
        let resolve = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fk timestamps resolved"),
            size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let slots = (0..SLOTS)
            .map(|_| Slot {
                buffer: gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("fk timestamps read back"),
                    size: bytes,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                names: Vec::new(),
                mapped: None,
            })
            .collect();
        Self(Some(Timer {
            queries,
            resolve,
            slots,
            current: None,
            next: 0,
            period: Real::from(gpu.queue.get_timestamp_period()),
            serial: 0,
        }))
    }

    /// Begins a frame: it is timed if a slot is free to read it back into.
    pub(crate) fn begin(&mut self) {
        let Some(timer) = &mut self.0 else { return };
        let slot = &mut timer.slots[timer.next];
        timer.current = slot.mapped.is_none().then_some(timer.next);
        if timer.current.is_some() {
            slot.names.clear();
        }
    }

    /// Writes a timestamp where the GPU starts on something new: the time since the last one
    /// is not counted.
    pub(crate) fn start(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.write(encoder, None);
    }

    /// Writes a timestamp ending the pass `name`, begun at the last one.
    pub(crate) fn mark(&mut self, encoder: &mut wgpu::CommandEncoder, name: &'static str) {
        self.write(encoder, Some(name));
    }

    fn write(&mut self, encoder: &mut wgpu::CommandEncoder, name: Option<&'static str>) {
        let Some(timer) = &mut self.0 else { return };
        let Some(current) = timer.current else { return };
        let names = &mut timer.slots[current].names;
        let index = names.len() as u32;
        if index < MARKS {
            encoder.write_timestamp(&timer.queries, index);
            names.push(name);
        }
    }

    /// Records copying the frame's timestamps out, in its last encoder.
    pub(crate) fn finish(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let Some(timer) = &mut self.0 else { return };
        let Some(current) = timer.current else { return };
        let slot = &timer.slots[current];
        let count = slot.names.len() as u32;
        if count == 0 {
            return;
        }
        encoder.resolve_query_set(&timer.queries, 0..count, &timer.resolve, 0);
        let bytes = u64::from(count) * u64::from(wgpu::QUERY_SIZE);
        encoder.copy_buffer_to_buffer(&timer.resolve, 0, &slot.buffer, 0, bytes);
    }

    /// After the frame's last submission: asks for its timestamps back.
    pub(crate) fn submitted(&mut self) {
        let Some(timer) = &mut self.0 else { return };
        let Some(current) = timer.current.take() else {
            return;
        };
        let slot = &mut timer.slots[current];
        if slot.names.is_empty() {
            return;
        }
        let mapped = Arc::new(OnceLock::new());
        let done = mapped.clone();
        slot.buffer
            .map_async(wgpu::MapMode::Read, .., move |result| {
                let _ = done.set(result);
            });
        slot.mapped = Some(mapped);
        timer.next = (current + 1) % SLOTS;
    }

    /// The latest frame read back since the last call, if any, without waiting for the GPU.
    pub(crate) fn collect(&mut self, gpu: &Gpu) -> Option<GpuTimings> {
        let timer = self.0.as_mut()?;
        if timer.slots.iter().all(|slot| slot.mapped.is_none()) {
            return None;
        }
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        let mut latest = None;
        // Oldest first, from the slot after the one written last.
        for offset in 0..SLOTS {
            let index = (timer.next + offset) % SLOTS;
            let slot = &mut timer.slots[index];
            let Some(result) = slot.mapped.as_ref().and_then(|mapped| mapped.get()) else {
                continue;
            };
            match result {
                Ok(()) => {
                    if let Ok(bytes) = slot.buffer.get_mapped_range(..) {
                        timer.serial += 1;
                        latest = Some(GpuTimings {
                            serial: timer.serial,
                            passes: passes(&bytes, &slot.names, timer.period),
                        });
                    }
                    slot.buffer.unmap();
                }
                Err(error) => tracing::warn!("timestamps: {error}"),
            }
            slot.mapped = None;
        }
        latest
    }
}

/// The passes ending at each named mark, from the raw timestamps in `bytes`, `period`
/// nanoseconds a tick.
fn passes(bytes: &[u8], names: &[Option<&'static str>], period: Real) -> Vec<(&'static str, Real)> {
    let ticks: Vec<u64> = bytes
        .as_chunks::<8>()
        .0
        .iter()
        .take(names.len())
        .map(|chunk| u64::from_le_bytes(*chunk))
        .collect();
    let mut passes: Vec<(&'static str, Real)> = Vec::new();
    for (i, name) in names.iter().enumerate().skip(1) {
        let Some(name) = name else { continue };
        // Ticks running backwards (a reordered or untimed write) count as nothing.
        let seconds = ticks[i].saturating_sub(ticks[i - 1]) as Real * period * 1e-9;
        match passes.iter_mut().find(|(seen, _)| seen == name) {
            Some((_, total)) => *total += seconds,
            None => passes.push((name, seconds)),
        }
    }
    passes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passes_run_from_mark_to_mark_and_skip_the_gaps() {
        let ticks: [u64; 5] = [100, 400, 1000, 1500, 1600];
        let bytes: Vec<u8> = ticks.iter().flat_map(|tick| tick.to_le_bytes()).collect();
        let names = [None, Some("a"), None, Some("b"), Some("a")];
        let passes = passes(&bytes, &names, 2.0);
        let names: Vec<_> = passes.iter().map(|(name, _)| *name).collect();
        assert_eq!(names, ["a", "b"]);
        assert!((passes[0].1 - 800e-9).abs() < 1e-15);
        assert!((passes[1].1 - 1000e-9).abs() < 1e-15);
    }
}
