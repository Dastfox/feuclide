//! What crosses to the GPU: plain `f32` vectors with WGSL layouts, and the device.

use bevy_ecs::resource::Resource;
use bytemuck::{Pod, Zeroable};
use fk_math::Real;
use fk_math::nalgebra::{Matrix4, Vector4};

/// A WGSL `vec2<f32>`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Vec2f(pub [f32; 2]);

/// A WGSL `vec4<f32>`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Vec4f(pub [f32; 4]);

macro_rules! vector_parts {
    ($type:ident, $n:literal) => {
        impl AsRef<[f32; $n]> for $type {
            fn as_ref(&self) -> &[f32; $n] {
                &self.0
            }
        }

        impl AsMut<[f32; $n]> for $type {
            fn as_mut(&mut self) -> &mut [f32; $n] {
                &mut self.0
            }
        }

        impl From<[f32; $n]> for $type {
            fn from(parts: [f32; $n]) -> Self {
                Self(parts)
            }
        }

        encase::impl_vector!($n, $type, f32; using AsRef AsMut From);
    };
}

vector_parts!(Vec2f, 2);
vector_parts!(Vec4f, 4);

impl Vec4f {
    /// Converts an `f64` vector. Only for camera-relative quantities, see [`fk_math::to_gpu`].
    pub fn from_real(v: &Vector4<Real>) -> Self {
        Self(fk_math::to_gpu(v))
    }
}

/// The columns of a matrix as `f32`, the layout of a WGSL `mat4x4<f32>`.
pub fn matrix_columns(m: &Matrix4<Real>) -> [[f32; 4]; 4] {
    std::array::from_fn(|column| std::array::from_fn(|row| m[(row, column)] as f32))
}

/// The device the renderer draws with, as a resource.
///
/// Inserted when the renderer starts, after the window exists. Systems that create their own
/// GPU resources take it from here.
#[derive(Resource, Clone, Debug)]
pub struct Gpu {
    /// The logical device.
    pub device: wgpu::Device,
    /// Its command queue.
    pub queue: wgpu::Queue,
    /// Which adapter it runs on.
    pub adapter: wgpu::AdapterInfo,
}

impl Gpu {
    /// A device with no window, for tests and tools: the first adapter found, or the software
    /// fallback (lavapipe, WARP) when there is no other. `None` where there is none at all.
    pub fn headless() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = [false, true].into_iter().find_map(|fallback| {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: fallback,
                compatible_surface: None,
                apply_limit_buckets: false,
            }))
            .ok()
        })?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("fk headless"),
            ..Default::default()
        }))
        .ok()?;
        Some(Self {
            device,
            queue,
            adapter: adapter.get_info(),
        })
    }
}
