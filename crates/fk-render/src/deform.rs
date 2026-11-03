//! Vertex deforms: game-supplied WGSL that shapes meshes on the GPU, and which instances use it.

use bevy_ecs::component::Component;
use bevy_ecs::resource::Resource;
use fk_geometry::GpuGeometry;
use fk_math::Real;

/// A vertex deform: a WGSL module that replaces the engine's `fk::deform` (the identity) for the
/// instances marked with its [`Deform`], plus the modules it imports.
///
/// The module defines
/// `fn deform(position: vec4<f32>, normal: vec4<f32>, iso: mat4x4<f32>, data: vec4<f32>) -> Shaped`
/// under `#define_import_path fk::deform`, usually with the helpers of `fk::displace`
/// (displacement by distance, stretching, a minimum width on screen). It runs in the opaque and
/// the shadow passes alike, and may read the world block, which is how a landscape follows a
/// game value with no work on the CPU.
///
/// Its instances are culled by a ball that must hold every vertex as the module moves it: its
/// [`reach`](Self::reach) says how far that may be. Without one ([`Reach::Unbounded`]) they are
/// culled by the mesh's ball about its origin ([`Mesh::radius`](crate::Mesh::radius)), so build
/// the mesh at its largest and let the deform shrink it.
#[derive(Clone, Debug, PartialEq)]
pub struct DeformShader {
    /// A name for messages, and the file path the module's errors point into.
    pub name: String,
    /// The `fk::deform` module.
    pub source: String,
    /// Modules it imports, as `(path, source)`, in an order where each may import the ones
    /// before it.
    pub libraries: Vec<(String, String)>,
    /// How far the module may move a vertex, whatever the instance's data say and the world
    /// block says this frame: a promise the culling keeps it to, an instance being left out
    /// wherever its mesh's ball grown by this is out of view. A game whose deform moves things
    /// further as its world values grow states it again each frame
    /// ([`Deforms::set_reach`]), before the frame is drawn.
    pub reach: Reach,
}

/// How far a [`DeformShader`] may move a vertex of a mesh from where the mesh has it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Reach {
    /// Anywhere within the mesh's ball about its origin ([`Mesh::radius`](crate::Mesh::radius)),
    /// and no further.
    #[default]
    Unbounded,
    /// Within `metres` plus `of_height` times how far the mesh reaches from its ground
    /// ([`Mesh::height`](crate::Mesh::height): the furthest of its vertices from the plane
    /// through its origin square to the reference frame's up, `origin_frame(1)`, above or
    /// below), from each vertex as built: the bound of a deform that stretches things by their
    /// height (`fk::displace::stretch_up`) and moves them a bounded way (a lean, a sway).
    Within {
        /// A distance any vertex may be moved, metres.
        metres: Real,
        /// A share of the mesh's height any vertex may be moved.
        of_height: Real,
    },
}

impl Reach {
    /// How much further than a mesh's tight ball ([`Mesh::ball`](crate::Mesh::ball)) its
    /// vertices may reach once deformed, for a mesh `height` from its ground; `None` when
    /// unbounded.
    pub fn grows_by(self, height: Real) -> Option<Real> {
        match self {
            Self::Unbounded => None,
            Self::Within { metres, of_height } => {
                Some(metres.max(0.0) + of_height.max(0.0) * height)
            }
        }
    }
}

impl DeformShader {
    /// Composes the raster pipeline's shader with this deform for `G` and validates it, as the
    /// renderer will when it first draws with it; for the game's tests.
    ///
    /// # Errors
    ///
    /// The composition or validation error, pointing into the WGSL.
    pub fn validate<G: GpuGeometry>(&self) -> Result<(), String> {
        use fk_shaders::naga::valid::{Capabilities, ValidationFlags, Validator};
        let module = self.compose(G::SHADER)?;
        Validator::new(ValidationFlags::all(), Capabilities::empty())
            .validate(&module)
            .map(|_| ())
            .map_err(|error| format!("{}: {error:?}", self.name))
    }

    pub(crate) fn compose(&self, geometry: &str) -> Result<fk_shaders::naga::Module, String> {
        let libraries: Vec<_> = self
            .libraries
            .iter()
            .map(|(path, source)| fk_shaders::Module { path, source })
            .collect();
        let hooks = fk_shaders::Hooks {
            deform: Some(fk_shaders::Module {
                path: &self.name,
                source: &self.source,
            }),
            libraries: &libraries,
            ..Default::default()
        };
        fk_shaders::compose_hooked(fk_shaders::Shader::Raster, geometry, &[], &hooks)
            .map_err(|error| error.to_string())
    }
}

/// Which deform, by its index in [`Deforms`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeformHandle(pub(crate) u32);

/// Every vertex deform the raster pipeline can draw with, as a resource. A deform is compiled
/// the first frame after it is added; deforms cannot be changed or removed afterwards.
#[derive(Resource, Debug, Default)]
pub struct Deforms(Vec<DeformShader>);

impl Deforms {
    /// Adds a deform.
    pub fn add(&mut self, shader: DeformShader) -> DeformHandle {
        let index = u32::try_from(self.0.len()).expect("too many deforms");
        self.0.push(shader);
        DeformHandle(index)
    }

    /// Every deform, in the order they were added.
    pub fn all(&self) -> &[DeformShader] {
        &self.0
    }

    /// States how far the deform `handle` may move a vertex from this frame on
    /// ([`DeformShader::reach`]). Only the culling reads it: nothing is compiled again.
    pub fn set_reach(&mut self, handle: DeformHandle, reach: Reach) {
        if let Some(shader) = self.0.get_mut(handle.0 as usize)
            && shader.reach != reach
        {
            shader.reach = reach;
        }
    }

    /// How far the deform `handle` may move a vertex just now ([`DeformShader::reach`]).
    pub fn reach(&self, handle: DeformHandle) -> Reach {
        self.0
            .get(handle.0 as usize)
            .map_or(Reach::Unbounded, |shader| shader.reach)
    }
}

/// Draws an entity's [`MeshInstances`](crate::MeshInstances) through a deform instead of as
/// built.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Deform(pub DeformHandle);

/// Keeps an entity's [`MeshInstances`](crate::MeshInstances) out of the shadow map: they still
/// receive shadows but cast none. For the ground under everything.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CastsNoShadow;
