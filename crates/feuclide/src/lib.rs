//! Feuclide: a game engine whose core does not assume Euclidean geometry.
//!
//! This crate re-exports the engine's crates. Most programs only need the [`prelude`].

pub use fk_app as app;
pub use fk_audio as audio;
pub use fk_geometry as geometry;
pub use fk_geometry_euclidean as euclidean;
pub use fk_math as math;
pub use fk_physics as physics;
pub use fk_quotient as quotient;
pub use fk_render as render;
pub use fk_scene as scene;
pub use fk_shaders as shaders;

/// The types and traits most programs need.
pub mod prelude {
    pub use fk_app::keymap::{
        Action, ActionKind, Binding, Bindings, KeyMap, KeyMapPlugin, KeyMapSystems, Rebind,
        RebindTarget, Resolution,
    };
    pub use fk_app::{
        App, AppExit, ButtonInput, CursorGrab, FixedUpdate, Handover, KeyCode, MouseButton, Plugin,
        PostUpdate, PreUpdate, RawInput, Render, Startup, Time, Update, WindowConfig, WindowSize,
        config_dir,
    };
    pub use fk_audio::{
        Acoustics, Audio, AudioPlugin, Emitter, Outside, Playback, Sound, Voice, decibels,
    };
    pub use fk_geometry::{
        Algebra, Geometry, GpuGeometry, GroupElement, SampledMetric, VectorSpace,
    };
    pub use fk_geometry_euclidean::{E2, E3};
    pub use fk_math::Real;
    pub use fk_physics::{
        Character, CharacterShape, Collider, Colliders, DistanceField, FieldBody, Gravity, Ground,
        Inertia, RigidBody, Surroundings, move_through,
    };
    pub use fk_quotient::Quotient;
    pub use fk_render::{
        Afterimage, CarriedImage, CastsNoShadow, Color, Deform, DeformShader, Deforms, DiscShape,
        Elsewhere, Gauge, Hidden, Ink, Instance, LayerRender, Lids, Lighting, Mark, MeshBuilder,
        MeshInstances, Meshes, Mosh, MoshKind, MoshPatch, Multisampling, OnLayer, PostMode,
        PostSettings, RayMarched, RayMarching, RenderPlugin, Screenshots, SdfShader, Sdfs, Shadows,
        ShowsElsewhere, Sky, SkyDisc, TextPanel, Tunnel, Warp, WorldUniforms, shapes,
    };
    pub use fk_scene::{
        ActiveView, Camera, ChartTag, FloatingOrigin, GlobalPose, Pose, ScenePlugin, SceneSystems,
        Velocity,
    };
}
