use bevy_ecs::resource::Resource;
use fk_math::Real;

/// How long the CPU spent in each schedule of the last whole frame, seconds, as a resource.
///
/// [`App::update`](crate::App::update) writes it once the frame is over, so a system reading it
/// sees the frame before its own. [`render`](Self::render) includes the time the renderer waited
/// for the window's next image, which is where a frame bound by the GPU spends it.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameProfile {
    /// [`PreUpdate`](crate::PreUpdate).
    pub pre_update: Real,
    /// Every [`FixedUpdate`](crate::FixedUpdate) step of the frame together.
    pub fixed_update: Real,
    /// How many fixed steps ran.
    pub fixed_steps: u32,
    /// [`Update`](crate::Update).
    pub update: Real,
    /// [`PostUpdate`](crate::PostUpdate).
    pub post_update: Real,
    /// [`Render`](crate::Render).
    pub render: Real,
}

impl FrameProfile {
    /// The time of every schedule together.
    pub fn total(&self) -> Real {
        self.pre_update + self.fixed_update + self.update + self.post_update + self.render
    }

    /// Each schedule's name (`pre_update`, `fixed_update`, `update`, `post_update`, `render`)
    /// and time, in the order they run.
    pub fn schedules(&self) -> [(&'static str, Real); 5] {
        [
            ("pre_update", self.pre_update),
            ("fixed_update", self.fixed_update),
            ("update", self.update),
            ("post_update", self.post_update),
            ("render", self.render),
        ]
    }
}
