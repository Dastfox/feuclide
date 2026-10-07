//! The schedules an [`App`](crate::App) runs, in the order it runs them.

use bevy_ecs::schedule::ScheduleLabel;

/// Runs once, after the window has been created and before the first frame.
#[derive(ScheduleLabel, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Startup;

/// Runs first in every frame. Input is turned into actions here.
#[derive(ScheduleLabel, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PreUpdate;

/// Runs once per [`Time::fixed_step`](crate::Time::fixed_step) of accumulated frame time, zero
/// or more times per frame. Simulation that must not depend on the frame rate goes here.
#[derive(ScheduleLabel, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FixedUpdate;

/// Runs once per frame, after [`FixedUpdate`].
#[derive(ScheduleLabel, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Update;

/// Runs once per frame, after [`Update`].
#[derive(ScheduleLabel, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PostUpdate;

/// Runs last in every frame and draws it.
#[derive(ScheduleLabel, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Render;
