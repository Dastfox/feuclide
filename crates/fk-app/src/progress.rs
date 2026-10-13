//! What an app is doing while it is built, for whoever waits for it.

use std::cell::RefCell;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

use bevy_ecs::resource::Resource;

/// The step an app has reached while it is built and started: shared between the app (a
/// resource, written by its plugins and its [`Startup`](crate::Startup) systems) and whoever
/// waits for it, such as the app handing it the window ([`Handover::step`](crate::Handover::step)).
///
/// Every app has one ([`App::new`](crate::App::new)); one built for a [`Handover`](crate::Handover)
/// shares the handover's. Each step is logged as it begins, with the time the one before took.
#[derive(Resource, Clone, Debug, Default)]
pub struct Progress(Arc<Mutex<Option<(String, Instant)>>>);

thread_local! {
    /// The progress the app built on this thread reports to.
    static BUILDING: RefCell<Option<Progress>> = const { RefCell::new(None) };
}

impl Progress {
    /// Begins `step`, ending the one before.
    pub fn step(&self, step: impl Into<String>) {
        let step = step.into();
        let mut current = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        match current.as_ref() {
            Some((before, since)) => tracing::info!(
                "{step} (after {before}, {:.2}s)",
                since.elapsed().as_secs_f64()
            ),
            None => tracing::info!("{step}"),
        }
        *current = Some((step, Instant::now()));
    }

    /// The step under way, if any.
    pub fn current(&self) -> Option<String> {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .as_ref()
            .map(|(step, _)| step.clone())
    }

    /// The progress an app built on this thread now reports to: the one being watched, or a
    /// new one.
    pub(crate) fn building() -> Self {
        BUILDING.with(|building| building.borrow().clone().unwrap_or_default())
    }

    /// Has the apps built on this thread from now on report to `self`.
    pub(crate) fn watch_this_thread(&self) {
        BUILDING.with(|building| *building.borrow_mut() = Some(self.clone()));
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::{App, Handover};

    /// Polls `done` until it holds, for at most a few seconds.
    fn wait_for(mut done: impl FnMut() -> bool) -> bool {
        for _ in 0..500 {
            if done() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        false
    }

    #[test]
    fn a_prepared_app_reports_its_steps() {
        let mut handover = Handover::default();
        assert_eq!(handover.step(), None, "nothing prepared");
        let (go_on, wait) = std::sync::mpsc::channel::<()>();
        handover.prepare(move || {
            let mut app = App::new();
            app.step("halfway");
            wait.recv().ok();
            app
        });
        assert!(wait_for(|| handover.step().as_deref() == Some("halfway")));
        go_on.send(()).ok();
        assert!(wait_for(|| handover.is_ready()));
        assert_eq!(handover.step().as_deref(), Some("ready"));
        handover.cancel();
        assert_eq!(handover.step(), None);
    }
}
