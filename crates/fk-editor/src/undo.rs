//! Undo and redo, after `bevy_undo`, but by whole-scene snapshots: the scenes an editor like
//! this one holds are small, and a snapshot catches every kind of edit (a value dragged, a
//! thing moved by the gizmo, added, deleted, reparented) without each pane recording its own.
//!
//! Each frame the scene is captured; when it differs from the last settled one and no gesture
//! is going on (no button held, no text field focused), the settled one goes on the undo stack.

use crate::scene::SceneDoc;

/// The undo and redo stacks.
#[derive(Default)]
pub struct History {
    settled: Option<SceneDoc>,
    undo: Vec<SceneDoc>,
    redo: Vec<SceneDoc>,
}

/// How many steps back are kept.
const DEPTH: usize = 200;

impl History {
    /// Starts again from `scene`, forgetting every step.
    pub fn reset(&mut self, scene: SceneDoc) {
        self.settled = Some(scene);
        self.undo.clear();
        self.redo.clear();
    }

    /// Sees the scene as it is now; records a step if it changed and the gesture is over.
    /// Returns whether one was recorded.
    pub fn observe(&mut self, now: &SceneDoc, gesture: bool) -> bool {
        if gesture {
            return false;
        }
        match &self.settled {
            Some(settled) if settled == now => false,
            Some(_) => {
                let before = self.settled.replace(now.clone()).expect("settled");
                self.undo.push(before);
                if self.undo.len() > DEPTH {
                    self.undo.remove(0);
                }
                self.redo.clear();
                true
            }
            None => {
                self.settled = Some(now.clone());
                false
            }
        }
    }

    /// The scene one step back, if any; `now` goes on the redo stack.
    pub fn undo(&mut self, now: SceneDoc) -> Option<SceneDoc> {
        let back = self.undo.pop()?;
        self.redo.push(now);
        self.settled = Some(back.clone());
        Some(back)
    }

    /// The scene one step forward, if any.
    pub fn redo(&mut self, now: SceneDoc) -> Option<SceneDoc> {
        let forward = self.redo.pop()?;
        self.undo.push(now);
        self.settled = Some(forward.clone());
        Some(forward)
    }

    /// Steps back and forward.
    pub fn depth(&self) -> (usize, usize) {
        (self.undo.len(), self.redo.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_settled_edit_is_undone_and_redone() {
        let mut history = History::default();
        let a = SceneDoc::default();
        let mut b = SceneDoc::default();
        b.lighting.bands = 9;
        history.observe(&a, false);
        assert!(
            !history.observe(&b, true),
            "mid-gesture nothing is recorded"
        );
        assert!(history.observe(&b, false));
        assert_eq!(history.undo(b.clone()), Some(a.clone()));
        assert_eq!(history.redo(a), Some(b));
    }
}
