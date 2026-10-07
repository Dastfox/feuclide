//! The list of every key over the image, F1 to show it and F1 again to hide it, as in the
//! game's debug builds. Each example's key enum implements [`Listed`] and calls [`add`]; the
//! list follows the bindings, so it always shows the keys the example reads.

use bevy_ecs::prelude::*;
use fk_app::keymap::{Action, KeyMap};
use fk_app::{App, PostUpdate, WindowSize};
use fk_render::{PostSettings, TextPanel};

/// A key enum the list can show.
pub trait Listed: Action {
    /// The action that shows and hides the list.
    const HELP: Self;

    /// What it does, in a few words of ASCII. Actions next to each other in [`Action::ALL`]
    /// that say the same are one line, their keys joined by `/`.
    fn describe(self) -> &'static str;
}

/// Shows the list of `A`'s keys under `title` while [`Listed::HELP`] has toggled it on.
pub fn add<A: Listed>(app: &mut App, title: &'static str) {
    app.add_systems(
        PostUpdate,
        move |map: Res<KeyMap<A>>,
              window: Res<WindowSize>,
              mut post: ResMut<PostSettings>,
              mut shown: Local<bool>| {
            if map.pressed(A::HELP) {
                *shown = !*shown;
            }
            if !*shown {
                post.text.lines.clear();
                return;
            }
            let lines = lines(&map, title);
            // As large as fits the window, in whole pixels: 8 by 10 font pixels a character, a
            // margin of 4 around them.
            let wide = 8 * lines.iter().map(String::len).max().unwrap_or(0) as u32 + 8;
            let tall = 10 * lines.len() as u32 + 8;
            let scale = (window.width.saturating_sub(32) / wide)
                .min(window.height.saturating_sub(32) / tall)
                .clamp(1, 3);
            post.text = TextPanel {
                lines,
                scale,
                ..TextPanel::default()
            };
        },
    );
}

fn lines<A: Listed>(map: &KeyMap<A>, title: &str) -> Vec<String> {
    let keys = |action: A| {
        let names: Vec<String> = map
            .bindings()
            .get(action)
            .iter()
            .map(ToString::to_string)
            .collect();
        if names.is_empty() {
            "-".to_owned()
        } else {
            names.join(", ")
        }
    };
    let mut rows: Vec<(String, &str)> = Vec::new();
    for &action in A::ALL {
        let what = action.describe();
        match rows.last_mut() {
            Some((joined, last)) if *last == what => {
                joined.push('/');
                joined.push_str(&keys(action));
            }
            _ => rows.push((keys(action), what)),
        }
    }
    let width = rows.iter().map(|(keys, _)| keys.len()).max().unwrap_or(0);
    let mut lines = vec![
        format!("{title}: every key ({} hides)", keys(A::HELP)),
        String::new(),
    ];
    lines.extend(
        rows.into_iter()
            .map(|(keys, what)| format!("  {keys:<width$}  {what}")),
    );
    lines
}
