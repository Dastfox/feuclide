# fk-editor

An editor for scenes and tuning files. Part of
[Feuclide](https://github.com/Dastfox/feuclide), a game engine in Rust whose core never assumes
Euclidean geometry.

```sh
cargo install fk-editor
fk-editor [scene.toml] [--tuning <folder>]
```

or from code, `fk_editor::run(fk_editor::Options::default())` (also `feuclide::editor` with
the umbrella's `editor` feature).

## Panes

| Where  | Pane                     | What it does |
|--------|--------------------------|--------------|
| left   | Scene                    | Every thing by its hierarchy. Click to select, drag a row onto another to hang it there, "+ Add" for shapes and lights. |
| right  | Properties / Environment | Name, place and turn (relative to the parent), shape and sizes, colour, point light. The scene's lighting: sun, sky, fog, bands. |
| middle | Viewport                 | The scene drawn by the engine's own renderer (`fk_render::Offscreen`). Orbit, pan, zoom, fly; click to select; drag a gizmo arrow to move. |
| middle | Tuning                   | A folder of `fk_app::tuning` TOML files, every value a field, comments kept and shown as tooltips, `0xRRGGBB` colours as colour pickers. "Live" saves as each edit ends: an app run with `FK_TUNING_DIR` set to that folder reloads it at once. |

Scenes are TOML: the lighting and every thing with its parent, readable and editable by hand.
Only E³ scenes are edited for now.

## Keys

Viewport: right-drag orbit, middle-drag (or shift+right) pan, wheel zoom, right held and
WASD/QE fly, F focus on the selection, click select (ctrl or shift adds), drag a gizmo arrow to
move. Anywhere: Ctrl+Z undo, Ctrl+Y or Ctrl+Shift+Z redo, Ctrl+S save, Ctrl+D duplicate, Del
delete.

`FK_EDITOR_CAPTURE=<file.png>` takes one screenshot of the window after `FK_EDITOR_AFTER`
seconds (3) and quits, to look at the editor from a script; `FK_EDITOR_SELECT=<id>`,
`FK_EDITOR_TAB=tuning` and `FK_EDITOR_TUNING_FILE=<name.toml>` set it up first.

## Bevy counterpart

The Bevy editor prototypes (`bevy_editor_prototypes`), which this follows pane for pane:
`bevy_editor_core`'s selection, `bevy_scene_tree`, `bevy_properties_pane`, `bevy_3d_viewport`
with `bevy_editor_cam`, `bevy_infinite_grid` and `bevy_transform_gizmos`, `bevy_undo`, and BSN
scene files. The differences: it is built on egui rather than on the engine's own UI; the
inspector knows the editor's components, since `bevy_ecs` here has no reflection; and undo
works by whole-scene snapshots, so every edit from any pane is undone alike without each
recording itself.

## License

MIT OR Apache-2.0.
