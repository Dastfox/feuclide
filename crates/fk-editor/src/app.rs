//! The editor's window: the panes laid out, the menus, the shortcuts, the scene file.

use std::path::PathBuf;

use crate::Options;
use crate::core::{EditorWorld, Selection, ThingId};
use crate::panes::tuning::Tuning;
use crate::scene::{LightingDoc, SceneDoc};
use crate::undo::History;
use crate::viewport::Viewport;

/// Opens the editor's window and runs until it is closed.
pub(crate) fn run(options: Options) -> eframe::Result {
    let native = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Feuclide Editor")
            .with_inner_size([1600.0, 940.0]),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    eframe::run_native(
        "fk-editor",
        native,
        Box::new(move |cc| Ok(Box::new(Editor::new(cc, options)))),
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Middle {
    Viewport,
    Tuning,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Side {
    Properties,
    Environment,
}

struct Editor {
    editor: EditorWorld,
    lighting: LightingDoc,
    viewport: Viewport,
    tuning: Tuning,
    history: History,
    scene_path: String,
    last: Option<SceneDoc>,
    middle: Middle,
    side: Side,
    status: String,
    capture: Option<Capture>,
}

/// `FK_EDITOR_CAPTURE=<file.png>`: one screenshot of the window after `FK_EDITOR_AFTER`
/// seconds (default 3), then the editor closes, as the engine's `FK_CAPTURE` does for an app.
/// With `FK_EDITOR_SELECT=<id>` a thing is selected first, `FK_EDITOR_TAB=tuning` shows the
/// tuning and `FK_EDITOR_TUNING_FILE=<name.toml>` opens one: how to look at the editor from a
/// script.
struct Capture {
    path: PathBuf,
    at: std::time::Instant,
    asked: bool,
    done: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl Editor {
    fn new(cc: &eframe::CreationContext<'_>, options: Options) -> Self {
        cc.egui_ctx.set_theme(egui::Theme::Dark);
        let tuning_dir = options
            .tuning
            .or_else(|| std::env::var_os("FK_TUNING_DIR").map(PathBuf::from))
            .unwrap_or_else(|| PathBuf::from("assets/tuning"));
        let scene_path = options
            .scene
            .clone()
            .unwrap_or_else(|| PathBuf::from("scene.toml"));
        let mut editor = Self {
            editor: EditorWorld::new(),
            lighting: LightingDoc::default(),
            viewport: Viewport::default(),
            tuning: Tuning::new(tuning_dir),
            history: History::default(),
            scene_path: scene_path.display().to_string(),
            last: None,
            middle: Middle::Viewport,
            side: Side::Properties,
            status: String::new(),
            capture: std::env::var_os("FK_EDITOR_CAPTURE").map(|path| {
                let after = std::env::var("FK_EDITOR_AFTER")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(3.0);
                Capture {
                    path: PathBuf::from(path),
                    at: std::time::Instant::now() + std::time::Duration::from_secs_f64(after),
                    asked: false,
                    done: Default::default(),
                }
            }),
        };
        // The file named, if it can be read; the example scene otherwise, saved only when asked.
        let scene = options
            .scene
            .and_then(|path| SceneDoc::load(&path).ok())
            .unwrap_or_else(SceneDoc::example);
        editor.set_scene(scene);
        if let Some(id) = std::env::var("FK_EDITOR_SELECT")
            .ok()
            .and_then(|s| s.parse().ok())
        {
            editor
                .editor
                .world
                .resource_mut::<Selection>()
                .set(ThingId(id));
        }
        if std::env::var("FK_EDITOR_TAB").is_ok_and(|t| t == "tuning") {
            editor.middle = Middle::Tuning;
        }
        if let Ok(file) = std::env::var("FK_EDITOR_TUNING_FILE") {
            editor.tuning.open_named(&file);
        }
        editor
    }

    fn capture_step(&mut self, ctx: &egui::Context) {
        use std::sync::atomic::Ordering;
        let Some(capture) = &mut self.capture else {
            return;
        };
        if capture.done.load(Ordering::SeqCst) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }
        ctx.request_repaint();
        if capture.asked || std::time::Instant::now() < capture.at {
            return;
        }
        capture.asked = true;
        let (path, done) = (capture.path.clone(), capture.done.clone());
        ctx.request_screenshot(move |image| {
            let rgba = image.pixels.iter().flat_map(|c| c.to_array()).collect();
            let image = fk_render::Image {
                width: image.size[0] as u32,
                height: image.size[1] as u32,
                rgba,
            };
            if let Err(e) = image.save(&path) {
                eprintln!("capture: {e}");
            }
            done.store(true, Ordering::SeqCst);
        });
    }

    fn set_scene(&mut self, scene: SceneDoc) {
        scene.restore(&mut self.editor);
        self.lighting = scene.lighting.clone();
        self.editor.sync();
        // As captured, not as read: Euler angles do not come back from a turn to the last bit.
        let settled = self.capture();
        self.last = None;
        self.history.reset(settled);
    }

    fn capture(&mut self) -> SceneDoc {
        SceneDoc::capture(&mut self.editor, &self.lighting)
    }

    fn restore(&mut self, scene: SceneDoc) {
        scene.restore(&mut self.editor);
        self.lighting = scene.lighting;
        self.editor.sync();
    }

    fn undo(&mut self) {
        let now = self.capture();
        if let Some(back) = self.history.undo(now) {
            self.restore(back);
        }
    }

    fn redo(&mut self) {
        let now = self.capture();
        if let Some(forward) = self.history.redo(now) {
            self.restore(forward);
        }
    }

    fn save(&mut self) {
        let scene = self.capture();
        self.status = match scene.save(std::path::Path::new(&self.scene_path)) {
            Ok(()) => format!("saved {}", self.scene_path),
            Err(e) => format!("could not save: {e}"),
        };
    }

    fn load(&mut self) {
        match SceneDoc::load(std::path::Path::new(&self.scene_path)) {
            Ok(scene) => {
                self.set_scene(scene);
                self.status = format!("loaded {}", self.scene_path);
            }
            Err(e) => self.status = format!("could not load: {e}"),
        }
    }

    fn delete_selection(&mut self) {
        let selected = self.editor.world.resource::<Selection>().0.clone();
        for id in selected {
            self.editor.despawn(id);
        }
    }

    /// Copies the primary selection and everything under it, beside it.
    fn duplicate(&mut self) {
        let Some(primary) = self.editor.world.resource::<Selection>().primary() else {
            return;
        };
        let scene = self.capture();
        let mut keep = vec![primary.0];
        let mut copies = Vec::new();
        for doc in &scene.things {
            if doc.id == primary.0 || doc.parent.is_some_and(|p| keep.contains(&p)) {
                keep.push(doc.id);
                copies.push(doc.clone());
            }
        }
        let mut map = std::collections::HashMap::new();
        for doc in &copies {
            let id = self
                .editor
                .world
                .resource_mut::<crate::core::NextId>()
                .take();
            map.insert(doc.id, id.0);
        }
        let mut first = None;
        for mut doc in copies {
            let is_root = doc.id == primary.0;
            doc.id = map[&doc.id];
            if is_root {
                doc.name = format!("{} copy", doc.name);
                doc.translation[0] += 0.5;
                first = Some(ThingId(doc.id));
            } else {
                doc.parent = doc.parent.map(|p| map[&p]);
            }
            self.editor.spawn_with(
                ThingId(doc.id),
                &doc.name,
                doc.thing,
                crate::core::pose_of(doc.translation, doc.rotation),
                doc.parent.map(ThingId),
            );
        }
        if let Some(id) = first {
            self.editor.world.resource_mut::<Selection>().set(id);
        }
    }

    fn shortcuts(&mut self, ctx: &egui::Context) {
        let typing = ctx.egui_wants_keyboard_input();
        let (undo, redo, save, delete, duplicate) = ctx.input_mut(|i| {
            use egui::{Key, KeyboardShortcut, Modifiers};
            let ctrl = |key| KeyboardShortcut::new(Modifiers::COMMAND, key);
            let redo = i.consume_shortcut(&KeyboardShortcut::new(
                Modifiers::COMMAND | Modifiers::SHIFT,
                Key::Z,
            )) || i.consume_shortcut(&ctrl(Key::Y));
            (
                !typing && i.consume_shortcut(&ctrl(Key::Z)),
                !typing && redo,
                i.consume_shortcut(&ctrl(Key::S)),
                !typing && i.key_pressed(Key::Delete),
                !typing && i.consume_shortcut(&ctrl(Key::D)),
            )
        });
        if undo {
            self.undo();
        }
        if redo {
            self.redo();
        }
        if save {
            self.save();
        }
        if delete {
            self.delete_selection();
        }
        if duplicate {
            self.duplicate();
        }
    }

    fn menu_bar(&mut self, ui: &mut egui::Ui) {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("New scene").clicked() {
                    self.set_scene(SceneDoc::default());
                    ui.close();
                }
                if ui.button("Example scene").clicked() {
                    self.set_scene(SceneDoc::example());
                    ui.close();
                }
                ui.separator();
                if ui.button("Load").clicked() {
                    self.load();
                    ui.close();
                }
                if ui.button("Save  (Ctrl+S)").clicked() {
                    self.save();
                    ui.close();
                }
            });
            ui.menu_button("Edit", |ui| {
                let (back, forward) = self.history.depth();
                if ui
                    .add_enabled(back > 0, egui::Button::new("Undo  (Ctrl+Z)"))
                    .clicked()
                {
                    self.undo();
                    ui.close();
                }
                if ui
                    .add_enabled(forward > 0, egui::Button::new("Redo  (Ctrl+Y)"))
                    .clicked()
                {
                    self.redo();
                    ui.close();
                }
                ui.separator();
                if ui.button("Duplicate  (Ctrl+D)").clicked() {
                    self.duplicate();
                    ui.close();
                }
                if ui.button("Delete  (Del)").clicked() {
                    self.delete_selection();
                    ui.close();
                }
            });
            ui.separator();
            ui.selectable_value(&mut self.middle, Middle::Viewport, "Viewport");
            ui.selectable_value(&mut self.middle, Middle::Tuning, "Tuning");
            ui.separator();
            ui.label("Scene file");
            ui.add(egui::TextEdit::singleline(&mut self.scene_path).desired_width(420.0));
        });
    }
}

impl eframe::App for Editor {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.shortcuts(&ctx);
        self.capture_step(&ctx);
        egui::Panel::top("menu").show(ui, |ui| self.menu_bar(ui));
        egui::Panel::bottom("status").show(ui, |ui| {
            ui.horizontal(|ui| {
                let (back, forward) = self.history.depth();
                let count = self.editor.ids().len();
                ui.weak(format!(
                    "{count} things · undo {back} / redo {forward} · image drawn in {:.1} ms",
                    self.viewport.draw_time.as_secs_f64() * 1000.0
                ));
                ui.separator();
                ui.weak(&self.status);
            });
        });
        egui::Panel::left("tree")
            .resizable(true)
            .default_size(240.0)
            .show(ui, |ui| crate::panes::scene_tree::ui(ui, &mut self.editor));
        egui::Panel::right("side")
            .resizable(true)
            .default_size(330.0)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.side, Side::Properties, "Properties");
                    ui.selectable_value(&mut self.side, Side::Environment, "Environment");
                });
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| match self.side {
                    Side::Properties => crate::panes::properties::ui(ui, &mut self.editor),
                    Side::Environment => {
                        crate::panes::properties::environment_ui(ui, &mut self.lighting)
                    }
                });
            });

        self.editor.sync();
        let now = self.capture();
        let dirty = self.last.as_ref() != Some(&now);
        let lighting = self.lighting.lighting();
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ui, |ui| match self.middle {
                Middle::Viewport => self.viewport.ui(ui, &mut self.editor, &lighting, dirty),
                Middle::Tuning => {
                    egui::Frame::central_panel(ui.style()).show(ui, |ui| self.tuning.ui(ui));
                }
            });
        let gesture = ctx.input(|i| i.pointer.any_down()) || ctx.egui_wants_keyboard_input();
        self.history.observe(&now, gesture);
        self.last = Some(now);
    }
}
