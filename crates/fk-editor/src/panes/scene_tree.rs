//! The scene tree, after `bevy_scene_tree`: every thing by its hierarchy, click to select,
//! drag one onto another to hang it there (onto the root row to take it down), and a menu to
//! add things.

use bevy_ecs::name::Name;
use bevy_ecs::prelude::Entity;
use fk_geometry::GroupElement;
use fk_geometry_euclidean::Se3;

use crate::core::{EditorWorld, Lamp, Selection, Shape, Thing, ThingId};

/// What the tree asks for, done by the app after the pane is drawn.
pub enum TreeAction {
    /// Hang `0` under `1`, or at the root.
    Reparent(Entity, Option<Entity>),
}

/// Draws the pane.
pub fn ui(ui: &mut egui::Ui, editor: &mut EditorWorld) {
    ui.horizontal(|ui| {
        ui.heading("Scene");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            add_menu(ui, editor);
        });
    });
    ui.separator();
    let mut actions = Vec::new();
    egui::ScrollArea::vertical()
        .auto_shrink(false)
        .show(ui, |ui| {
            let root = ui.label(egui::RichText::new("(root: drop here to unparent)").weak());
            if let Some(dragged) = root.dnd_release_payload::<Entity>() {
                actions.push(TreeAction::Reparent(*dragged, None));
            }
            for entity in editor.roots() {
                row(ui, editor, entity, &mut actions);
            }
        });
    for action in actions {
        match action {
            TreeAction::Reparent(entity, parent) => editor.reparent(entity, parent),
        }
    }
}

fn row(ui: &mut egui::Ui, editor: &mut EditorWorld, entity: Entity, actions: &mut Vec<TreeAction>) {
    let Some(id) = editor.id(entity) else { return };
    let name = editor
        .world
        .get::<Name>(entity)
        .map(|n| n.as_str().to_owned())
        .unwrap_or_default();
    let thing = editor
        .world
        .get::<Thing>(entity)
        .cloned()
        .unwrap_or_default();
    let (icon, tint) = match (thing.light.is_some(), thing.shape) {
        (true, _) => ("•", egui::Color32::from_rgb(250, 210, 90)),
        (false, Shape::Empty) => ("•", egui::Color32::GRAY),
        _ => {
            let [r, g, b] = thing
                .color
                .map(|c| (c.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8);
            ("•", egui::Color32::from_rgb(r, g, b))
        }
    };
    let children = editor.children(entity);
    let selected = editor.world.resource::<Selection>().contains(id);
    let mut label = egui::text::LayoutJob::default();
    let font = egui::TextStyle::Body.resolve(ui.style());
    label.append(icon, 0.0, egui::TextFormat::simple(font.clone(), tint));
    label.append(
        &format!(" {name}"),
        0.0,
        egui::TextFormat::simple(font, ui.visuals().text_color()),
    );

    let mut draw_label = |ui: &mut egui::Ui, editor: &mut EditorWorld| {
        let source = ui.dnd_drag_source(ui.scope_id().with(("thing", id.0)), entity, |ui| {
            ui.selectable_label(selected, label.clone())
        });
        let response = source.response;
        // A click lands on the label inside the drag source, or on the source itself.
        if source.inner.clicked() || response.clicked() {
            let toggle = ui.input(|i| i.modifiers.ctrl || i.modifiers.shift);
            let mut selection = editor.world.resource_mut::<Selection>();
            if toggle {
                selection.toggle(id);
            } else {
                selection.set(id);
            }
        }
        if let Some(dragged) = response.dnd_release_payload::<Entity>()
            && *dragged != entity
        {
            actions.push(TreeAction::Reparent(*dragged, Some(entity)));
        }
    };

    if children.is_empty() {
        ui.horizontal(|ui| {
            ui.add_space(18.0);
            draw_label(ui, editor);
        });
    } else {
        let collapsing = egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(),
            ui.scope_id().with(("tree", id.0)),
            true,
        );
        collapsing
            .show_header(ui, |ui| draw_label(ui, editor))
            .body(|ui| {
                for child in children {
                    row(ui, editor, child, actions);
                }
            });
    }
}

fn add_menu(ui: &mut egui::Ui, editor: &mut EditorWorld) {
    ui.menu_button("+ Add", |ui| {
        let parent = editor.world.resource::<Selection>().primary();
        for shape in Shape::KINDS {
            if ui.button(shape.label()).clicked() {
                let thing = Thing {
                    shape,
                    ..Thing::default()
                };
                let id = spawn_named(editor, shape.label(), thing, None);
                editor.world.resource_mut::<Selection>().set(id);
                ui.close();
            }
        }
        ui.separator();
        if ui.button("Light").clicked() {
            let thing = Thing {
                shape: Shape::Empty,
                light: Some(Lamp::default()),
                ..Thing::default()
            };
            let id = spawn_named(editor, "Light", thing, None);
            editor.world.resource_mut::<Selection>().set(id);
            ui.close();
        }
        if let Some(parent) = parent {
            ui.separator();
            if ui.button("Child of the selection").clicked() {
                let id = spawn_named(editor, "Child", Thing::default(), Some(parent));
                editor.world.resource_mut::<Selection>().set(id);
                ui.close();
            }
        }
    });
}

fn spawn_named(
    editor: &mut EditorWorld,
    base: &str,
    thing: Thing,
    parent: Option<ThingId>,
) -> ThingId {
    let mut pose = Se3::identity();
    if parent.is_none() && !matches!(thing.shape, Shape::Ground { .. }) {
        pose.translation.y = 0.5;
    }
    let name = format!(
        "{} {}",
        base.to_lowercase(),
        editor.world.resource::<crate::core::NextId>().0 + 1
    );
    editor.spawn(&name, thing, pose, parent)
}
