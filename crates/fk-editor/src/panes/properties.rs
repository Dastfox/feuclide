//! The properties pane, after `bevy_properties_pane`. Bevy's inspects any component through
//! reflection; Feuclide has none, so this pane knows the editor's own components: the name,
//! the pose (place and turn, relative to the parent), the shape and its sizes, the colour and
//! the light. And the scene's lighting, the environment, below.

use bevy_ecs::name::Name;
use egui::DragValue;
use fk_geometry_euclidean::E3;
use fk_math::Real;
use fk_scene::Pose;

use crate::core::{EditorWorld, Lamp, Selection, Shape, Thing, parts_of, pose_of};
use crate::scene::LightingDoc;

/// Draws the properties of the primary selection.
pub fn ui(ui: &mut egui::Ui, editor: &mut EditorWorld) {
    ui.heading("Properties");
    ui.separator();
    let selection = editor.world.resource::<Selection>().clone();
    let Some(id) = selection.primary() else {
        ui.weak("Nothing selected. Click a thing in the viewport or the tree.");
        return;
    };
    let Some(entity) = editor.entity(id) else {
        return;
    };
    if selection.0.len() > 1 {
        ui.weak(format!("{} selected; showing the last.", selection.0.len()));
    }

    let mut name = editor
        .world
        .get::<Name>(entity)
        .map(|n| n.as_str().to_owned())
        .unwrap_or_default();
    ui.horizontal(|ui| {
        ui.label("Name");
        if ui.text_edit_singleline(&mut name).changed() {
            editor
                .world
                .entity_mut(entity)
                .insert(Name::new(name.clone()));
        }
    });
    ui.weak(format!("id {}", id.0));

    egui::CollapsingHeader::new("Transform")
        .default_open(true)
        .show(ui, |ui| {
            let Some(pose) = editor.world.get::<Pose<E3>>(entity).map(|p| p.0) else {
                return;
            };
            let (mut t, mut r) = parts_of(&pose);
            let mut changed = false;
            egui::Grid::new("transform").num_columns(4).show(ui, |ui| {
                ui.label("Place");
                for v in &mut t {
                    changed |= ui
                        .add(DragValue::new(v).speed(0.01).max_decimals(3))
                        .changed();
                }
                ui.end_row();
                ui.label("Turn °");
                for v in &mut r {
                    changed |= ui
                        .add(DragValue::new(v).speed(0.5).max_decimals(1))
                        .changed();
                }
                ui.end_row();
            });
            if changed {
                editor
                    .world
                    .entity_mut(entity)
                    .insert(Pose::<E3>(pose_of(t, r)));
            }
            ui.weak("Relative to its parent. Drag a gizmo arrow in the viewport to move it.");
        });

    let Some(mut thing) = editor.world.get::<Thing>(entity).cloned() else {
        return;
    };
    let before = thing.clone();
    egui::CollapsingHeader::new("Shape")
        .default_open(true)
        .show(ui, |ui| shape_ui(ui, &mut thing));
    egui::CollapsingHeader::new("Light")
        .default_open(true)
        .show(ui, |ui| {
            let mut on = thing.light.is_some();
            if ui.checkbox(&mut on, "Shines").changed() {
                thing.light = on.then(Lamp::default);
            }
            if let Some(lamp) = &mut thing.light {
                egui::Grid::new("lamp").num_columns(2).show(ui, |ui| {
                    ui.label("Colour");
                    color(ui, &mut lamp.color);
                    ui.end_row();
                    ui.label("Intensity");
                    ui.add(
                        DragValue::new(&mut lamp.intensity)
                            .speed(0.02)
                            .range(0.0..=100.0),
                    );
                    ui.end_row();
                    ui.label("Range");
                    ui.add(
                        DragValue::new(&mut lamp.range)
                            .speed(0.05)
                            .range(0.1..=200.0),
                    );
                    ui.end_row();
                });
            }
        });
    if thing != before {
        editor.world.entity_mut(entity).insert(thing);
    }
}

fn shape_ui(ui: &mut egui::Ui, thing: &mut Thing) {
    egui::ComboBox::from_id_salt("shape kind")
        .selected_text(thing.shape.label())
        .show_ui(ui, |ui| {
            for kind in Shape::KINDS {
                if ui
                    .selectable_label(
                        std::mem::discriminant(&kind) == std::mem::discriminant(&thing.shape),
                        kind.label(),
                    )
                    .clicked()
                {
                    thing.shape = kind;
                }
            }
        });
    let len = |ui: &mut egui::Ui, label: &str, v: &mut Real| {
        ui.label(label);
        ui.add(
            DragValue::new(v)
                .speed(0.01)
                .range(0.001..=500.0)
                .max_decimals(3),
        );
        ui.end_row();
    };
    egui::Grid::new("shape")
        .num_columns(4)
        .show(ui, |ui| match &mut thing.shape {
            Shape::Empty => {}
            Shape::Cuboid { half } => vec3(ui, "Half size", half),
            Shape::Bevelled { half, bevel } => {
                vec3(ui, "Half size", half);
                len(ui, "Bevel", bevel);
            }
            Shape::Sphere { radius } => len(ui, "Radius", radius),
            Shape::Cylinder { radius, height } | Shape::Cone { radius, height } => {
                len(ui, "Radius", radius);
                len(ui, "Height", height);
            }
            Shape::Ground { size } => len(ui, "Size", size),
            Shape::Stone { scale, lumps, seed } => {
                vec3(ui, "Size", scale);
                len(ui, "Lumps", lumps);
                ui.label("Seed");
                ui.add(DragValue::new(seed).speed(0.2));
                ui.end_row();
            }
        });
    if thing.shape != Shape::Empty {
        ui.horizontal(|ui| {
            ui.label("Colour");
            color(ui, &mut thing.color);
        });
    }
}

/// A colour button that writes back only when it was used: egui's turns a linear colour
/// through other spaces and back, which changes its last bits every frame otherwise.
fn color(ui: &mut egui::Ui, rgb: &mut [f32; 3]) {
    let mut edited = *rgb;
    if egui::color_picker::color_edit_button_rgb(ui, &mut edited).changed() {
        *rgb = edited;
    }
}

fn vec3(ui: &mut egui::Ui, label: &str, v: &mut [Real; 3]) {
    ui.label(label);
    for x in v.iter_mut() {
        ui.add(
            DragValue::new(x)
                .speed(0.01)
                .range(0.001..=500.0)
                .max_decimals(3),
        );
    }
    ui.end_row();
}

/// The scene's lighting.
pub fn environment_ui(ui: &mut egui::Ui, lighting: &mut LightingDoc) {
    ui.heading("Environment");
    ui.separator();
    egui::Grid::new("environment")
        .num_columns(2)
        .show(ui, |ui| {
            ui.label("Towards the sun");
            ui.horizontal(|ui| {
                for v in &mut lighting.sun {
                    ui.add(
                        DragValue::new(v)
                            .speed(0.01)
                            .range(-1.0..=1.0)
                            .max_decimals(2),
                    );
                }
            });
            ui.end_row();
            for (label, c) in [
                ("Sun", &mut lighting.sun_color),
                ("Shadow tint", &mut lighting.shadow_tint),
                ("Sky above", &mut lighting.zenith),
                ("Horizon", &mut lighting.horizon),
            ] {
                ui.label(label);
                color(ui, c);
                ui.end_row();
            }
            ui.label("Fog");
            ui.add(
                DragValue::new(&mut lighting.fog_density)
                    .speed(0.001)
                    .range(0.0..=1.0)
                    .max_decimals(4),
            );
            ui.end_row();
            ui.label("Bands");
            ui.add(DragValue::new(&mut lighting.bands).range(1..=64));
            ui.end_row();
            ui.label("Smooth");
            ui.add(egui::Slider::new(&mut lighting.smooth, 0.0..=1.0));
            ui.end_row();
        });
    if ui.button("Reset to the engine's").clicked() {
        *lighting = LightingDoc::default();
    }
}
