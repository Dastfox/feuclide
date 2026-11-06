//! The 3D viewport, after `bevy_3d_viewport`, `bevy_editor_cam`, `bevy_transform_gizmos` and
//! the selection box: the scene drawn by the engine's own renderer (`fk_render::Offscreen`,
//! the raster path the golden images use) into a texture shown in the pane, an orbiting
//! camera, picking by the meshes' triangles, the selection outlined and a gizmo to move it.
//!
//! The image is drawn again only when the scene, the light, the camera or the pane's size
//! changed (Bevy's editor runs reactive the same way), and the renderer is rebuilt for a new
//! size only once the pane has stopped being resized.

use std::time::{Duration, Instant};

use egui::{Color32, Pos2, Rect, Sense, Stroke, Vec2};
use fk_geometry::GroupElement;
use fk_geometry_euclidean::{E3, Se3};
use fk_math::Real;
use fk_math::nalgebra::{UnitQuaternion, Vector3};
use fk_render::{Gpu, Lighting, Offscreen, Paths};
use fk_scene::{Camera, Pose};

use crate::core::{EditorWorld, MeshCache, Selection, Shape, Thing, ThingId, apply, inverse};

/// An eye turning about a point it looks at.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrbitCam {
    /// What it looks at.
    pub target: Vector3<Real>,
    /// Turned about the vertical, radians.
    pub yaw: Real,
    /// Up (positive) or down, radians.
    pub pitch: Real,
    /// How far back from the target.
    pub distance: Real,
    /// Vertical field of view, radians.
    pub fov_y: Real,
}

impl Default for OrbitCam {
    fn default() -> Self {
        Self {
            target: Vector3::new(0.0, 0.4, 0.0),
            yaw: 0.6,
            pitch: -0.45,
            distance: 6.0,
            fov_y: 0.9,
        }
    }
}

impl OrbitCam {
    /// The eye's pose: looking along its −z, its y up.
    pub fn eye(&self) -> Se3 {
        let rotation = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), self.yaw)
            * UnitQuaternion::from_axis_angle(&Vector3::x_axis(), self.pitch);
        Se3 {
            rotation,
            translation: self.target + rotation * Vector3::new(0.0, 0.0, self.distance),
        }
    }
}

const NEAR: Real = 0.05;
const FAR: Real = 400.0;

/// From the world to the pane's points and back.
struct Projector {
    eye: Se3,
    to_eye: Se3,
    tan_y: Real,
    aspect: Real,
    rect: Rect,
}

impl Projector {
    fn new(cam: &OrbitCam, rect: Rect) -> Self {
        let eye = cam.eye();
        Self {
            eye,
            to_eye: inverse(&eye),
            tan_y: (cam.fov_y / 2.0).tan(),
            aspect: Real::from(rect.width() / rect.height().max(1.0)),
            rect,
        }
    }

    fn screen_of_eye(&self, c: Vector3<Real>) -> Option<Pos2> {
        if c.z > -NEAR {
            return None;
        }
        let x = c.x / (-c.z * self.tan_y * self.aspect);
        let y = c.y / (-c.z * self.tan_y);
        let centre = self.rect.center();
        Some(Pos2::new(
            centre.x + (x as f32) * self.rect.width() / 2.0,
            centre.y - (y as f32) * self.rect.height() / 2.0,
        ))
    }

    fn screen(&self, p: Vector3<Real>) -> Option<Pos2> {
        self.screen_of_eye(apply(&self.to_eye, &p))
    }

    /// The part of a segment ahead of the near plane, on screen.
    fn segment(&self, a: Vector3<Real>, b: Vector3<Real>) -> Option<(Pos2, Pos2)> {
        let (mut a, mut b) = (apply(&self.to_eye, &a), apply(&self.to_eye, &b));
        let limit = -NEAR * 1.01;
        if a.z > limit && b.z > limit {
            return None;
        }
        if a.z > limit {
            a = b + (a - b) * ((limit - b.z) / (a.z - b.z));
        } else if b.z > limit {
            b = a + (b - a) * ((limit - a.z) / (b.z - a.z));
        }
        Some((self.screen_of_eye(a)?, self.screen_of_eye(b)?))
    }

    /// The ray through a point of the pane: its origin and unit direction, in the world.
    fn ray(&self, at: Pos2) -> (Vector3<Real>, Vector3<Real>) {
        let centre = self.rect.center();
        let x = Real::from((at.x - centre.x) / (self.rect.width() / 2.0));
        let y = Real::from(-(at.y - centre.y) / (self.rect.height() / 2.0));
        let d = Vector3::new(x * self.tan_y * self.aspect, y * self.tan_y, -1.0).normalize();
        (self.eye.translation, self.eye.rotation * d)
    }
}

/// A gizmo handle being dragged.
#[derive(Clone, Copy)]
struct GizmoDrag {
    id: ThingId,
    axis: usize,
}

/// The viewport pane.
pub struct Viewport {
    /// Its camera.
    pub cam: OrbitCam,
    gpu: Option<Option<Gpu>>,
    offscreen: Option<(Offscreen<E3>, [u32; 2])>,
    texture: Option<egui::TextureHandle>,
    drawn: Option<(OrbitCam, [u32; 2])>,
    wanted: Option<([u32; 2], Instant)>,
    drag: Option<GizmoDrag>,
    /// How long the last image took to draw.
    pub draw_time: Duration,
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            cam: OrbitCam::default(),
            gpu: None,
            offscreen: None,
            texture: None,
            drawn: None,
            wanted: None,
            drag: None,
            draw_time: Duration::ZERO,
        }
    }
}

const AXES: [(Vector3<Real>, Color32); 3] = [
    (Vector3::new(1.0, 0.0, 0.0), Color32::from_rgb(230, 70, 70)),
    (Vector3::new(0.0, 1.0, 0.0), Color32::from_rgb(90, 210, 90)),
    (Vector3::new(0.0, 0.0, 1.0), Color32::from_rgb(80, 130, 240)),
];

impl Viewport {
    /// Draws the pane. `dirty`: the scene or its light changed since the last frame.
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        editor: &mut EditorWorld,
        lighting: &Lighting,
        dirty: bool,
    ) {
        let gpu = self.gpu.get_or_insert_with(Gpu::headless).clone();
        let Some(gpu) = gpu else {
            ui.centered_and_justified(|ui| ui.label("No GPU adapter for the engine's renderer."));
            return;
        };
        let (rect, response) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        if rect.width() < 8.0 || rect.height() < 8.0 {
            return;
        }
        self.controls(ui, &response, editor, rect);

        let ppp = ui.ctx().pixels_per_point();
        let size = [
            ((rect.width() * ppp) as u32).clamp(16, 3840),
            ((rect.height() * ppp) as u32).clamp(16, 2160),
        ];
        self.resize(&gpu, size);
        if let Some((offscreen, built)) = &mut self.offscreen {
            let key = (self.cam, *built);
            if dirty || self.drawn != Some(key) || self.texture.is_none() {
                let mut cam = self.cam;
                cam.fov_y = self.cam.fov_y;
                let started = Instant::now();
                let image = offscreen.draw(
                    &mut editor.world,
                    &cam.eye(),
                    &Camera::new(cam.fov_y, NEAR, FAR),
                    lighting,
                    Paths::Raster,
                );
                self.draw_time = started.elapsed();
                let mut rgba = image.rgba;
                for px in rgba.as_chunks_mut::<4>().0 {
                    px[3] = 255;
                }
                let color = egui::ColorImage::from_rgba_unmultiplied(
                    [image.width as usize, image.height as usize],
                    &rgba,
                );
                match &mut self.texture {
                    Some(texture) => texture.set(color, egui::TextureOptions::LINEAR),
                    None => {
                        self.texture = Some(ui.ctx().load_texture(
                            "fk viewport",
                            color,
                            egui::TextureOptions::LINEAR,
                        ));
                    }
                }
                self.drawn = Some(key);
            }
        }
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, Color32::from_gray(20));
        if let Some(texture) = &self.texture {
            painter.image(
                texture.id(),
                rect,
                Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                Color32::WHITE,
            );
        }
        self.overlay(&painter, editor, rect);
        painter.text(
            rect.left_bottom() + Vec2::new(8.0, -8.0),
            egui::Align2::LEFT_BOTTOM,
            "right-drag orbit · middle-drag pan · wheel zoom · right + WASDQE fly · F focus · click select",
            egui::FontId::proportional(11.0),
            Color32::from_white_alpha(110),
        );
    }

    /// Builds the renderer for `size`, at once the first time and then only once the pane has
    /// kept that size a moment (building compiles its pipelines).
    fn resize(&mut self, gpu: &Gpu, size: [u32; 2]) {
        let built = self.offscreen.as_ref().map(|(_, s)| *s);
        if built == Some(size) {
            self.wanted = None;
            return;
        }
        let now = Instant::now();
        let since = match self.wanted {
            Some((wanted, since)) if wanted == size => since,
            _ => {
                self.wanted = Some((size, now));
                now
            }
        };
        if built.is_none() || now.duration_since(since) > Duration::from_millis(250) {
            self.offscreen = Some((Offscreen::new(gpu, size[0], size[1]), size));
            self.wanted = None;
        }
    }

    fn controls(
        &mut self,
        ui: &egui::Ui,
        response: &egui::Response,
        editor: &mut EditorWorld,
        rect: Rect,
    ) {
        let input = ui.input(|i| i.clone());
        let projector = Projector::new(&self.cam, rect);
        if response.hovered() {
            let scroll = input.smooth_scroll_delta.y;
            if scroll != 0.0 {
                self.cam.distance =
                    (self.cam.distance * (-Real::from(scroll) * 0.002).exp()).clamp(0.2, 300.0);
            }
        }
        let delta = response.drag_delta();
        let shift = input.modifiers.shift;
        if response.dragged_by(egui::PointerButton::Middle)
            || (shift && response.dragged_by(egui::PointerButton::Secondary))
        {
            let k = self.cam.distance * 2.0 * projector.tan_y / Real::from(rect.height());
            let right = projector.eye.rotation * Vector3::x();
            let up = projector.eye.rotation * Vector3::y();
            self.cam.target -= right * Real::from(delta.x) * k;
            self.cam.target += up * Real::from(delta.y) * k;
        } else if response.dragged_by(egui::PointerButton::Secondary) {
            self.cam.yaw -= Real::from(delta.x) * 0.008;
            self.cam.pitch = (self.cam.pitch - Real::from(delta.y) * 0.008).clamp(-1.55, 1.55);
        }
        if response.is_pointer_button_down_on() && input.pointer.secondary_down() {
            let dt = Real::from(input.stable_dt.min(0.1));
            let speed = self.cam.distance.max(1.0) * 1.2 * dt;
            let r = projector.eye.rotation;
            let mut go = Vector3::zeros();
            for (key, way) in [
                (egui::Key::W, -Vector3::z()),
                (egui::Key::S, Vector3::z()),
                (egui::Key::A, -Vector3::x()),
                (egui::Key::D, Vector3::x()),
            ] {
                if input.key_down(key) {
                    go += r * way;
                }
            }
            if input.key_down(egui::Key::E) {
                go += Vector3::y();
            }
            if input.key_down(egui::Key::Q) {
                go -= Vector3::y();
            }
            if go.norm() > 0.0 {
                self.cam.target += go.normalize() * speed;
                ui.ctx().request_repaint();
            }
        }
        if response.hovered()
            && !ui.ctx().egui_wants_keyboard_input()
            && input.key_pressed(egui::Key::F)
            && let Some(entity) = editor
                .world
                .resource::<Selection>()
                .primary()
                .and_then(|id| editor.entity(id))
        {
            self.cam.target = editor.global(entity).translation;
        }

        // The gizmo first, then picking.
        let pointer = input.pointer.interact_pos();
        if response.drag_started_by(egui::PointerButton::Primary)
            && let Some(at) = input.pointer.press_origin()
        {
            self.drag = self.gizmo_hit(editor, &projector, at);
        }
        if let Some(drag) = self.drag {
            if response.dragged_by(egui::PointerButton::Primary) {
                self.gizmo_drag(editor, &projector, drag, delta);
            } else {
                self.drag = None;
            }
        }
        if response.clicked_by(egui::PointerButton::Primary)
            && let Some(at) = pointer
        {
            let hit = pick(editor, &projector, at);
            let toggle = input.modifiers.ctrl || input.modifiers.shift;
            let mut selection = editor.world.resource_mut::<Selection>();
            match (hit, toggle) {
                (Some(id), true) => selection.toggle(id),
                (Some(id), false) => selection.set(id),
                (None, false) => selection.0.clear(),
                (None, true) => {}
            }
        }
    }

    fn gizmo_ends(
        editor: &mut EditorWorld,
        projector: &Projector,
    ) -> Option<(ThingId, Vector3<Real>, Real)> {
        let id = editor.world.resource::<Selection>().primary()?;
        let entity = editor.entity(id)?;
        let origin = editor.global(entity).translation;
        let length = (origin - projector.eye.translation).norm() * 0.15;
        Some((id, origin, length))
    }

    fn gizmo_hit(
        &self,
        editor: &mut EditorWorld,
        projector: &Projector,
        at: Pos2,
    ) -> Option<GizmoDrag> {
        let (id, origin, length) = Self::gizmo_ends(editor, projector)?;
        let mut best = None;
        for (axis, (way, _)) in AXES.iter().enumerate() {
            let Some((a, b)) = projector.segment(origin, origin + way * length) else {
                continue;
            };
            let d = distance_to_segment(at, a, b);
            if d < 9.0 && best.is_none_or(|(_, bd)| d < bd) {
                best = Some((axis, d));
            }
        }
        best.map(|(axis, _)| GizmoDrag { id, axis })
    }

    fn gizmo_drag(
        &self,
        editor: &mut EditorWorld,
        projector: &Projector,
        drag: GizmoDrag,
        delta: Vec2,
    ) {
        let Some((_, origin, length)) = Self::gizmo_ends(editor, projector) else {
            return;
        };
        let way = AXES[drag.axis].0;
        let Some((a, b)) = projector.segment(origin, origin + way * length) else {
            return;
        };
        let s = b - a;
        if s.length() < 2.0 {
            return;
        }
        let along = Real::from(delta.dot(s / s.length())) * length / Real::from(s.length());
        let Some(entity) = editor.entity(drag.id) else {
            return;
        };
        let parent = editor
            .parent(entity)
            .map(|p| editor.global(p))
            .unwrap_or_else(Se3::identity);
        let local = parent.rotation.inverse() * (way * along);
        if let Some(mut pose) = editor.world.get_mut::<Pose<E3>>(entity) {
            pose.0.translation += local;
        }
        // Moved now, so the gizmo and outline follow within the frame.
        editor.sync();
    }

    fn overlay(&self, painter: &egui::Painter, editor: &mut EditorWorld, rect: Rect) {
        let projector = Projector::new(&self.cam, rect);
        let selected = editor.world.resource::<Selection>().clone();
        // Lights and empties, which have no mesh to click.
        let mut marks = Vec::new();
        let mut boxes = Vec::new();
        for id in editor.ids() {
            let Some(entity) = editor.entity(id) else {
                continue;
            };
            let global = editor.global(entity);
            let Some(thing) = editor.world.get::<Thing>(entity) else {
                continue;
            };
            if thing.light.is_some() || thing.shape == Shape::Empty {
                marks.push((id, global.translation, thing.light.is_some()));
            }
            if selected.contains(id) {
                boxes.push((global, thing.shape, id == selected.primary().unwrap_or(id)));
            }
        }
        for (id, at, light) in marks {
            if let Some(p) = projector.screen(at) {
                let chosen = selected.contains(id);
                let fill = if light {
                    Color32::from_rgb(250, 210, 90)
                } else {
                    Color32::from_gray(200)
                };
                painter.circle(
                    p,
                    5.0,
                    fill,
                    Stroke::new(
                        if chosen { 2.5 } else { 1.0 },
                        if chosen {
                            Color32::from_rgb(255, 150, 40)
                        } else {
                            Color32::BLACK
                        },
                    ),
                );
            }
        }
        let cache = editor.world.resource::<MeshCache>();
        for (global, shape, primary) in boxes {
            let Some(builder) = cache.builder(shape) else {
                continue;
            };
            let (lo, hi) = bounds(builder.positions());
            let corner = |i: usize| {
                Vector3::new(
                    if i & 1 == 0 { lo.x } else { hi.x },
                    if i & 2 == 0 { lo.y } else { hi.y },
                    if i & 4 == 0 { lo.z } else { hi.z },
                )
            };
            let color = if primary {
                Color32::from_rgb(255, 150, 40)
            } else {
                Color32::from_rgb(255, 210, 120)
            };
            for (i, j) in [
                (0, 1),
                (2, 3),
                (4, 5),
                (6, 7),
                (0, 2),
                (1, 3),
                (4, 6),
                (5, 7),
                (0, 4),
                (1, 5),
                (2, 6),
                (3, 7),
            ] {
                if let Some((a, b)) =
                    projector.segment(apply(&global, &corner(i)), apply(&global, &corner(j)))
                {
                    painter.line_segment([a, b], Stroke::new(1.5, color));
                }
            }
        }
        if let Some((_, origin, length)) = Self::gizmo_ends(editor, &projector) {
            for (axis, (way, color)) in AXES.iter().enumerate() {
                if let Some((a, b)) = projector.segment(origin, origin + way * length) {
                    let active = self.drag.is_some_and(|d| d.axis == axis);
                    let color = if active { Color32::YELLOW } else { *color };
                    painter.line_segment([a, b], Stroke::new(3.0, color));
                    painter.circle_filled(b, 5.0, color);
                }
            }
        }
    }
}

/// The thing whose mesh the ray through `at` meets first, or whose mark is under it.
fn pick(editor: &mut EditorWorld, projector: &Projector, at: Pos2) -> Option<ThingId> {
    let (origin, way) = projector.ray(at);
    let mut best: Option<(ThingId, Real)> = None;
    for id in editor.ids() {
        let Some(entity) = editor.entity(id) else {
            continue;
        };
        let global = editor.global(entity);
        let Some(thing) = editor.world.get::<Thing>(entity).cloned() else {
            continue;
        };
        if (thing.light.is_some() || thing.shape == Shape::Empty)
            && let Some(p) = projector.screen(global.translation)
            && p.distance(at) < 8.0
        {
            let d = (global.translation - origin).norm() * 0.999;
            if best.is_none_or(|(_, bd)| d < bd) {
                best = Some((id, d));
            }
        }
        let cache = editor.world.resource::<MeshCache>();
        let Some(builder) = cache.builder(thing.shape) else {
            continue;
        };
        let to_local = inverse(&global);
        let o = apply(&to_local, &origin);
        let d = to_local.rotation * way;
        let positions = builder.positions();
        for tri in builder.indices().as_chunks::<3>().0 {
            let [a, b, c] = [0, 1, 2].map(|k| positions[tri[k] as usize]);
            if let Some(t) = ray_triangle(o, d, a, b, c)
                && best.is_none_or(|(_, bd)| t < bd)
            {
                best = Some((id, t));
            }
        }
    }
    best.map(|(id, _)| id)
}

/// Möller–Trumbore, both faces.
fn ray_triangle(
    o: Vector3<Real>,
    d: Vector3<Real>,
    a: Vector3<Real>,
    b: Vector3<Real>,
    c: Vector3<Real>,
) -> Option<Real> {
    let e1 = b - a;
    let e2 = c - a;
    let p = d.cross(&e2);
    let det = e1.dot(&p);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / det;
    let s = o - a;
    let u = s.dot(&p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(&e1);
    let v = d.dot(&q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(&q) * inv;
    (t > NEAR).then_some(t)
}

fn bounds(points: &[Vector3<Real>]) -> (Vector3<Real>, Vector3<Real>) {
    let mut lo = Vector3::repeat(Real::INFINITY);
    let mut hi = Vector3::repeat(Real::NEG_INFINITY);
    for p in points {
        lo = lo.inf(p);
        hi = hi.sup(p);
    }
    if points.is_empty() {
        (Vector3::zeros(), Vector3::zeros())
    } else {
        (lo, hi)
    }
}

fn distance_to_segment(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_sq().max(1e-6)).clamp(0.0, 1.0);
    p.distance(a + ab * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ray_through_a_projected_point_comes_back_to_it() {
        let cam = OrbitCam::default();
        let rect = Rect::from_min_size(Pos2::new(10.0, 20.0), Vec2::new(800.0, 450.0));
        let projector = Projector::new(&cam, rect);
        let p = Vector3::new(0.7, 0.3, -0.4);
        let at = projector.screen(p).expect("ahead of the eye");
        let (o, d) = projector.ray(at);
        let closest = o + d * (p - o).dot(&d);
        assert!((closest - p).norm() < 1e-4, "{closest} vs {p}");
    }

    #[test]
    fn a_ray_down_the_middle_meets_a_triangle_across_it() {
        let t = ray_triangle(
            Vector3::new(0.0, 0.0, 5.0),
            -Vector3::z(),
            Vector3::new(-1.0, -1.0, 0.0),
            Vector3::new(1.0, -1.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        );
        assert!((t.unwrap() - 5.0).abs() < 1e-9);
    }
}
