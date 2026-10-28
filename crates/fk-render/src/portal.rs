//! Portals within portals, drawn in place: the tree of views seen through the
//! [`Portal`](fk_scene::Portal)s from the frame's eye, each cut to its region.
//!
//! A view's region is where what it draws is kept: a convex cone from its eye through a polygon
//! on the plane of the last mouth it looks through, beyond that plane. The polygon is the
//! mouth's rectangle clipped by its parent's cone, each side of the cone the plane through the
//! eye and one of its edges. All of them are totally geodesic planes, which in the projective
//! models the geometries are embedded in are the zero sets of linear forms on the embedded
//! 4-vectors, so the region is a handful of dot products in every geometry; and since what a
//! portal carries by is an isometry, a child view's eye sees its cone exactly where its parent's
//! saw it: its forms are the same numbers in both frames.
//!
//! Each view drops what lies in its children's regions (its cuts), so that every point a ray
//! reaches is drawn by one view: the views go into one image against one depth buffer, in any
//! order, the depth along a ray through a mouth the distance travelled through it.

use std::collections::HashMap;

use bevy_ecs::resource::Resource;
use bevy_ecs::world::World;
use fk_geometry::{Geometry, GpuGeometry, GroupElement};
use fk_math::Real;
use fk_math::nalgebra::Vector4;
use fk_scene::{Mouth, Portal, Realm, form_through, plane_form};

use crate::{Lighting, Vec4f};

/// How many linear forms a view's region and cuts may hold in all (`fk::view`'s
/// `portal_forms`).
pub const MAX_PORTAL_FORMS: usize = 96;

/// How many regions may be cut out of one view (`fk::view`'s `portal_cuts`).
pub const MAX_PORTAL_CUTS: usize = 8;

/// How far portals are followed, as a resource: [`depth`](Self::depth) portals within portals
/// at most (4), [`views`](Self::views) views in all, the frame's own among them (16), and none
/// covering fewer than [`min_pixels`](Self::min_pixels) of the image (16). The largest on the
/// image are followed first. Each view costs about a frame of what it sees, cut to its part of
/// the image; a mouth not followed shows what lies behind it in its own realm.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct PortalBudget {
    /// Portals within portals, at most.
    pub depth: usize,
    /// Views in all, the frame's own among them.
    pub views: usize,
    /// The least a mouth must cover of the image to be followed, in pixels.
    pub min_pixels: Real,
}

impl Default for PortalBudget {
    fn default() -> Self {
        Self {
            depth: 4,
            views: 16,
            min_pixels: 16.0,
        }
    }
}

/// A convex region of a view: the cone's sides through the eye, each a linear form on embedded
/// points of the eye's frame positive inside, and the plane of the last mouth looked through,
/// positive beyond it. With no sides and no plane, everything.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Region {
    /// The cone's sides, through the eye.
    pub sides: Vec<Vector4<Real>>,
    /// The mouth's plane, positive beyond it.
    pub plane: Option<Vector4<Real>>,
}

impl Region {
    /// Its forms, the sides then the plane.
    pub fn forms(&self) -> impl Iterator<Item = &Vector4<Real>> {
        self.sides.iter().chain(&self.plane)
    }

    /// Whether the embedded point `p` (in the eye's frame) is in it.
    pub fn holds(&self, p: &Vector4<Real>) -> bool {
        self.forms().all(|form| form.dot(p) >= 0.0)
    }

    /// Everything.
    pub fn is_whole(&self) -> bool {
        self.sides.is_empty() && self.plane.is_none()
    }
}

/// One view of the tree: an eye in a realm, what it keeps and what it leaves to others.
pub struct PortalView<G: Geometry> {
    /// The eye, global in [`realm`](Self::realm): the frame's eye carried through every portal
    /// on the way.
    pub eye: G::Isometry,
    /// The realm it sees.
    pub realm: Realm,
    /// How many portals it looks through, 0 for the frame's own.
    pub depth: usize,
    /// The view it is seen through, `None` for the frame's own.
    pub parent: Option<usize>,
    /// Where what it draws is kept.
    pub region: Region,
    /// Where it draws nothing: its children's regions, and those of nearer mouths that hide its
    /// own.
    pub cuts: Vec<Region>,
    /// The rectangle of the image it covers, `[x0, y0, x1, y1]` in normalized device
    /// coordinates.
    pub part: [Real; 4],
    /// How much of the image its polygon covers, in pixels.
    pub pixels: Real,
    /// Whether what it sees casts shadows.
    pub shadows: bool,
    /// How far it sees, in place of the camera's far.
    pub far: Option<Real>,
}

impl<G: Geometry> Clone for PortalView<G> {
    fn clone(&self) -> Self {
        Self {
            eye: self.eye,
            realm: self.realm,
            depth: self.depth,
            parent: self.parent,
            region: self.region.clone(),
            cuts: self.cuts.clone(),
            part: self.part,
            pixels: self.pixels,
            shadows: self.shadows,
            far: self.far,
        }
    }
}

impl<G: Geometry> std::fmt::Debug for PortalView<G> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PortalView")
            .field("eye", &self.eye)
            .field("realm", &self.realm)
            .field("depth", &self.depth)
            .field("parent", &self.parent)
            .field("region", &self.region)
            .field("cuts", &self.cuts)
            .field("part", &self.part)
            .field("pixels", &self.pixels)
            .field("shadows", &self.shadows)
            .field("far", &self.far)
            .finish()
    }
}

impl<G: Geometry> PortalView<G> {
    /// What the view block carries of it: the counts (x the region's forms, y the cuts), each
    /// cut's first form and count, and the forms, the region's first. What does not fit is left
    /// out, the cuts last first.
    pub(crate) fn block(&self) -> (Vec4f, [Vec4f; MAX_PORTAL_CUTS], [Vec4f; MAX_PORTAL_FORMS]) {
        let mut forms = [Vec4f::default(); MAX_PORTAL_FORMS];
        let mut cuts = [Vec4f::default(); MAX_PORTAL_CUTS];
        let mut n = 0;
        let mut put = |form: &Vector4<Real>, n: &mut usize| {
            // Only its sign matters: scaled to keep the f32 well inside its range.
            let scale = form.amax().max(1e-30);
            forms[*n] = Vec4f((form / scale).map(|c| c as f32).into());
            *n += 1;
        };
        let region: Vec<_> = self.region.forms().take(MAX_PORTAL_FORMS).collect();
        for form in &region {
            put(form, &mut n);
        }
        let mut count = 0;
        for cut in &self.cuts {
            let length = cut.forms().count();
            if count == MAX_PORTAL_CUTS || n + length > MAX_PORTAL_FORMS {
                break;
            }
            cuts[count] = Vec4f([n as f32, length as f32, 0.0, 0.0]);
            for form in cut.forms() {
                put(form, &mut n);
            }
            count += 1;
        }
        (
            Vec4f([region.len() as f32, count as f32, 0.0, 0.0]),
            cuts,
            forms,
        )
    }

    /// Whether it drops anything: a region or a cut.
    pub fn clips(&self) -> bool {
        !self.region.is_whole() || !self.cuts.is_empty()
    }
}

/// The views of this frame, as a resource: the frame's own first, then those seen through
/// portals, every view after its parent. Written by the render graph each frame.
#[derive(Resource)]
pub struct PortalViews<G: Geometry>(pub Vec<PortalView<G>>);

impl<G: Geometry> Default for PortalViews<G> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<G: Geometry> Clone for PortalViews<G> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<G: Geometry> std::fmt::Debug for PortalViews<G> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("PortalViews").field(&self.0).finish()
    }
}

/// A mouth to look through, with what its portal says of what is seen through it.
pub(crate) struct Opening<G: Geometry> {
    pub mouth: Mouth<G>,
    pub shadows: bool,
    pub far: Option<Real>,
}

/// The camera a tree is built for: its eye's field of view, near and far, and the image's size.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Lens {
    pub fov_y: Real,
    pub far: Real,
    pub size: [u32; 2],
}

/// A mouth seen from a view, ready to become a child view.
struct Seen<G: Geometry> {
    opening: usize,
    region: Region,
    /// Its corners, embedded in the view's eye frame, unclipped.
    corners: [Vector4<Real>; 4],
    part: [Real; 4],
    pixels: Real,
    _geometry: std::marker::PhantomData<fn() -> G>,
}

/// The tree of views seen from `eye` in `realm` through `openings`, to `budget`.
pub(crate) fn portal_views<G: GpuGeometry>(
    eye: &G::Isometry,
    realm: Realm,
    lens: Lens,
    openings: &[Opening<G>],
    budget: &PortalBudget,
) -> Vec<PortalView<G>> {
    let mut views = vec![PortalView {
        eye: *eye,
        realm,
        depth: 0,
        parent: None,
        region: Region::default(),
        cuts: Vec::new(),
        part: [-1.0, -1.0, 1.0, 1.0],
        pixels: Real::from(lens.size[0]) * Real::from(lens.size[1]),
        shadows: true,
        far: None,
    }];
    // What cuts each view because a nearer mouth hides the one it is seen through.
    let mut hidden: Vec<Vec<Region>> = vec![Vec::new()];
    // The corners of the mouth each view is seen through, in its parent's eye frame (the same
    // numbers in its own).
    let mut corners: Vec<Option<[Vector4<Real>; 4]>> = vec![None];
    let mut pending = vec![0];
    while views.len() < budget.views.max(1) {
        // The largest view not yet looked through.
        let Some(at) = (0..pending.len()).max_by(|&a, &b| {
            views[pending[a]]
                .pixels
                .total_cmp(&views[pending[b]].pixels)
        }) else {
            break;
        };
        let parent = pending.swap_remove(at);
        if views[parent].depth >= budget.depth {
            continue;
        }
        let mut seen: Vec<Seen<G>> = openings
            .iter()
            .enumerate()
            .filter(|(_, opening)| opening.mouth.realm == views[parent].realm)
            .filter_map(|(k, opening)| see(&views[parent], k, &opening.mouth, lens, budget))
            .collect();
        seen.sort_by(|a, b| b.pixels.total_cmp(&a.pixels));
        let first = views.len();
        for mouth in seen {
            if views.len() >= budget.views {
                break;
            }
            let opening = &openings[mouth.opening];
            let child = views.len();
            // Siblings either side of the other's plane: the nearer one hides the other where
            // they overlap.
            let mut cut_by = hidden[parent].clone();
            for theirs in views[first..child].iter().map(|view| &view.region) {
                if let Some(plane) = &theirs.plane
                    && mouth.corners.iter().all(|c| plane.dot(c) >= 0.0)
                {
                    cut_by.push(theirs.clone());
                }
            }
            for sibling in first..child {
                if let (Some(plane), Some(theirs)) = (&mouth.region.plane, &corners[sibling])
                    && theirs.iter().all(|c| plane.dot(c) >= 0.0)
                {
                    views[sibling].cuts.push(mouth.region.clone());
                    hidden[sibling].push(mouth.region.clone());
                }
            }
            views[parent].cuts.push(mouth.region.clone());
            views.push(PortalView {
                eye: opening.mouth.carry.compose(&views[parent].eye),
                realm: opening.mouth.into,
                depth: views[parent].depth + 1,
                parent: Some(parent),
                region: mouth.region,
                cuts: cut_by.clone(),
                part: mouth.part,
                pixels: mouth.pixels,
                shadows: opening.shadows,
                far: opening.far,
            });
            hidden.push(cut_by);
            corners.push(Some(mouth.corners));
            pending.push(child);
        }
    }
    views
}

/// `mouth` seen from `view`: its polygon clipped by the eye's plane and the view's cone, the
/// region through it, the part of the image it covers; `None` when it is behind, faced from
/// behind, beyond the far surface or too small.
fn see<G: GpuGeometry>(
    view: &PortalView<G>,
    opening: usize,
    mouth: &Mouth<G>,
    lens: Lens,
    budget: &PortalBudget,
) -> Option<Seen<G>> {
    // Nearer the eye's plane than this, a polygon is cut off.
    const AHEAD: Real = 1e-4;
    let to_eye = view.eye.inverse();
    let frame = to_eye.compose(&mouth.frame);
    let origin = G::embed_point(&G::origin());
    let front = plane_form::<G>(&frame);
    // Faced from its front.
    if front.dot(&origin) <= 0.0 {
        return None;
    }
    let far = view.far.unwrap_or(lens.far);
    let reach = G::distance(&G::origin(), &G::apply(&frame, &G::origin()))
        - (mouth.half[0].hypot(mouth.half[1]));
    if reach > far {
        return None;
    }
    let corners = mouth
        .corners()
        .map(|corner| G::embed_point(&G::apply(&to_eye, &corner)));
    // `z ≤ −AHEAD·w`: in front of the eye (it looks along −z).
    let ahead = Vector4::new(0.0, 0.0, -1.0, -AHEAD);
    let mut polygon = clip(&corners, &ahead);
    // Within the view's region: inside its cone, beyond the mouth it is seen through.
    for form in view.region.forms() {
        polygon = clip(&polygon, form);
    }
    if polygon.len() < 3 {
        return None;
    }
    let tan_y = (lens.fov_y / 2.0).tan();
    let aspect = Real::from(lens.size[0]) / Real::from(lens.size[1]);
    let tan_x = tan_y * aspect;
    let ndc: Vec<[Real; 2]> = polygon
        .iter()
        .map(|p| [p.x / -p.z / tan_x, p.y / -p.z / tan_y])
        .collect();
    let mut part = [
        Real::INFINITY,
        Real::INFINITY,
        Real::NEG_INFINITY,
        Real::NEG_INFINITY,
    ];
    for [x, y] in &ndc {
        part = [
            part[0].min(*x),
            part[1].min(*y),
            part[2].max(*x),
            part[3].max(*y),
        ];
    }
    let part = part.map(|t| t.clamp(-1.0, 1.0));
    if part[0] >= part[2] || part[1] >= part[3] {
        return None;
    }
    // Its area on the image (the shoelace), in pixels, within the image.
    let (w, h) = (Real::from(lens.size[0]), Real::from(lens.size[1]));
    let shoelace: Real = (0..ndc.len())
        .map(|k| {
            let ([x0, y0], [x1, y1]) = (ndc[k], ndc[(k + 1) % ndc.len()]);
            x0 * y1 - x1 * y0
        })
        .sum();
    let whole = (part[2] - part[0]) * (part[3] - part[1]);
    let pixels = (shoelace.abs() / 2.0).min(whole) * w * h / 4.0;
    if pixels < budget.min_pixels {
        return None;
    }
    let middle: Vector4<Real> = polygon.iter().sum();
    let sides = (0..polygon.len())
        .filter_map(|k| {
            let side = form_through(&origin, &polygon[k], &polygon[(k + 1) % polygon.len()]);
            (side.amax() > 1e-12).then(|| if side.dot(&middle) < 0.0 { -side } else { side })
        })
        .collect();
    Some(Seen {
        opening,
        region: Region {
            sides,
            plane: Some(-front),
        },
        corners,
        part,
        pixels,
        _geometry: std::marker::PhantomData,
    })
}

/// The part of the convex polygon `polygon` (embedded points, round it) where `form` is not
/// negative (Sutherland–Hodgman): in the projective models every geometry is embedded in, a
/// geodesic segment is straight, so where it meets a plane is a combination of its ends.
fn clip(polygon: &[Vector4<Real>], form: &Vector4<Real>) -> Vec<Vector4<Real>> {
    let mut kept = Vec::with_capacity(polygon.len() + 1);
    for (k, a) in polygon.iter().enumerate() {
        let b = &polygon[(k + 1) % polygon.len()];
        let (fa, fb) = (form.dot(a), form.dot(b));
        if fa >= 0.0 {
            kept.push(*a);
        }
        if (fa >= 0.0) != (fb >= 0.0) {
            kept.push(a * (fb / (fb - fa)) - b * (fa / (fb - fa)));
        }
    }
    kept
}

/// The light, the sky and the fog of the realms that have their own, as a resource: every view
/// of a realm here (the frame's own too, while its eye stands in it) is lit by its [`Lighting`],
/// every other by the frame's.
#[derive(Resource, Clone, Debug, Default)]
pub struct RealmLighting(pub HashMap<Realm, Lighting>);

impl RealmLighting {
    /// What `realm` is lit by: its own, or `frame`.
    pub fn of<'a>(&'a self, realm: Realm, frame: &'a Lighting) -> &'a Lighting {
        self.0.get(&realm).unwrap_or(frame)
    }
}

/// Every opening of every [`Portal`] in the world.
pub(crate) fn openings<G: GpuGeometry>(world: &mut World) -> Vec<Opening<G>> {
    let Some(mut query) = world.try_query::<&Portal<G>>() else {
        return Vec::new();
    };
    query
        .iter(world)
        .flat_map(|portal| {
            portal.mouths().map(|mouth| Opening {
                mouth,
                shadows: portal.shadows,
                far: portal.far,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use fk_geometry_euclidean::E3;
    use fk_math::nalgebra::Vector3;
    use fk_scene::Portal;

    use super::*;

    fn shift(x: Real, y: Real, z: Real) -> <E3 as Geometry>::Isometry {
        <E3 as Geometry>::Isometry::exp(&E3::transvection(&Vector3::new(x, y, z)))
    }

    fn lens() -> Lens {
        Lens {
            fov_y: 1.2,
            far: 100.0,
            size: [800, 600],
        }
    }

    fn openings(portals: &[Portal<E3>]) -> Vec<Opening<E3>> {
        portals
            .iter()
            .flat_map(|portal| {
                portal.mouths().map(|mouth| Opening {
                    mouth,
                    shadows: portal.shadows,
                    far: portal.far,
                })
            })
            .collect()
    }

    fn embedded(x: Real, y: Real, z: Real) -> Vector4<Real> {
        Vector4::new(x, y, z, 1.0)
    }

    #[test]
    fn a_door_ahead_is_a_view_whose_region_is_beyond_it_and_cut_from_the_frames() {
        // A door 1 by 1 four metres ahead, facing the eye, into a room realm 50 m off.
        let door = Portal::between(
            Realm::MAIN,
            shift(0.0, 0.0, -4.0),
            Realm(1),
            shift(50.0, 0.0, 0.0),
            [1.0, 1.0],
        );
        let views = portal_views::<E3>(
            &shift(0.0, 0.0, 0.0),
            Realm::MAIN,
            lens(),
            &openings(&[door]),
            &PortalBudget::default(),
        );
        assert_eq!(views.len(), 2);
        let room = &views[1];
        assert_eq!(room.realm, Realm(1));
        // Through the middle of the door, beyond it: the room's; before it, the frame's.
        assert!(room.region.holds(&embedded(0.0, 0.0, -6.0)));
        assert!(!room.region.holds(&embedded(0.0, 0.0, -3.0)));
        // Beside the door, beyond its plane: neither the room's nor cut from the frame's.
        assert!(!room.region.holds(&embedded(3.0, 0.0, -6.0)));
        assert_eq!(views[0].cuts.len(), 1);
        assert!(views[0].cuts[0].holds(&embedded(0.0, 0.0, -6.0)));
        // The room's eye stands where the door carries the frame's: 50 m along x, as far in
        // front of the way in as it stands in front of the door.
        assert!(
            (E3::apply(&room.eye, &E3::origin()).coords - Vector3::new(50.0, 0.0, 4.0)).norm()
                < 1e-9
        );
    }

    #[test]
    fn a_door_behind_or_faced_from_behind_is_not_looked_through() {
        let behind = Portal::between(
            Realm::MAIN,
            shift(0.0, 0.0, 4.0),
            Realm(1),
            shift(50.0, 0.0, 0.0),
            [1.0, 1.0],
        )
        .one_way();
        let views = portal_views::<E3>(
            &shift(0.0, 0.0, 0.0),
            Realm::MAIN,
            lens(),
            &openings(&[behind]),
            &PortalBudget::default(),
        );
        assert_eq!(views.len(), 1);
        // Its back, seen from its −z side, opens nothing while it is one way.
        let back = Portal::between(
            Realm::MAIN,
            shift(0.0, 0.0, -4.0).compose(&<E3 as Geometry>::Isometry::exp(
                &(E3::rotation(&Vector3::z(), &Vector3::x()) * std::f64::consts::PI),
            )),
            Realm(1),
            shift(50.0, 0.0, 0.0),
            [1.0, 1.0],
        )
        .one_way();
        let views = portal_views::<E3>(
            &shift(0.0, 0.0, 0.0),
            Realm::MAIN,
            lens(),
            &openings(&[back]),
            &PortalBudget::default(),
        );
        assert_eq!(views.len(), 1);
    }

    #[test]
    fn a_door_seen_through_itself_nests_to_the_budgets_depth() {
        // An endless corridor: a door 2 m ahead that comes out 2 m behind the eye, the same
        // realm. Through it the same door is seen again, 4 m further, smaller, and again.
        let door = Portal::between(
            Realm::MAIN,
            shift(0.0, 0.0, -2.0),
            Realm::MAIN,
            shift(0.0, 0.0, 2.0),
            [1.0, 1.0],
        )
        .one_way();
        let budget = PortalBudget::default();
        let views = portal_views::<E3>(
            &shift(0.0, 0.0, 0.0),
            Realm::MAIN,
            lens(),
            &openings(&[door]),
            &budget,
        );
        assert_eq!(views.len(), budget.depth + 1);
        let depths: Vec<_> = views.iter().map(|view| view.depth).collect();
        assert_eq!(depths, vec![0, 1, 2, 3, 4]);
        // Each deeper view is kept beyond the door as its parent saw it, 4 m further each time:
        // the region of depth 4 starts 14 m ahead.
        let deepest = &views[4];
        assert!(deepest.region.holds(&embedded(0.0, 0.0, -14.5)));
        assert!(!deepest.region.holds(&embedded(0.0, 0.0, -13.5)));
        // And it is the frame's eye carried 16 m on.
        let at = E3::apply(&deepest.eye, &E3::origin()).coords;
        assert!((at - Vector3::new(0.0, 0.0, 16.0)).norm() < 1e-9, "{at}");
        // Every view but the deepest has its child cut from it.
        for view in &views[..4] {
            assert_eq!(view.cuts.len(), 1);
        }
        assert!(views[4].cuts.is_empty());
    }

    #[test]
    fn the_view_block_holds_the_region_then_the_cuts() {
        let door = Portal::between(
            Realm::MAIN,
            shift(0.0, 0.0, -4.0),
            Realm(1),
            shift(50.0, 0.0, 0.0),
            [1.0, 1.0],
        );
        let views = portal_views::<E3>(
            &shift(0.0, 0.0, 0.0),
            Realm::MAIN,
            lens(),
            &openings(&[door]),
            &PortalBudget::default(),
        );
        let (counts, cuts, _) = views[0].block();
        assert_eq!(counts.0[..2], [0.0, 1.0]);
        assert_eq!(cuts[0].0[..2], [0.0, 5.0]);
        let (counts, _, forms) = views[1].block();
        assert_eq!(counts.0[..2], [5.0, 0.0]);
        // The region's plane, last, is positive beyond the door.
        let plane = Vector4::from(forms[4].0.map(Real::from));
        assert!(plane.dot(&embedded(0.0, 0.0, -6.0)) > 0.0);
    }
}
