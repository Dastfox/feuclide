//! A first-person body on a flat ground: a [`Character`] that walks, and where its eyes look
//! up or down (the counterpart of a first-person controller crate for Bevy).

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use fk_geometry_euclidean::{E3, Se3};
use fk_math::Real;
use fk_math::nalgebra::{Vector2, Vector3};
use fk_physics::{Character, CharacterShape};

/// A first-person body: its [`Character`] (feet, yaw, velocity) and its eyes' pitch.
#[derive(Debug)]
pub struct FirstPerson {
    /// The body that walks: its pose is its feet, turned by its yaw.
    pub body: Character<E3>,
    /// How far the eyes look up, radians (down negative).
    pub pitch: Real,
}

impl FirstPerson {
    /// Standing at `feet`, of `shape`, looking level.
    pub fn standing(feet: Se3, shape: CharacterShape) -> Self {
        Self {
            body: Character::new(feet, shape),
            pitch: 0.0,
        }
    }

    /// Where its feet are on the ground (x, z), and its yaw (0 along −z, positive to the
    /// right).
    pub fn on_ground(&self) -> (Vector2<Real>, Real) {
        let feet = self.body.feet();
        let ahead = self.body.pose.rotation * -Vector3::z();
        (Vector2::new(feet.x, feet.z), ahead.x.atan2(-ahead.z))
    }

    /// Flies `distance` along `input` (x to the right, y along its gaze, pitch and all) and
    /// `rise` straight up, through everything, and stops any fall: a free camera, noclip.
    pub fn fly(&mut self, input: &Vector2<Real>, rise: Real, distance: Real) {
        let rotation = self.body.pose.rotation;
        let ahead = rotation * Vector3::new(0.0, self.pitch.sin(), -self.pitch.cos());
        let right = rotation * Vector3::x();
        let way = right * input.x + ahead * input.y + Vector3::y() * rise;
        self.body.pose.translation += way * distance;
        self.body.velocity = Vector3::zeros();
        self.body.on_ground = false;
    }

    /// Turns its head towards `point` from its eye `eye` above its feet, `share` of the way
    /// there: the yaw turns the body, the pitch the eyes, short of straight up or down.
    pub fn look_towards(&mut self, point: &Vector3<Real>, eye: Real, share: Real) {
        let feet = self.body.pose.translation;
        let to = point - (feet + Vector3::new(0.0, eye, 0.0));
        let ahead = self.body.pose.rotation * -Vector3::z();
        let yaw = ahead.x.atan2(-ahead.z);
        let wanted = to.x.atan2(-to.z);
        let turn = (wanted - yaw + PI).rem_euclid(TAU) - PI;
        self.body.turn(turn * share);
        let limit = FRAC_PI_2 - 0.01;
        let pitch =
            to.y.atan2(Vector2::new(to.x, to.z).norm())
                .clamp(-limit, limit);
        self.pitch += (pitch - self.pitch) * share;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_looks_where_it_is_told_and_flies_where_it_looks() {
        let shape = CharacterShape {
            radius: 0.3,
            height: 1.8,
        };
        let mut body = FirstPerson::standing(crate::kit::translation(Vector3::zeros()), shape);
        body.look_towards(&Vector3::new(10.0, 1.7, 0.0), 1.7, 1.0);
        let (at, yaw) = body.on_ground();
        assert!(at.norm() < 1e-12);
        assert!((yaw - FRAC_PI_2).abs() < 1e-9, "turned to the right: {yaw}");
        assert!(body.pitch.abs() < 1e-9, "level");
        body.fly(&Vector2::new(0.0, 1.0), 0.0, 2.0);
        let (at, _) = body.on_ground();
        assert!((at - Vector2::new(2.0, 0.0)).norm() < 1e-9, "{at}");
    }
}
