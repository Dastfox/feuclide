//! The geometry property battery, run on S² and S³, with points and steps within 1.2 of the
//! origin: well inside the injectivity radius π, so triangles stay away from the antipodes.

use fk_geometry::battery::BatteryConfig;
use fk_geometry_spherical::{S2, S3};

fn sphere_config() -> BatteryConfig {
    BatteryConfig {
        radius: 1.2,
        ..BatteryConfig::default()
    }
}

fk_geometry::geometry_battery!(s2, S2, sphere_config());
fk_geometry::geometry_battery!(s3, S3, sphere_config());
