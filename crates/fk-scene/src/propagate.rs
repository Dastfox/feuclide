use bevy_ecs::prelude::*;
use fk_geometry::{Geometry, GroupElement};

use crate::{ChartTag, GlobalPose, GlobalRealm, Pose, Realm};

type Node<'a, G> = (
    &'a Pose<G>,
    Option<&'a ChartTag>,
    Option<&'a Realm>,
    Option<&'a Children>,
);

/// An entity to visit, under its parent's global pose, chart and realm.
type Visit<G> = (Entity, <G as Geometry>::Isometry, ChartTag, Realm);

/// Writes every [`GlobalPose`] from the [`Pose`]s and [`ChartTag`]s down the hierarchy, and
/// every [`GlobalRealm`] from the [`Realm`]s.
///
/// A subtree under an entity without a `Pose` is left alone.
pub fn propagate_poses<G: Geometry>(
    roots: Query<(Entity, Node<G>), Without<ChildOf>>,
    nodes: Query<Node<G>, With<ChildOf>>,
    mut globals: Query<(&mut GlobalPose<G>, Option<&mut GlobalRealm>)>,
    mut stack: Local<Vec<Visit<G>>>,
) {
    for (entity, (pose, chart, realm, children)) in &roots {
        let global = GlobalPose::<G> {
            isometry: pose.0,
            chart: chart.cloned().unwrap_or_default(),
        };
        let realm = realm.copied().unwrap_or_default();
        push_children(&mut stack, children, &global, realm);
        write(&mut globals, entity, global, realm);
        while let Some((entity, parent, parent_chart, parent_realm)) = stack.pop() {
            let Ok((pose, chart, realm, children)) = nodes.get(entity) else {
                continue;
            };
            let global = GlobalPose::<G> {
                isometry: parent.compose(&pose.0),
                chart: match chart {
                    Some(chart) => parent_chart.then(chart),
                    None => parent_chart,
                },
            };
            let realm = realm.copied().unwrap_or(parent_realm);
            push_children(&mut stack, children, &global, realm);
            write(&mut globals, entity, global, realm);
        }
    }
}

fn push_children<G: Geometry>(
    stack: &mut Vec<Visit<G>>,
    children: Option<&Children>,
    parent: &GlobalPose<G>,
    realm: Realm,
) {
    for &child in children.into_iter().flatten() {
        stack.push((child, parent.isometry, parent.chart.clone(), realm));
    }
}

fn write<G: Geometry>(
    globals: &mut Query<(&mut GlobalPose<G>, Option<&mut GlobalRealm>)>,
    entity: Entity,
    pose: GlobalPose<G>,
    realm: Realm,
) {
    if let Ok((mut global, in_realm)) = globals.get_mut(entity) {
        *global = pose;
        if let Some(mut in_realm) = in_realm
            && in_realm.0 != realm
        {
            in_realm.0 = realm;
        }
    }
}
