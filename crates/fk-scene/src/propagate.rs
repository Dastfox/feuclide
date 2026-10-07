use bevy_ecs::prelude::*;
use fk_geometry::{Geometry, GroupElement};

use crate::{ChartTag, GlobalPose, Pose};

type Node<'a, G> = (&'a Pose<G>, Option<&'a ChartTag>, Option<&'a Children>);

/// Writes every [`GlobalPose`] from the [`Pose`]s and [`ChartTag`]s down the hierarchy.
///
/// A subtree under an entity without a `Pose` is left alone.
pub fn propagate_poses<G: Geometry>(
    roots: Query<(Entity, Node<G>), Without<ChildOf>>,
    nodes: Query<Node<G>, With<ChildOf>>,
    mut globals: Query<&mut GlobalPose<G>>,
    mut stack: Local<Vec<(Entity, G::Isometry, ChartTag)>>,
) {
    for (entity, (pose, chart, children)) in &roots {
        let global = GlobalPose::<G> {
            isometry: pose.0,
            chart: chart.cloned().unwrap_or_default(),
        };
        push_children(&mut stack, children, &global);
        write(&mut globals, entity, global);
        while let Some((entity, parent, parent_chart)) = stack.pop() {
            let Ok((pose, chart, children)) = nodes.get(entity) else {
                continue;
            };
            let global = GlobalPose::<G> {
                isometry: parent.compose(&pose.0),
                chart: match chart {
                    Some(chart) => parent_chart.then(chart),
                    None => parent_chart,
                },
            };
            push_children(&mut stack, children, &global);
            write(&mut globals, entity, global);
        }
    }
}

fn push_children<G: Geometry>(
    stack: &mut Vec<(Entity, G::Isometry, ChartTag)>,
    children: Option<&Children>,
    parent: &GlobalPose<G>,
) {
    for &child in children.into_iter().flatten() {
        stack.push((child, parent.isometry, parent.chart.clone()));
    }
}

fn write<G: Geometry>(
    globals: &mut Query<&mut GlobalPose<G>>,
    entity: Entity,
    pose: GlobalPose<G>,
) {
    if let Ok(mut global) = globals.get_mut(entity) {
        *global = pose;
    }
}
