//! Persistent CPU render-world cache for world-mesh draw preparation.
//!
//! The scene layer remains the authoritative host-world mirror. This cache lives in the backend
//! side of world-mesh draw prep and stores only renderer-facing expansion products that are
//! expensive to rediscover every frame.

use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;

use glam::{Mat4, Vec3};

use crate::camera::view_matrix_from_render_transform;
use crate::gpu_pools::MeshPool;
use crate::scene::{
    RenderLightRow, RenderSpaceId, ResolvedLight, SceneApplyReport, SceneCacheFlushReport,
    SceneCoordinator,
};
use crate::shared::RenderingContext;

use super::prepared_renderables::{
    FramePreparedDraw, FramePreparedRenderables, FramePreparedSpace, estimated_draw_count,
    expand_space_into_aggressive,
};

/// Local axis for light propagation before world transform.
const LOCAL_LIGHT_PROPAGATION: Vec3 = Vec3::new(0.0, 0.0, 1.0);

/// Persistent renderer-facing cache of expanded world-mesh renderables.
pub struct RenderWorld {
    spaces: HashMap<RenderSpaceId, RenderWorldSpace>,
    dirty_spaces: HashSet<RenderSpaceId>,
    full_rebuild_requested: bool,
    mesh_pool_generation: u64,
    prepared: FramePreparedRenderables,
}

#[derive(Default)]
struct RenderWorldSpace {
    active: bool,
    space: Option<FramePreparedSpace>,
    draws: Vec<FramePreparedDraw>,
    lights: Vec<RenderLightRow>,
    chunk_scratch: Vec<Vec<FramePreparedDraw>>,
}

struct DirtyRenderWorldSpace {
    id: RenderSpaceId,
    cached: RenderWorldSpace,
}

fn refresh_render_world_space(
    cached: &mut RenderWorldSpace,
    scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    id: RenderSpaceId,
) {
    profiling::scope!("mesh::render_world::refresh_space");
    cached.draws.clear();
    cached.lights.clear();
    cached.space = build_prepared_space_from_scene(scene, id, render_context);
    scene.render_light_rows_for_space_into(id, &mut cached.lights);
    if !cached.active {
        return;
    }
    let Some(space_meta) = cached.space.as_ref() else {
        return;
    };
    {
        profiling::scope!("mesh::render_world::reserve_space_draws");
        let estimate = estimated_draw_count(scene, id);
        if estimate > cached.draws.capacity() {
            cached.draws.reserve(estimate - cached.draws.capacity());
        }
    }
    expand_space_into_aggressive(
        &mut cached.draws,
        &mut cached.chunk_scratch,
        scene,
        mesh_pool,
        render_context,
        id,
        space_meta,
    );
}

/// Builds renderer-facing space metadata from the scene mirror.
pub(in crate::world_mesh::draw_prep) fn build_prepared_space_from_scene(
    scene: &SceneCoordinator,
    id: RenderSpaceId,
    render_context: RenderingContext,
) -> Option<FramePreparedSpace> {
    let space = scene.space(id)?;
    let node_count = space.local_transforms().len();
    let mut context_world_matrices = Vec::with_capacity(node_count);
    let mut degenerate_scales = Vec::with_capacity(node_count);
    let mut overlay_layer_model_matrices = Vec::with_capacity(node_count);
    for node in 0..node_count {
        context_world_matrices.push(scene.world_matrix_for_context(id, node, render_context));
        degenerate_scales.push(scene.transform_has_degenerate_scale_for_context(
            id,
            node,
            render_context,
        ));
        overlay_layer_model_matrices.push(scene.overlay_layer_model_matrix_for_context(
            id,
            node,
            render_context,
        ));
    }
    Some(FramePreparedSpace {
        is_overlay_space: space.is_overlay(),
        root_transform: *space.root_transform(),
        view_matrix: view_matrix_from_render_transform(space.view_transform()),
        node_parents: space.node_parents().to_vec(),
        context_world_matrices,
        degenerate_scales,
        overlay_layer_model_matrices,
    })
}

impl RenderWorld {
    /// Creates an empty render-world cache.
    pub fn new(render_context: RenderingContext) -> Self {
        Self {
            spaces: HashMap::new(),
            dirty_spaces: HashSet::new(),
            full_rebuild_requested: true,
            mesh_pool_generation: 0,
            prepared: FramePreparedRenderables::empty(render_context),
        }
    }

    /// Marks spaces touched or removed by scene apply as needing render-world maintenance.
    pub fn note_scene_apply_report(&mut self, report: &SceneApplyReport) {
        for &id in &report.changed_spaces {
            self.dirty_spaces.insert(id);
        }
        for &id in &report.removed_spaces {
            self.spaces.remove(&id);
            self.dirty_spaces.remove(&id);
        }
        if !report.removed_spaces.is_empty() {
            self.full_rebuild_requested = true;
        }
    }

    /// Marks spaces whose world matrices changed as needing cached bounds refresh.
    pub fn note_cache_flush_report(&mut self, report: &SceneCacheFlushReport) {
        for &id in &report.flushed_spaces {
            self.dirty_spaces.insert(id);
        }
    }

    /// Returns the prepared draw snapshot for this frame, refreshing dirty cached spaces first.
    pub fn prepare_for_frame(
        &mut self,
        scene: &SceneCoordinator,
        mesh_pool: &MeshPool,
        render_context: RenderingContext,
    ) -> &FramePreparedRenderables {
        profiling::scope!("mesh::render_world::prepare_for_frame");
        let mesh_pool_generation = mesh_pool.mutation_generation();
        let context_changed = self.prepared.render_context() != render_context;
        let mesh_pool_changed = self.mesh_pool_generation != mesh_pool_generation;
        if context_changed || mesh_pool_changed {
            self.full_rebuild_requested = true;
            self.mesh_pool_generation = mesh_pool_generation;
        }

        let full_rebuild = self.full_rebuild_requested;
        if full_rebuild {
            profiling::scope!("mesh::render_world::mark_all_dirty");
            self.mark_all_scene_spaces_dirty(scene);
        }

        let had_dirty = !self.dirty_spaces.is_empty();
        if had_dirty {
            profiling::scope!("mesh::render_world::refresh_dirty_spaces");
            self.refresh_dirty_spaces(scene, mesh_pool, render_context);
        }

        if full_rebuild || had_dirty || context_changed || mesh_pool_changed {
            profiling::scope!("mesh::render_world::rebuild_snapshot");
            self.rebuild_prepared_snapshot(scene, render_context);
        }
        self.full_rebuild_requested = false;
        &self.prepared
    }

    /// Prepared draw snapshot from the most recent [`Self::prepare_for_frame`] call.
    pub(crate) fn prepared(&self) -> &FramePreparedRenderables {
        &self.prepared
    }

    /// Marks every cached space dirty on the next frame.
    pub(crate) fn mark_all_dirty(&mut self) {
        self.full_rebuild_requested = true;
    }

    /// Collects active render spaces that should contribute lights for a view.
    pub(crate) fn collect_light_space_ids(
        &self,
        render_space_filter: Option<RenderSpaceId>,
        out: &mut Vec<RenderSpaceId>,
    ) {
        out.clear();
        if let Some(id) = render_space_filter {
            if self.spaces.get(&id).is_some_and(|space| space.active) {
                out.push(id);
            }
            return;
        }
        out.extend(
            self.prepared
                .active_space_ids()
                .iter()
                .copied()
                .filter(|id| self.spaces.get(id).is_some_and(|space| space.active)),
        );
    }

    /// Appends resolved world-space lights for `space_id`.
    pub(crate) fn resolve_lights_for_space_into(
        &self,
        space_id: RenderSpaceId,
        head_output_transform: Mat4,
        out: &mut Vec<ResolvedLight>,
    ) {
        let Some(space) = self.spaces.get(&space_id).filter(|space| space.active) else {
            return;
        };
        let Some(space_meta) = space.space.as_ref() else {
            return;
        };
        out.reserve(space.lights.len());
        for light in &space.lights {
            let world = space_meta
                .render_context_model_matrix(light.transform_id as i32, head_output_transform)
                .unwrap_or(Mat4::IDENTITY);
            out.push(resolve_light_row(light, world));
        }
    }

    fn mark_all_scene_spaces_dirty(&mut self, scene: &SceneCoordinator) {
        profiling::scope!("mesh::render_world::mark_all_scene_spaces_dirty");
        self.spaces.retain(|id, _| scene.space(*id).is_some());
        for id in scene.render_space_ids() {
            self.dirty_spaces.insert(id);
        }
    }

    fn refresh_dirty_spaces(
        &mut self,
        scene: &SceneCoordinator,
        mesh_pool: &MeshPool,
        render_context: RenderingContext,
    ) {
        profiling::scope!("mesh::render_world::refresh_dirty_spaces_inner");
        let mut dirty_spaces = std::mem::take(&mut self.dirty_spaces);
        let mut work = Vec::with_capacity(dirty_spaces.len());
        {
            profiling::scope!("mesh::render_world::collect_dirty_spaces");
            for id in dirty_spaces.drain() {
                let Some(space) = scene.space(id) else {
                    self.spaces.remove(&id);
                    continue;
                };
                let mut cached = self.spaces.remove(&id).unwrap_or_default();
                cached.active = space.is_active();
                work.push(DirtyRenderWorldSpace { id, cached });
            }
        }

        if work.len() < 2 {
            profiling::scope!("mesh::render_world::refresh_dirty_serial");
            for mut slot in work.drain(..) {
                refresh_render_world_space(
                    &mut slot.cached,
                    scene,
                    mesh_pool,
                    render_context,
                    slot.id,
                );
                self.spaces.insert(slot.id, slot.cached);
            }
        } else {
            {
                profiling::scope!("mesh::render_world::refresh_dirty_parallel");
                work.par_iter_mut().for_each(|slot| {
                    profiling::scope!("mesh::render_world::dirty_space_worker");
                    refresh_render_world_space(
                        &mut slot.cached,
                        scene,
                        mesh_pool,
                        render_context,
                        slot.id,
                    );
                });
            }
            {
                profiling::scope!("mesh::render_world::reinsert_dirty_spaces");
                for slot in work.drain(..) {
                    self.spaces.insert(slot.id, slot.cached);
                }
            }
        }
        self.dirty_spaces = dirty_spaces;
    }

    fn rebuild_prepared_snapshot(
        &mut self,
        scene: &SceneCoordinator,
        render_context: RenderingContext,
    ) {
        profiling::scope!("mesh::render_world::rebuild_prepared_snapshot");
        self.prepared.rebuild_from_cached_spaces(
            render_context,
            scene.render_space_ids().filter_map(|id| {
                self.spaces.get(&id).filter(|s| s.active).and_then(|s| {
                    s.space
                        .as_ref()
                        .map(|space| (id, space, s.draws.as_slice()))
                })
            }),
        );
    }
}

/// Resolves one cached light row through `world`.
fn resolve_light_row(light: &RenderLightRow, world: Mat4) -> ResolvedLight {
    let point = light.data.point;
    let p = Vec3::new(point.x, point.y, point.z);
    let world_position = world.transform_point3(p);

    let world_direction = (world.to_scale_rotation_translation().1 * light.data.orientation)
        * LOCAL_LIGHT_PROPAGATION;
    let world_direction = if world_direction.length_squared() > 1e-10 {
        world_direction.normalize()
    } else {
        LOCAL_LIGHT_PROPAGATION
    };

    let color = light.data.color;
    let color = Vec3::new(color.x, color.y, color.z);
    let range = if light.state.global_unique_id >= 0 {
        let (scale, _, _) = world.to_scale_rotation_translation();
        let uniform_scale = (scale.x + scale.y + scale.z) / 3.0;
        light.data.range * uniform_scale
    } else {
        light.data.range
    };

    ResolvedLight {
        world_position,
        world_direction,
        color,
        intensity: light.data.intensity,
        range,
        spot_angle: light.data.angle,
        light_type: light.state.light_type,
        shadow_type: light.state.shadow_type,
        shadow_strength: light.state.shadow_strength,
        shadow_near_plane: light.state.shadow_near_plane,
        shadow_bias: light.state.shadow_bias,
        shadow_normal_bias: light.state.shadow_normal_bias,
    }
}

impl Default for RenderWorld {
    fn default() -> Self {
        Self::new(RenderingContext::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::SceneCacheFlushReport;
    use crate::shared::RenderTransform;

    #[test]
    fn apply_report_marks_changed_spaces_dirty() {
        let mut world = RenderWorld::default();
        world.note_scene_apply_report(&SceneApplyReport {
            frame_index: 7,
            submitted_spaces: vec![RenderSpaceId(1)],
            changed_spaces: vec![RenderSpaceId(1), RenderSpaceId(2)],
            removed_spaces: Vec::new(),
        });

        assert!(world.dirty_spaces.contains(&RenderSpaceId(1)));
        assert!(world.dirty_spaces.contains(&RenderSpaceId(2)));
    }

    #[test]
    fn removed_space_evicts_cached_rows_and_requests_snapshot_rebuild() {
        let mut world = RenderWorld::default();
        world.spaces.insert(
            RenderSpaceId(3),
            RenderWorldSpace {
                active: true,
                draws: Vec::new(),
                ..Default::default()
            },
        );
        world.dirty_spaces.insert(RenderSpaceId(3));

        world.note_scene_apply_report(&SceneApplyReport {
            frame_index: 8,
            submitted_spaces: Vec::new(),
            changed_spaces: Vec::new(),
            removed_spaces: vec![RenderSpaceId(3)],
        });

        assert!(!world.spaces.contains_key(&RenderSpaceId(3)));
        assert!(!world.dirty_spaces.contains(&RenderSpaceId(3)));
        assert!(world.full_rebuild_requested);
    }

    #[test]
    fn cache_flush_marks_bounds_dirty() {
        let mut world = RenderWorld::default();
        world.note_cache_flush_report(&SceneCacheFlushReport {
            flushed_spaces: vec![RenderSpaceId(9)],
        });

        assert!(world.dirty_spaces.contains(&RenderSpaceId(9)));
    }

    #[test]
    fn removed_space_wins_over_changed_space_in_apply_report() {
        let mut world = RenderWorld::default();
        world
            .spaces
            .insert(RenderSpaceId(5), RenderWorldSpace::default());

        world.note_scene_apply_report(&SceneApplyReport {
            frame_index: 9,
            submitted_spaces: vec![RenderSpaceId(5)],
            changed_spaces: vec![RenderSpaceId(5)],
            removed_spaces: vec![RenderSpaceId(5)],
        });

        assert!(!world.spaces.contains_key(&RenderSpaceId(5)));
        assert!(!world.dirty_spaces.contains(&RenderSpaceId(5)));
        assert!(world.full_rebuild_requested);
    }

    #[test]
    fn mark_all_scene_spaces_dirty_retain_only_existing_scene_spaces() {
        let mut scene = SceneCoordinator::new();
        let keep = RenderSpaceId(10);
        scene.test_seed_space_identity_worlds(keep, vec![RenderTransform::default()], vec![-1]);
        let mut world = RenderWorld::default();
        world.spaces.insert(keep, RenderWorldSpace::default());
        world
            .spaces
            .insert(RenderSpaceId(11), RenderWorldSpace::default());

        world.mark_all_scene_spaces_dirty(&scene);

        assert!(world.spaces.contains_key(&keep));
        assert!(!world.spaces.contains_key(&RenderSpaceId(11)));
        assert!(world.dirty_spaces.contains(&keep));
    }

    #[test]
    fn rebuild_prepared_snapshot_skips_inactive_cached_spaces() {
        let mut scene = SceneCoordinator::new();
        let active = RenderSpaceId(20);
        let inactive = RenderSpaceId(21);
        scene.test_seed_space_identity_worlds(active, vec![RenderTransform::default()], vec![-1]);
        scene.test_seed_space_identity_worlds(inactive, vec![RenderTransform::default()], vec![-1]);
        let mut world = RenderWorld::default();
        world.spaces.insert(
            active,
            RenderWorldSpace {
                active: true,
                space: build_prepared_space_from_scene(
                    &scene,
                    active,
                    RenderingContext::RenderToAsset,
                ),
                draws: Vec::new(),
                ..Default::default()
            },
        );
        world.spaces.insert(
            inactive,
            RenderWorldSpace {
                active: false,
                space: build_prepared_space_from_scene(
                    &scene,
                    inactive,
                    RenderingContext::RenderToAsset,
                ),
                draws: Vec::new(),
                ..Default::default()
            },
        );

        world.rebuild_prepared_snapshot(&scene, RenderingContext::RenderToAsset);

        assert_eq!(
            world.prepared.render_context(),
            RenderingContext::RenderToAsset
        );
        assert_eq!(world.prepared.active_space_ids(), &[active]);
        assert!(world.prepared.draws().is_empty());
    }
}
