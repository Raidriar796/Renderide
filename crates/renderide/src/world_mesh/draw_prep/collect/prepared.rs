//! Prepared-renderable draw collection path for world-mesh renderables.

use hashbrown::HashMap;

use glam::{Mat4, Vec3};

use crate::materials::RasterFrontFace;
use crate::scene::RenderSpaceId;

use crate::world_mesh::culling::{CpuCullFailure, mesh_cpu_cull_with_geometry};
use crate::world_mesh::materials::FrameMaterialBatchCache;

use super::super::item::{WorldMeshDrawItem, stacked_material_submesh_topology};
use super::super::prepared_renderables::{
    FramePreparedDraw, FramePreparedRun, FramePreparedSpace, mesh_cull_geometry_from_prepared,
    overlay_space_root_matrix,
};
use super::DrawCollectionContext;
use super::candidate::{DrawCandidate, evaluate_draw_candidate};
use super::world_matrix::front_face_for_draw_matrices;

/// Returns true when two prepared slot entries came from the same source renderer.
#[inline]
pub(in crate::world_mesh::draw_prep) fn prepared_draws_share_renderer(
    a: &FramePreparedDraw,
    b: &FramePreparedDraw,
) -> bool {
    a.space_id == b.space_id
        && a.renderable_index == b.renderable_index
        && a.instance_id == b.instance_id
        && a.node_id == b.node_id
        && a.mesh_asset_id == b.mesh_asset_id
        && a.is_overlay == b.is_overlay
        && a.sorting_order == b.sorting_order
        && a.skinned == b.skinned
        && a.world_space_deformed == b.world_space_deformed
        && a.blendshape_deformed == b.blendshape_deformed
        && a.tangent_blendshape_deform_active == b.tangent_blendshape_deform_active
        && a.space_is_overlay == b.space_is_overlay
        && a.context_world_matrix == b.context_world_matrix
        && a.overlay_layer_model_matrix == b.overlay_layer_model_matrix
        && a.skinned_root_world_matrix == b.skinned_root_world_matrix
}

/// Per-renderer view-local state shared by every material slot in a prepared run.
#[derive(Clone, Copy)]
struct PreparedRunViewState {
    /// Rigid model matrix reused by all emitted slot draws.
    rigid_world_matrix: Option<Mat4>,
    /// World-space object AABB reused by all emitted slot draws for transparent sorting and probes.
    world_aabb: Option<(Vec3, Vec3)>,
    /// Raster front-face winding selected from [`Self::rigid_world_matrix`].
    front_face: RasterFrontFace,
    /// Camera distance reused by alpha-blended slot draws.
    alpha_distance_sq: f32,
}

/// Returns whether the renderer run passes the view's optional transform filter.
fn prepared_run_passes_filter(
    first: &FramePreparedDraw,
    space: &FramePreparedSpace,
    ctx: &DrawCollectionContext<'_>,
    filter_masks: &HashMap<RenderSpaceId, Vec<bool>>,
) -> bool {
    let Some(filter) = ctx.transform_filter else {
        return true;
    };
    match filter_masks.get(&first.space_id) {
        Some(mask) => {
            first.node_id >= 0
                && (first.node_id as usize) < mask.len()
                && mask[first.node_id as usize]
        }
        None => filter.passes_node_with_parents(first.node_id, &space.node_parents),
    }
}

/// Builds shared view-local state for one prepared renderer run and reports draw-slot cull stats.
fn prepared_run_view_state(
    run: &[FramePreparedDraw],
    first: &FramePreparedDraw,
    space: &FramePreparedSpace,
    is_overlay: bool,
    mesh: &crate::assets::mesh::GpuMesh,
    ctx: &DrawCollectionContext<'_>,
) -> (Option<PreparedRunViewState>, (usize, usize, usize)) {
    let mut cull_stats = (0usize, 0usize, 0usize);
    let mut rigid_world_matrix = None;
    let mut world_aabb = None;
    let mut deformed_front_face_world_matrix = None;
    let needs_geometry =
        ctx.reflection_probes.is_some() || ctx.culling.is_some() || first.world_space_deformed;
    let geometry = needs_geometry.then(|| {
        // Reuse the per-renderer geometry that `FramePreparedRenderables::build_for_frame` already
        // computed for non-overlay spaces. Overlay spaces (geometry depends on the per-view
        // `head_output_transform`) keep recomputing per-view via the fallback path below.
        first
            .cull_geometry
            .unwrap_or_else(|| prepared_geometry_for_view(first, space, mesh, ctx))
    });
    if let Some(geom) = geometry {
        world_aabb = geom.world_aabb;
        deformed_front_face_world_matrix = geom.front_face_world_matrix;
        if let Some(c) = ctx.culling {
            cull_stats.0 += run.len();
            let view = c
                .host_camera
                .explicit_world_to_view()
                .unwrap_or(space.view_matrix);
            match mesh_cpu_cull_with_geometry(geom, view, first.space_id, is_overlay, c, None) {
                Err(CpuCullFailure::Frustum | CpuCullFailure::UiRectMask) => {
                    cull_stats.1 += run.len();
                    return (None, cull_stats);
                }
                Err(CpuCullFailure::HiZ) => {
                    cull_stats.2 += run.len();
                    return (None, cull_stats);
                }
                Ok(m) => {
                    rigid_world_matrix = m;
                }
            }
        } else if rigid_world_matrix.is_none() {
            rigid_world_matrix = geom.rigid_world_matrix;
        }
    }
    if is_overlay && !first.world_space_deformed {
        rigid_world_matrix =
            space.local_vertex_model_matrix(first.node_id, true, ctx.head_output_transform);
    } else if !first.world_space_deformed && rigid_world_matrix.is_none() {
        rigid_world_matrix =
            space.local_vertex_model_matrix(first.node_id, false, ctx.head_output_transform);
    }
    let front_face = front_face_for_draw_matrices(
        first.world_space_deformed,
        rigid_world_matrix,
        deformed_front_face_world_matrix,
    );
    let alpha_distance_sq = rigid_world_matrix.map_or(0.0, |m| {
        (m.col(3).truncate() - ctx.view_origin_world).length_squared()
    });
    (
        Some(PreparedRunViewState {
            rigid_world_matrix,
            world_aabb,
            front_face,
            alpha_distance_sq,
        }),
        cull_stats,
    )
}

/// Builds cull geometry from prepared transform rows for view-dependent overlay spaces.
fn prepared_geometry_for_view(
    first: &FramePreparedDraw,
    space: &FramePreparedSpace,
    mesh: &crate::assets::mesh::GpuMesh,
    ctx: &DrawCollectionContext<'_>,
) -> crate::world_mesh::culling::MeshCullGeometry {
    let context_world_matrix = if first.space_is_overlay {
        first
            .context_world_matrix
            .map(|m| overlay_space_root_matrix(space.root_transform, ctx.head_output_transform) * m)
    } else {
        first.context_world_matrix
    };
    let skinned_root_world_matrix = if first.space_is_overlay {
        first
            .skinned_root_world_matrix
            .map(|m| overlay_space_root_matrix(space.root_transform, ctx.head_output_transform) * m)
    } else {
        first.skinned_root_world_matrix
    };
    mesh_cull_geometry_from_prepared(
        mesh,
        first.skinned,
        context_world_matrix,
        skinned_root_world_matrix,
        first.posed_object_bounds.as_ref(),
    )
}

/// Emits one [`WorldMeshDrawItem`] per material slot in a surviving prepared renderer run.
fn append_prepared_run_draws(
    run: &[FramePreparedDraw],
    ctx: &DrawCollectionContext<'_>,
    cache: &FrameMaterialBatchCache,
    mesh: &crate::assets::mesh::GpuMesh,
    is_overlay: bool,
    state: PreparedRunViewState,
    out: &mut Vec<WorldMeshDrawItem>,
) {
    for d in run {
        let primitive_topology =
            stacked_material_submesh_topology(d.slot_index, &mesh.submesh_topologies);
        let candidate = DrawCandidate {
            space_id: d.space_id,
            node_id: d.node_id,
            renderable_index: d.renderable_index,
            instance_id: d.instance_id,
            mesh_asset_id: d.mesh_asset_id,
            slot_index: d.slot_index,
            first_index: d.first_index,
            index_count: d.index_count,
            is_overlay,
            sorting_order: d.sorting_order,
            skinned: d.skinned,
            world_space_deformed: d.world_space_deformed,
            blendshape_deformed: d.blendshape_deformed,
            tangent_blendshape_deform_active: d.tangent_blendshape_deform_active,
            material_asset_id: d.material_asset_id,
            property_block_id: d.property_block_id,
            world_aabb: state.world_aabb,
        };
        if let Some(item) = evaluate_draw_candidate(
            ctx,
            cache,
            candidate,
            state.front_face,
            primitive_topology,
            state.rigid_world_matrix,
            state.alpha_distance_sq,
        ) {
            out.push(item);
        }
    }
}

/// Collects one prepared renderer run after frame-global slot expansion.
fn collect_prepared_renderer_run(
    run: &[FramePreparedDraw],
    ctx: &DrawCollectionContext<'_>,
    cache: &FrameMaterialBatchCache,
    filter_masks: &HashMap<RenderSpaceId, Vec<bool>>,
    out: &mut Vec<WorldMeshDrawItem>,
) -> (usize, usize, usize) {
    let Some(first) = run.first() else {
        return (0, 0, 0);
    };
    if ctx
        .render_space_filter
        .is_some_and(|space_id| first.space_id != space_id)
    {
        return (0, 0, 0);
    }
    let Some(prepared) = ctx.prepared else {
        return (0, 0, 0);
    };
    let Some(space) = prepared.space(first.space_id) else {
        return (0, 0, 0);
    };
    if !prepared_run_passes_filter(first, space, ctx, filter_masks) {
        return (0, 0, 0);
    }
    let is_overlay = first.is_overlay;
    let Some(mesh) = ctx.mesh_pool.get(first.mesh_asset_id) else {
        return (0, 0, 0);
    };
    let (state, cull_stats) = prepared_run_view_state(run, first, space, is_overlay, mesh, ctx);
    if let Some(state) = state {
        append_prepared_run_draws(run, ctx, cache, mesh, is_overlay, state, out);
    }
    cull_stats
}

/// Collects draw items for one chunk of a pre-expanded [`super::FramePreparedRenderables`] list.
///
/// Unlike the scene-walk chunk collector, there is no scene walk: the prepared draws already
/// captured every valid `(renderer x material slot)` tuple plus its frame-global resolution
/// (material override, submesh index range, overlay flag, skin deform flag). This per-view pass
/// only applies filters and per-view CPU culling per renderer, then builds [`WorldMeshDrawItem`]s
/// for each material slot.
pub(super) fn collect_prepared_chunk(
    draws: &[FramePreparedDraw],
    runs: &[FramePreparedRun],
    ctx: &DrawCollectionContext<'_>,
    cache: &FrameMaterialBatchCache,
    filter_masks: &HashMap<RenderSpaceId, Vec<bool>>,
) -> (Vec<WorldMeshDrawItem>, (usize, usize, usize)) {
    profiling::scope!("mesh::collect_prepared::chunk");
    let chunk_draws = {
        profiling::scope!("mesh::collect_prepared::chunk_capacity");
        runs.first()
            .and_then(|first| runs.last().map(|last| last.end - first.start))
            .unwrap_or(0) as usize
    };
    let mut out: Vec<WorldMeshDrawItem> = Vec::with_capacity(chunk_draws);
    let mut cull_stats = (0usize, 0usize, 0usize);

    {
        profiling::scope!("mesh::collect_prepared::renderer_runs");
        for prepared_run in runs {
            let start = prepared_run.start as usize;
            let end = prepared_run.end as usize;
            let run = &draws[start..end];
            let run_stats = collect_prepared_renderer_run(run, ctx, cache, filter_masks, &mut out);
            cull_stats.0 += run_stats.0;
            cull_stats.1 += run_stats.1;
            cull_stats.2 += run_stats.2;
        }
    }

    (out, cull_stats)
}
