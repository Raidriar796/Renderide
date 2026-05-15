//! Material batch packet resolution for world-mesh forward draws.
//!
//! The resolver is the single boundary between sorted CPU draw runs and concrete raster state.
//! Backend frame planning builds [`PipelineVariantKey`] once per batch so raster recording cannot
//! drift on MSAA, front-face, blend, render-state, or shader permutations.

use std::sync::Arc;

use rayon::prelude::*;

use crate::materials::ShaderPermutation;
use crate::materials::{
    EmbeddedMaterialBindResources, EmbeddedMaterialBindShader, EmbeddedTexturePools,
};
use crate::materials::{
    MaterialBlendMode, MaterialPipelineDesc, MaterialPipelineSet, MaterialPipelineVariantSpec,
    MaterialRegistry, MaterialRenderState, RasterFrontFace, RasterPipelineKind,
    RasterPrimitiveTopology,
};
use crate::passes::WorldMeshForwardEncodeRefs;
use crate::render_graph::frame_upload_batch::GraphUploadSink;
use crate::world_mesh::draw_prep::WorldMeshDrawItem;

/// Inclusive `(first_draw_idx, last_draw_idx)` span over the sorted world-mesh draw list
/// identifying one contiguous material batch run.
pub(crate) type MaterialBatchBoundary = (usize, usize);

/// One resolved per-batch draw packet covering a contiguous range of sorted draws with the same
/// [`crate::world_mesh::MaterialDrawBatchKey`].
///
/// Populated by backend frame planning so the recording loop can drive pipeline and bind-group state
/// entirely from this table, without material-cache lookups inside `RenderPass`.
#[derive(Clone)]
pub(crate) struct MaterialBatchPacket {
    /// First draw index (into the sorted draw list) covered by this entry.
    pub first_draw_idx: usize,
    /// Last draw index (inclusive) covered by this entry.
    pub last_draw_idx: usize,
    /// Exact pipeline variant requested for this batch.
    pub(crate) pipeline_key: PipelineVariantKey,
    /// Resolved `@group(1)` bind group for this batch's material, or `None` for the empty fallback.
    pub bind_group: Option<Arc<wgpu::BindGroup>>,
    /// Dynamic offset for the material uniform arena, when this batch's bind group has one.
    pub material_uniform_dynamic_offset: Option<u32>,
    /// Resolved pipeline set for this batch, or `None` when the pipeline is unavailable (skip draws).
    pub pipelines: Option<MaterialPipelineSet>,
}

/// Inputs needed to build a [`PipelineVariantKey`] for one material draw run.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PipelineVariantKeyInput {
    /// Base pass descriptor for the owning view.
    pub pass_desc: MaterialPipelineDesc,
    /// Shader permutation selected for the owning view.
    pub shader_perm: ShaderPermutation,
    /// Host shader asset id for diagnostics and material registry lookup.
    pub shader_asset_id: i32,
    /// Resolved material blend state.
    pub blend_mode: MaterialBlendMode,
    /// Resolved material render state.
    pub render_state: MaterialRenderState,
    /// Front-face winding selected from the draw transform.
    pub front_face: RasterFrontFace,
    /// Primitive topology selected from the mesh's per-submesh topology.
    pub primitive_topology: RasterPrimitiveTopology,
}

/// Exact material pipeline variant used by backend frame planning.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PipelineVariantKey {
    /// Host shader asset id for diagnostics and material registry lookup.
    pub shader_asset_id: i32,
    /// Color attachment format.
    pub surface_format: wgpu::TextureFormat,
    /// Optional depth/stencil format.
    pub depth_stencil_format: Option<wgpu::TextureFormat>,
    /// Effective sample count for the active render pass.
    pub sample_count: u32,
    /// Optional multiview mask.
    pub multiview_mask: Option<std::num::NonZeroU32>,
    /// Shader permutation selected for the view.
    pub shader_perm: ShaderPermutation,
    /// Resolved material blend state.
    pub blend_mode: MaterialBlendMode,
    /// Resolved material render state.
    pub render_state: MaterialRenderState,
    /// Front-face winding selected from the draw transform.
    pub front_face: RasterFrontFace,
    /// Primitive topology selected from the mesh's per-submesh topology.
    pub primitive_topology: RasterPrimitiveTopology,
}

impl PipelineVariantKey {
    /// Builds the key used for material packet resolution.
    pub(crate) fn new(input: PipelineVariantKeyInput) -> Self {
        let PipelineVariantKeyInput {
            pass_desc,
            shader_perm,
            shader_asset_id,
            blend_mode,
            render_state,
            front_face,
            primitive_topology,
        } = input;
        Self {
            shader_asset_id,
            surface_format: pass_desc.surface_format,
            depth_stencil_format: pass_desc.depth_stencil_format,
            sample_count: pass_desc.sample_count,
            multiview_mask: pass_desc.multiview_mask,
            shader_perm,
            blend_mode,
            render_state,
            front_face,
            primitive_topology,
        }
    }

    /// Rehydrates the material pipeline descriptor used by [`MaterialRegistry`].
    pub(crate) fn pass_desc(self) -> MaterialPipelineDesc {
        MaterialPipelineDesc {
            surface_format: self.surface_format,
            depth_stencil_format: self.depth_stencil_format,
            sample_count: self.sample_count,
            multiview_mask: self.multiview_mask,
        }
    }

    /// Builds a key directly from a sorted draw item and view-level pipeline state.
    pub(crate) fn for_draw_item(
        item: &WorldMeshDrawItem,
        pass_desc: MaterialPipelineDesc,
        shader_perm: ShaderPermutation,
    ) -> Self {
        let batch_key = &item.batch_key;
        Self::new(PipelineVariantKeyInput {
            pass_desc,
            shader_perm,
            shader_asset_id: batch_key.shader_asset_id,
            blend_mode: batch_key.blend_mode,
            render_state: batch_key.render_state,
            front_face: batch_key.front_face,
            primitive_topology: batch_key.primitive_topology,
        })
    }
}

/// Material pipeline and embedded-bind resolver for one world-mesh forward view plan.
pub(crate) struct MaterialDrawResolver<'a> {
    /// Material registry used for pipeline lookup.
    registry: Option<&'a MaterialRegistry>,
    /// Embedded material bind resources used for `@group(1)` lookup.
    embedded_bind: Option<&'a EmbeddedMaterialBindResources>,
    /// Material property store used by embedded bind resolution.
    store: &'a crate::materials::host_data::MaterialPropertyStore,
    /// Texture pools used by embedded bind resolution.
    pools: EmbeddedTexturePools<'a>,
    /// Upload sink used by embedded uniform updates.
    uploads: GraphUploadSink<'a>,
    /// View-level material pipeline descriptor before per-material overrides.
    pass_desc: MaterialPipelineDesc,
    /// Shader permutation for this view.
    shader_perm: ShaderPermutation,
    /// Offscreen render texture being written by this view, if any.
    offscreen_write_render_texture_asset_id: Option<i32>,
}

impl<'a> MaterialDrawResolver<'a> {
    /// Builds a resolver from the forward encode references for this view.
    pub(crate) fn new(
        encode: &'a WorldMeshForwardEncodeRefs<'_>,
        uploads: GraphUploadSink<'a>,
        pass_desc: MaterialPipelineDesc,
        shader_perm: ShaderPermutation,
        offscreen_write_render_texture_asset_id: Option<i32>,
    ) -> Self {
        Self {
            registry: encode.materials.material_registry(),
            embedded_bind: encode.materials.embedded_material_bind(),
            store: encode.materials.material_property_store(),
            pools: encode.embedded_texture_pools(),
            uploads,
            pass_desc,
            shader_perm,
            offscreen_write_render_texture_asset_id,
        }
    }

    /// Resolves every contiguous material run in `draws` into record-ready packets.
    ///
    /// `boundaries_scratch` is cleared and refilled with the material-batch boundary spans; the
    /// caller owns the buffer so its capacity survives across frames and reallocates only on
    /// growth past the previous high-water mark.
    pub(crate) fn resolve_batches(
        &self,
        draws: &[WorldMeshDrawItem],
        boundaries_scratch: &mut Vec<MaterialBatchBoundary>,
    ) -> Vec<MaterialBatchPacket> {
        profiling::scope!("world_mesh_forward::resolve_material_packets");
        boundaries_scratch.clear();
        if draws.is_empty() {
            return Vec::new();
        }

        collect_material_batch_boundaries_into(draws, boundaries_scratch);
        if boundaries_scratch.len() < 2 {
            let mut packets = Vec::with_capacity(boundaries_scratch.len());
            for &(first, last) in boundaries_scratch.iter() {
                packets.push(self.resolve_one_batch(draws, first, last));
            }
            packets
        } else {
            boundaries_scratch
                .par_iter()
                .copied()
                .map(|(first, last)| self.resolve_one_batch(draws, first, last))
                .collect()
        }
    }

    /// Resolves one material run into a record-ready packet.
    fn resolve_one_batch(
        &self,
        draws: &[WorldMeshDrawItem],
        first: usize,
        last: usize,
    ) -> MaterialBatchPacket {
        let item = &draws[first];
        let mut pipeline_key =
            PipelineVariantKey::for_draw_item(item, self.pass_desc, self.shader_perm);
        if self.offscreen_write_render_texture_asset_id.is_some() {
            // View-projection matrices for offscreen-RT views are pre-multiplied by a clip-space
            // Y flip so the resulting render-texture lands in Unity (V=0 bottom) orientation.
            // That mirrors triangle winding, so the pipeline needs the inverted `front_face` to
            // keep back-face culling correct.
            pipeline_key.front_face = pipeline_key.front_face.flipped();
        }

        let (bind_group, material_uniform_dynamic_offset) = self.resolve_embedded_bind_group(item);
        let pipeline_kind =
            pipeline_kind_for_material_packet(&item.batch_key.pipeline, bind_group.is_some());
        let pipelines = self.resolve_pipelines(&pipeline_kind, pipeline_key);

        MaterialBatchPacket {
            first_draw_idx: first,
            last_draw_idx: last,
            pipeline_key,
            bind_group,
            material_uniform_dynamic_offset,
            pipelines,
        }
    }

    /// Resolves the material pipeline set for one batch.
    fn resolve_pipelines(
        &self,
        pipeline_kind: &RasterPipelineKind,
        pipeline_key: PipelineVariantKey,
    ) -> Option<MaterialPipelineSet> {
        let registry = self.registry?;

        let pass_desc = pipeline_key.pass_desc();
        let pipelines = registry.pipeline_for_resolved_kind(
            pipeline_key.shader_asset_id,
            pipeline_kind,
            &pass_desc,
            MaterialPipelineVariantSpec {
                permutation: pipeline_key.shader_perm,
                blend_mode: pipeline_key.blend_mode,
                render_state: pipeline_key.render_state,
                front_face: pipeline_key.front_face,
                primitive_topology: pipeline_key.primitive_topology,
            },
        );

        match pipelines {
            Some(p) if !p.is_empty() => Some(p),
            Some(_) => {
                logger::trace!(
                    "WorldMeshForward: empty pipeline for shader {:?} kind {:?}, skipping batch",
                    pipeline_key.shader_asset_id,
                    pipeline_kind
                );
                None
            }
            None => {
                logger::trace!(
                    "WorldMeshForward: no pipeline for shader {:?} kind {:?}, skipping batch",
                    pipeline_key.shader_asset_id,
                    pipeline_kind
                );
                None
            }
        }
    }

    /// Resolves the embedded material bind group for one batch when the pipeline is embedded.
    fn resolve_embedded_bind_group(
        &self,
        item: &WorldMeshDrawItem,
    ) -> (Option<Arc<wgpu::BindGroup>>, Option<u32>) {
        let batch_key = &item.batch_key;
        let RasterPipelineKind::EmbeddedStem(stem) = &batch_key.pipeline else {
            return (None, None);
        };

        let Some(bind) = self.embedded_bind else {
            logger::warn!(
                "WorldMeshForward: embedded material bind resources unavailable; \
                 falling back to Null pipeline for shader_asset_id={} material_asset_id={} \
                 property_block_slot0={:?} stem={}",
                batch_key.shader_asset_id,
                item.lookup_ids.material_asset_id,
                item.lookup_ids.mesh_property_block_slot0,
                stem.as_ref()
            );
            return (None, None);
        };

        let shader_variant_bits = self
            .registry
            .and_then(|registry| registry.variant_bits_for_shader_asset(batch_key.shader_asset_id));
        match bind.embedded_material_bind_group_with_cache_key(
            EmbeddedMaterialBindShader {
                stem: stem.as_ref(),
                shader_variant_bits,
            },
            self.uploads,
            self.store,
            &self.pools,
            item.lookup_ids,
            self.offscreen_write_render_texture_asset_id,
        ) {
            Ok((_, bg)) => (Some(bg.bind_group), bg.uniform_dynamic_offset),
            Err(e) => {
                logger::warn!(
                    "WorldMeshForward: embedded material bind failed; \
                     falling back to Null pipeline for shader_asset_id={} material_asset_id={} \
                     property_block_slot0={:?} stem={} error={}",
                    batch_key.shader_asset_id,
                    item.lookup_ids.material_asset_id,
                    item.lookup_ids.mesh_property_block_slot0,
                    stem.as_ref(),
                    e
                );
                (None, None)
            }
        }
    }
}

/// Selects the pipeline family that is safe for the resolved material bind-group state.
fn pipeline_kind_for_material_packet(
    batch_pipeline: &RasterPipelineKind,
    embedded_bind_group_resolved: bool,
) -> RasterPipelineKind {
    match batch_pipeline {
        RasterPipelineKind::EmbeddedStem(stem) if embedded_bind_group_resolved => {
            RasterPipelineKind::EmbeddedStem(stem.clone())
        }
        RasterPipelineKind::EmbeddedStem(_) | RasterPipelineKind::Null => RasterPipelineKind::Null,
    }
}

/// Walks `draws` once and writes `(first_idx, last_idx)` runs of identical material batch keys
/// into the caller-supplied `out` buffer. `out` is cleared before filling.
fn collect_material_batch_boundaries_into(
    draws: &[WorldMeshDrawItem],
    out: &mut Vec<MaterialBatchBoundary>,
) {
    out.clear();
    let mut current_start = 0usize;
    let mut last_key = &draws[0].batch_key;
    for (idx, item) in draws.iter().enumerate().skip(1) {
        if &item.batch_key != last_key {
            out.push((current_start, idx - 1));
            current_start = idx;
            last_key = &item.batch_key;
        }
    }
    out.push((current_start, draws.len() - 1));
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;
    use std::sync::Arc;

    use super::*;
    use crate::world_mesh::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};

    fn base_desc() -> MaterialPipelineDesc {
        MaterialPipelineDesc {
            surface_format: wgpu::TextureFormat::Rgba16Float,
            depth_stencil_format: Some(wgpu::TextureFormat::Depth24PlusStencil8),
            sample_count: 4,
            multiview_mask: NonZeroU32::new(3),
        }
    }

    fn key_for() -> PipelineVariantKey {
        PipelineVariantKey::new(PipelineVariantKeyInput {
            pass_desc: base_desc(),
            shader_perm: ShaderPermutation(1),
            shader_asset_id: 42,
            blend_mode: MaterialBlendMode::Opaque,
            render_state: MaterialRenderState::default(),
            front_face: RasterFrontFace::CounterClockwise,
            primitive_topology: RasterPrimitiveTopology::TriangleList,
        })
    }

    /// Builds an embedded pipeline kind for packet-selection tests.
    fn embedded_pipeline(stem: &'static str) -> RasterPipelineKind {
        RasterPipelineKind::EmbeddedStem(Arc::from(stem))
    }

    #[test]
    fn pipeline_key_preserves_regular_sample_count() {
        let key = key_for();
        assert_eq!(key.sample_count, 4);
        assert_eq!(key.pass_desc().sample_count, 4);
    }

    #[test]
    fn pipeline_key_preserves_grab_pass_sample_count() {
        let mut item = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 42,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 7,
            node_id: 1,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: false,
        });
        item.batch_key.shader_asset_id = 42;
        item.batch_key.blend_mode = MaterialBlendMode::Opaque;
        item.batch_key.front_face = RasterFrontFace::CounterClockwise;
        item.batch_key.embedded_uses_scene_color_snapshot = true;

        let key = PipelineVariantKey::for_draw_item(&item, base_desc(), ShaderPermutation(1));
        assert_eq!(key.sample_count, 4);
        assert_eq!(key.pass_desc().sample_count, 4);
        assert_eq!(key.surface_format, wgpu::TextureFormat::Rgba16Float);
        assert_eq!(
            key.depth_stencil_format,
            Some(wgpu::TextureFormat::Depth24PlusStencil8)
        );
        assert_eq!(key.multiview_mask, NonZeroU32::new(3));
    }

    #[test]
    fn pipeline_key_changes_when_front_face_changes() {
        let mut a = key_for();
        let mut b = key_for();
        a.front_face = RasterFrontFace::Clockwise;
        b.front_face = RasterFrontFace::CounterClockwise;
        assert_ne!(a, b);
    }

    /// Embedded packets keep their embedded pipeline only when group 1 was resolved.
    #[test]
    fn material_packet_keeps_embedded_pipeline_with_resolved_bind_group() {
        let pipeline = embedded_pipeline("xstoon2.0_default");

        assert_eq!(
            pipeline_kind_for_material_packet(&pipeline, true),
            embedded_pipeline("xstoon2.0_default")
        );
    }

    /// Embedded packets use the null pipeline when group 1 cannot match the embedded layout.
    #[test]
    fn material_packet_falls_back_to_null_when_embedded_bind_group_is_missing() {
        let pipeline = embedded_pipeline("xstoon2.0_default");

        assert_eq!(
            pipeline_kind_for_material_packet(&pipeline, false),
            RasterPipelineKind::Null
        );
    }

    /// Null packets can safely keep using the empty material bind group.
    #[test]
    fn material_packet_keeps_null_pipeline_with_empty_material_group() {
        assert_eq!(
            pipeline_kind_for_material_packet(&RasterPipelineKind::Null, false),
            RasterPipelineKind::Null
        );
    }

    /// Packet selection honors the draw batch snapshot instead of a later router route.
    #[test]
    fn material_packet_uses_draw_batch_pipeline_snapshot() {
        let current_router_route = embedded_pipeline("xstoon2.0_default");
        let stale_draw_batch_pipeline = RasterPipelineKind::Null;

        assert_eq!(
            pipeline_kind_for_material_packet(&stale_draw_batch_pipeline, true),
            RasterPipelineKind::Null
        );
        assert_ne!(stale_draw_batch_pipeline, current_router_route);
    }
}
