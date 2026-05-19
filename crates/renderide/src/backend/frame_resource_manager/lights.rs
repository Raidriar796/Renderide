//! Light preparation and per-view light access for [`FrameResourceManager`].

use crate::camera::ViewId;
use crate::scene::{SceneCoordinator, light_contributes, light_has_negative_contribution};

use super::super::light_gpu::{
    GpuLight, MAX_LIGHTS, gpu_light_from_resolved, order_lights_for_clustered_shading_in_place,
};
use super::manager::FrameResourceManager;
use super::per_view_state::PreparedViewLights;
use super::view_desc::FrameLightViewDesc;

impl FrameResourceManager {
    /// Packed GPU lights from the last [`Self::prepare_lights_from_scene`] call.
    pub fn frame_lights(&self) -> &[GpuLight] {
        &self.light_scratch
    }

    /// Packed GPU lights for `view_id`, falling back to the last default frame pack.
    pub fn frame_lights_for_view(&self, view_id: ViewId) -> &[GpuLight] {
        self.per_view_lights
            .get(view_id)
            .map_or(self.light_scratch.as_slice(), |lights| {
                lights.lights.as_slice()
            })
    }

    /// Returns true when the current packed light set needs signed scene-color storage.
    pub fn signed_scene_color_required(&self) -> bool {
        self.signed_scene_color_required
    }

    /// Light count for the specified view's frame uniforms and shaders.
    pub fn frame_light_count_for_view_u32(&self, view_id: ViewId) -> u32 {
        self.frame_lights_for_view(view_id).len().min(MAX_LIGHTS) as u32
    }

    /// Fills the default main-view light scratch buffer from active render spaces.
    ///
    /// This compatibility entry point is used by unit tests and callers that do not have explicit
    /// view planning information. Normal graph rendering should call
    /// [`Self::prepare_lights_for_views`] so secondary cameras get render-context-aware light
    /// packs.
    #[cfg(test)]
    pub fn prepare_lights_from_scene(&mut self, scene: &SceneCoordinator) {
        self.prepare_lights_for_views(
            scene,
            [FrameLightViewDesc {
                view_id: ViewId::Main,
                render_context: scene.active_main_render_context(),
                render_space_filter: None,
                head_output_transform: glam::Mat4::IDENTITY,
            }],
        );
    }

    /// Fills per-view light scratch buffers from [`SceneCoordinator`].
    ///
    /// Inactive spaces are skipped so lights from a previously focused world do not persist into
    /// the next frame's shading. Views with a render-space filter only receive lights from that
    /// space. Non-contributing lights are filtered via [`light_contributes`] before clustered
    /// ordering, and each view's transforms are resolved with the same render context and
    /// head-output transform used by draw collection.
    pub(crate) fn prepare_lights_for_views<I>(&mut self, scene: &SceneCoordinator, views: I)
    where
        I: IntoIterator<Item = FrameLightViewDesc>,
    {
        profiling::scope!("render::prepare_lights_for_views");
        self.light_scratch.clear();
        self.signed_scene_color_required = false;
        let mut wrote_fallback = false;
        for desc in views {
            self.prepare_lights_for_view(scene, desc);
            self.signed_scene_color_required |= self
                .per_view_lights
                .get(desc.view_id)
                .is_some_and(|lights| lights.signed_scene_color_required);
            if !wrote_fallback {
                let fallback_lights = self.frame_lights_for_view(desc.view_id).to_vec();
                self.light_scratch.clear();
                self.light_scratch.extend(fallback_lights);
                wrote_fallback = true;
            }
        }
        if self.signed_scene_color_required && !self.signed_scene_color_required_logged {
            logger::info!(
                "negative direct lights active: signed scene-color HDR will be used while negative lights are packed"
            );
            self.signed_scene_color_required_logged = true;
        }
    }

    fn prepare_lights_for_view(&mut self, scene: &SceneCoordinator, desc: FrameLightViewDesc) {
        profiling::scope!("render::prepare_lights_for_view");
        self.resolved_flatten_scratch.clear();
        self.collect_light_space_ids(scene, desc.render_space_filter);
        self.resolve_lights_for_space_ids(scene, desc);
        {
            profiling::scope!("render::prepare_lights::filter_contributors");
            self.resolved_flatten_scratch.retain(light_contributes);
        }
        order_lights_for_clustered_shading_in_place(&mut self.resolved_flatten_scratch);
        let resolved_len = self.resolved_flatten_scratch.len();
        if resolved_len > MAX_LIGHTS && !self.lights_overflow_warned {
            logger::warn!(
                "scene contains {resolved_len} contributing lights but the engine only uploads \
                 the first {MAX_LIGHTS} (MAX_LIGHTS); the remainder will be ignored for shading. \
                 This warning is only logged once per renderer instance."
            );
            self.lights_overflow_warned = true;
        }
        let kept = resolved_len.min(MAX_LIGHTS);
        let signed_scene_color_required = self
            .resolved_flatten_scratch
            .iter()
            .take(kept)
            .any(light_has_negative_contribution);
        let entry = self
            .per_view_lights
            .get_or_insert_with(desc.view_id, PreparedViewLights::default);
        entry.lights.clear();
        entry.lights.reserve(kept);
        entry.lights.extend(
            self.resolved_flatten_scratch
                .iter()
                .take(kept)
                .map(gpu_light_from_resolved),
        );
        entry.signed_scene_color_required = signed_scene_color_required;
        logger::trace!(
            "prepared lights for view {:?}: lights={} render_context={:?} render_space_filter={:?}",
            desc.view_id,
            entry.lights.len(),
            desc.render_context,
            desc.render_space_filter
        );
    }

    fn collect_light_space_ids(
        &mut self,
        scene: &SceneCoordinator,
        render_space_filter: Option<crate::scene::RenderSpaceId>,
    ) {
        profiling::scope!("render::prepare_lights::collect_active_spaces");
        self.light_space_ids_scratch.clear();
        if let Some(id) = render_space_filter {
            if scene.space(id).is_some_and(|space| space.is_active()) {
                self.light_space_ids_scratch.push(id);
            }
            return;
        }
        self.light_space_ids_scratch.extend(
            scene
                .render_space_ids()
                .filter(|id| scene.space(*id).is_some_and(|space| space.is_active())),
        );
    }

    fn resolve_lights_for_space_ids(&mut self, scene: &SceneCoordinator, desc: FrameLightViewDesc) {
        if self.light_space_ids_scratch.is_empty() {
            return;
        }
        profiling::scope!("render::prepare_lights::resolve_spaces");
        // The host scenes we render typically have one or two active render spaces (main world
        // plus an optional overlay), and each space's `resolve_lights_into` is short. Earlier
        // versions of this function fanned the multi-space path out across `rayon::par_iter` with
        // a fresh `Vec<Vec<ResolvedLight>>` per frame; the per-frame allocation and the rayon
        // dispatch overhead both exceeded the work being parallelized. Serial append into the
        // already-cleared `resolved_flatten_scratch` reuses last frame's allocation and lets the
        // CPU plough through the spaces with no synchronization.
        for &id in &self.light_space_ids_scratch {
            scene.resolve_lights_for_render_context_into(
                id,
                desc.render_context,
                desc.head_output_transform,
                &mut self.resolved_flatten_scratch,
            );
        }
    }
}
