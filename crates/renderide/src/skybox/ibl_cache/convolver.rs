//! Standalone GGX convolver for caller-owned cubemap textures.

use std::sync::Arc;

use crate::gpu::GpuContext;
use crate::profiling::GpuProfilerHandle;

use super::encode::{
    ConvolveEncodeContext, DownsampleEncodeContext, encode_convolve_mips, encode_downsample_mips,
};
use super::errors::SkyboxIblConvolveError;
use super::key::source_max_lod;
use super::pipeline_store::{PipelineSlot, PipelineStore};
use super::resources::{
    PendingBakeResources, copy_cube_mip0, create_full_cube_sample_view, create_ibl_cube,
};

/// Resources produced while encoding convolve passes and retained until submit completion.
pub(crate) struct SkyboxIblConvolveResources {
    _resources: PendingBakeResources,
    _source_sample_view: Arc<wgpu::TextureView>,
    _sampler: Arc<wgpu::Sampler>,
}

/// Minimal GGX convolver for caller-owned cubemap textures.
#[derive(Default)]
pub(crate) struct SkyboxIblConvolver {
    pipelines: PipelineStore,
}

impl SkyboxIblConvolver {
    /// Creates an empty convolver with lazily-built GPU resources.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Encodes GGX convolve passes for mips `1..mip_levels` of `texture`.
    pub(crate) fn encode_existing_cube_mips(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
        face_size: u32,
        mip_levels: u32,
        profiler: Option<&GpuProfilerHandle>,
    ) -> Result<SkyboxIblConvolveResources, SkyboxIblConvolveError> {
        profiling::scope!("skybox_ibl::encode_existing_cube_mips");
        let sampler = self.pipelines.ensure_sampler(gpu.device());
        self.pipelines
            .ensure(PipelineSlot::Convolve, gpu.device())
            .map_err(|_err| SkyboxIblConvolveError::MissingShader("skybox_ibl_convolve_params"))?;
        self.pipelines
            .ensure(PipelineSlot::Downsample, gpu.device())
            .map_err(|_err| SkyboxIblConvolveError::MissingShader("skybox_ibl_downsample"))?;
        let convolve_pipeline = self
            .pipelines
            .get(PipelineSlot::Convolve)
            .map_err(|_err| SkyboxIblConvolveError::MissingShader("skybox_ibl_convolve_params"))?;
        let downsample_pipeline = self
            .pipelines
            .get(PipelineSlot::Downsample)
            .map_err(|_err| SkyboxIblConvolveError::MissingShader("skybox_ibl_downsample"))?;

        let source_cube = create_ibl_cube(
            gpu.device(),
            "skybox_ibl_existing_source_cube",
            face_size,
            mip_levels,
        );
        let source_sample_view = Arc::new(create_full_cube_sample_view(
            source_cube.texture.as_ref(),
            mip_levels,
        ));
        let mut resources = PendingBakeResources::default();
        resources.textures.push(source_cube.texture.clone());
        resources.source_sample_view = Some(source_sample_view.clone());
        copy_cube_mip0(
            encoder,
            texture,
            source_cube.texture.as_ref(),
            face_size,
            profiler,
        );
        encode_downsample_mips(
            DownsampleEncodeContext {
                device: gpu.device(),
                encoder,
                pipeline: downsample_pipeline,
                texture: source_cube.texture.as_ref(),
                face_size,
                mip_levels,
                profiler,
            },
            &mut resources,
        );
        encode_convolve_mips(
            ConvolveEncodeContext {
                device: gpu.device(),
                encoder,
                pipeline: convolve_pipeline,
                texture,
                src_view: source_sample_view.as_ref(),
                sampler: sampler.as_ref(),
                face_size,
                mip_levels,
                src_max_lod: source_max_lod(mip_levels),
                profiler,
            },
            &mut resources,
        );
        Ok(SkyboxIblConvolveResources {
            _resources: resources,
            _source_sample_view: source_sample_view,
            _sampler: sampler,
        })
    }
}
