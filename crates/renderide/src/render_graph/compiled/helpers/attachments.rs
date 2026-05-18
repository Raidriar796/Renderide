//! Raster pass attachment resolution and the wgpu render-pass lifecycle for graph passes.

use crate::render_graph::context::{GraphResolvedResources, RasterPassCtx};
use crate::render_graph::error::GraphExecuteError;
use crate::render_graph::pass::{PassNode, RenderPassTemplate};
use crate::render_graph::resources::{
    TextureAttachmentResolve, TextureAttachmentTarget, TextureResourceHandle,
};

use super::super::CompiledPassInfo;

pub(in crate::render_graph::compiled) fn pass_info_raster_template(
    pass_info: &[CompiledPassInfo],
    pass_idx: usize,
) -> Result<RenderPassTemplate, GraphExecuteError> {
    let Some(info) = pass_info.get(pass_idx) else {
        return Err(GraphExecuteError::MissingRasterTemplate {
            pass: format!("pass#{pass_idx}"),
        });
    };
    info.raster_template
        .clone()
        .ok_or_else(|| GraphExecuteError::MissingRasterTemplate {
            pass: info.name.clone(),
        })
}

pub(in crate::render_graph::compiled) fn frame_sample_count_from_raster_ctx(
    ctx: &RasterPassCtx<'_, '_>,
) -> u32 {
    ctx.pass_frame.view.sample_count.max(1)
}

pub(in crate::render_graph::compiled) fn resolve_attachment_target(
    target: TextureAttachmentTarget,
    sample_count: u32,
) -> TextureResourceHandle {
    match target {
        TextureAttachmentTarget::Resource(handle) => handle,
        TextureAttachmentTarget::FrameSampled {
            single_sample,
            multisampled,
        } => {
            if sample_count > 1 {
                multisampled
            } else {
                single_sample
            }
        }
    }
}

pub(in crate::render_graph::compiled) fn resolve_attachment_resolve_target(
    target: TextureAttachmentResolve,
    sample_count: u32,
) -> Option<TextureResourceHandle> {
    match target {
        TextureAttachmentResolve::Always(handle) => Some(handle),
        TextureAttachmentResolve::FrameMultisampled(handle) => (sample_count > 1).then_some(handle),
    }
}

/// Resolves the raster pass template's color attachments to live `wgpu::TextureView`s.
fn resolve_color_attachments<'a>(
    pass_name: &str,
    template: &RenderPassTemplate,
    graph_resources: &'a GraphResolvedResources,
    sample_count: u32,
) -> Result<Vec<Option<wgpu::RenderPassColorAttachment<'a>>>, GraphExecuteError> {
    profiling::scope!("graph::raster::resolve_color_attachments");
    let mut color_attachments = Vec::with_capacity(template.color_attachments.len());
    for color in &template.color_attachments {
        let target = resolve_attachment_target(color.target, sample_count);
        let view = graph_resources.texture_view(target).ok_or_else(|| {
            GraphExecuteError::MissingGraphAttachment {
                pass: pass_name.to_owned(),
                resource: format!("{target:?}"),
            }
        })?;
        let resolve_target = match color
            .resolve_to
            .and_then(|t| resolve_attachment_resolve_target(t, sample_count))
        {
            Some(target) => Some(graph_resources.texture_view(target).ok_or_else(|| {
                GraphExecuteError::MissingGraphAttachment {
                    pass: pass_name.to_owned(),
                    resource: format!("{target:?}"),
                }
            })?),
            None => None,
        };
        color_attachments.push(Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target,
            ops: wgpu::Operations {
                load: color.load,
                store: color.store,
            },
            depth_slice: None,
        }));
    }
    Ok(color_attachments)
}

/// Resolves the raster pass template's depth/stencil attachment, if any, to a live view.
fn resolve_depth_attachment<'a>(
    pass: &PassNode,
    template: &RenderPassTemplate,
    graph_resources: &'a GraphResolvedResources,
    sample_count: u32,
    ctx: &RasterPassCtx<'_, '_>,
) -> Result<Option<wgpu::RenderPassDepthStencilAttachment<'a>>, GraphExecuteError> {
    profiling::scope!("graph::raster::resolve_depth_attachment");
    let Some(depth) = &template.depth_stencil_attachment else {
        return Ok(None);
    };
    let target = resolve_attachment_target(depth.target, sample_count);
    let view = graph_resources.texture_view(target).ok_or_else(|| {
        GraphExecuteError::MissingGraphAttachment {
            pass: pass.name().to_owned(),
            resource: format!("{target:?}"),
        }
    })?;
    let stencil_ops = pass.stencil_ops_override(ctx, depth);
    Ok(Some(wgpu::RenderPassDepthStencilAttachment {
        view,
        depth_ops: Some(depth.depth),
        stencil_ops,
    }))
}

/// Opens a graph-managed raster render pass for a [`PassNode::Raster`] variant and calls
/// [`crate::render_graph::pass::PassNode::record_raster`].
///
/// This is the primary path for raster passes in the new pass-node system. Attachment resolution
/// is delegated to [`resolve_color_attachments`] and [`resolve_depth_attachment`]; the body here
/// owns only the wgpu render-pass lifecycle and the GPU-profiler query bracketing.
pub(in crate::render_graph::compiled) fn execute_graph_raster_pass_node(
    pass: &PassNode,
    template: &RenderPassTemplate,
    graph_resources: &GraphResolvedResources,
    encoder: &mut wgpu::CommandEncoder,
    ctx: &mut RasterPassCtx<'_, '_>,
) -> Result<(), GraphExecuteError> {
    let should_record = {
        profiling::scope!("graph::raster::should_record");
        pass.should_record_raster(ctx)
            .map_err(GraphExecuteError::Pass)?
    };
    if !should_record {
        return Ok(());
    }

    let sample_count = frame_sample_count_from_raster_ctx(ctx);
    let color_attachments =
        resolve_color_attachments(pass.name(), template, graph_resources, sample_count)?;
    let depth_stencil_attachment =
        resolve_depth_attachment(pass, template, graph_resources, sample_count, ctx)?;
    let multiview_mask = pass.multiview_mask_override(ctx, template);

    let pass_profile_label = pass.profiling_label();
    let pass_query = ctx
        .profiler
        .map(|p| p.begin_pass_query(pass_profile_label, encoder));
    let timestamp_writes = crate::profiling::render_pass_timestamp_writes(pass_query.as_ref());
    let mut rpass = {
        profiling::scope!("graph::raster::begin_render_pass");
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render-graph-raster"),
            color_attachments: &color_attachments,
            depth_stencil_attachment,
            occlusion_query_set: None,
            timestamp_writes,
            multiview_mask,
        })
    };
    {
        profiling::scope!("graph::raster::record_draws");
        pass.record_raster(ctx, &mut rpass)
            .map_err(GraphExecuteError::Pass)?;
    }
    {
        profiling::scope!("graph::raster::end_render_pass");
        drop(rpass);
    }
    if let (Some(p), Some(q)) = (ctx.profiler, pass_query) {
        p.end_query(encoder, q);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render_graph::resources::{ImportedTextureHandle, TextureHandle};

    #[test]
    fn attachment_target_selects_single_or_multisampled_resource() {
        let single = TextureResourceHandle::Transient(TextureHandle(1));
        let multi = TextureResourceHandle::Transient(TextureHandle(2));
        let imported = TextureResourceHandle::Imported(ImportedTextureHandle(3));

        assert_eq!(
            resolve_attachment_target(TextureAttachmentTarget::Resource(imported), 8),
            imported
        );
        assert_eq!(
            resolve_attachment_target(
                TextureAttachmentTarget::FrameSampled {
                    single_sample: single,
                    multisampled: multi,
                },
                1,
            ),
            single
        );
        assert_eq!(
            resolve_attachment_target(
                TextureAttachmentTarget::FrameSampled {
                    single_sample: single,
                    multisampled: multi,
                },
                4,
            ),
            multi
        );
    }

    #[test]
    fn attachment_resolve_target_is_sample_count_aware() {
        let target = TextureResourceHandle::Transient(TextureHandle(7));

        assert_eq!(
            resolve_attachment_resolve_target(TextureAttachmentResolve::Always(target), 1),
            Some(target)
        );
        assert_eq!(
            resolve_attachment_resolve_target(
                TextureAttachmentResolve::FrameMultisampled(target),
                1,
            ),
            None
        );
        assert_eq!(
            resolve_attachment_resolve_target(
                TextureAttachmentResolve::FrameMultisampled(target),
                2,
            ),
            Some(target)
        );
    }
}
