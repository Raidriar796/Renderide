//! Vertex-color color-space conversion helpers.
//!
//! Material shaders receive vertex colors in whichever space the host marked them with via the
//! `_VERTEX_LINEAR_COLOR` / `_VERTEX_SRGB_COLOR` / `_VERTEX_HDRSRGB_COLOR` keyword groups.
//! Linear-tagged colors are passed through. SRGB-tagged colors run through the renderer transfer
//! curve to land in linear space.

#define_import_path renderide::material::vertex_color

fn srgb_channel_to_linear(value: f32) -> f32 {
    if (value <= 0.04045) {
        return value / 12.92;
    } else if (value < 1.0) {
        return pow((value + 0.055) / 1.055, 2.4);
    }
    return pow(value, 2.2);
}

/// Convert an sRGB-tagged LDR vertex color to linear. Alpha is passed through.
fn srgb_to_linear_ldr(color: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        srgb_channel_to_linear(color.r),
        srgb_channel_to_linear(color.g),
        srgb_channel_to_linear(color.b),
        color.a,
    );
}

/// Convert an sRGB-tagged HDR vertex color to linear. Alpha is passed through.
fn srgb_to_linear_hdr(color: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        srgb_channel_to_linear(color.r),
        srgb_channel_to_linear(color.g),
        srgb_channel_to_linear(color.b),
        color.a,
    );
}
