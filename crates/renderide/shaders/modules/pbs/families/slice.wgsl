//! Shared math for the PBS Slice material family.

#define_import_path renderide::pbs::families::slice

#import renderide::core::math as rmath

/// Result of evaluating up to eight slice planes against a fragment position.
struct SliceEvalResult {
    /// Signed minimum plane distance (negative means the fragment is clipped).
    min_distance: f32,
    /// `[0, 1]` factor that grows toward `1` inside the edge transition band.
    edge_lerp: f32,
}

fn plane_distance(p: vec3<f32>, normal: vec3<f32>, offset: f32) -> f32 {
    return dot(p, normal) + offset;
}

/// Walk the slicer plane array (stops at the first zero-normal entry) and pair
/// the minimum signed distance with the `[start, end]`-band edge transition.
fn evaluate_planes(
    slicers: array<vec4<f32>, 8>,
    slice_p: vec3<f32>,
    edge_start: f32,
    edge_end: f32,
) -> SliceEvalResult {
    var min_distance: f32 = 60000.0;
    for (var si: i32 = 0; si < 8; si = si + 1) {
        let slicer = slicers[si];
        if (all(slicer.xyz == vec3<f32>(0.0))) {
            break;
        }
        min_distance = min(min_distance, plane_distance(slice_p, slicer.xyz, slicer.w));
    }
    let edge_lerp = 1.0 - rmath::safe_lerp_factor(edge_start, edge_end, min_distance);
    return SliceEvalResult(min_distance, edge_lerp);
}

fn use_world_space(world_space_enabled: bool, object_space_enabled: bool) -> bool {
    if (object_space_enabled) {
        return false;
    }
    return world_space_enabled || (!object_space_enabled);
}

fn slice_position(
    world_pos: vec3<f32>,
    object_pos: vec3<f32>,
    world_space_enabled: bool,
    object_space_enabled: bool,
) -> vec3<f32> {
    return select(object_pos, world_pos, use_world_space(world_space_enabled, object_space_enabled));
}

fn blend_detail_normal(base_ts: vec3<f32>, detail_ts: vec3<f32>) -> vec3<f32> {
    return normalize(vec3<f32>(base_ts.xy + detail_ts.xy, base_ts.z * detail_ts.z));
}
