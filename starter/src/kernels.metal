// kernels.metal — Gaussian blur kernels for test task 03 ("Mac" track).
// Compiled at runtime; the variant name passed with --variant is the kernel
// function name (./build/blur_metal --list shows all of them). A two-pass
// variant "foo" consists of two kernels "foo_h" and "foo_v".
//
// The correct result is defined by cpu_reference/gaussian_blur_ref.h:
// clamp-to-edge, r = ceil(3 * sigma), normalized weights, rounding floor(v + 0.5),
// tolerance max |diff| <= 1 per channel.
//
// Buffers (set up by main.mm):
//   0 BlurParams, 1 src (uchar4 per pixel), 2 dst (uchar4 per pixel),
//   3 weights (2r+1 floats, center at index r), 4 tmp (float4 per pixel, GPU-only).
//
// TODO(naive): direct 2D convolution, (2r+1)^2 reads per pixel, one pass,
//           weight of (i, j) = weights[r + i] * weights[r + j].
// TODO(optimized): ONE optimized variant, for example:
//   * separable_h + separable_v — horizontal pass into tmp, vertical pass into dst;
//   * tiled — tile + apron in threadgroup memory, threadgroup_barrier, one pass.
// Write down your hypothesis ("why it should be faster") BEFORE measuring.

#include <metal_stdlib>
using namespace metal;

struct BlurParams {
    uint width;
    uint height;
    int radius;
};

kernel void naive(constant BlurParams& params  [[buffer(0)]],
                  device const uchar4* src      [[buffer(1)]],
                  device uchar4* dst            [[buffer(2)]],
                  constant float* weights       [[buffer(3)]],
                  uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.width || gid.y >= params.height) return;
    // TODO(naive): read the neighbourhood (clamp-to-edge), apply the weights,
    //           round and write the result. Placeholder: copy the input.
    (void)weights;
    dst[gid.y * params.width + gid.x] = src[gid.y * params.width + gid.x];
}
