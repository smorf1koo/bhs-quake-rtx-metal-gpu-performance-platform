// gaussian_blur_ref.h — CPU reference implementation of the Gaussian blur (test task 03).
//
// This is the DEFINITION of the correct result. Your GPU versions must match
// it within max |diff| <= 1 per channel (see CompareImages).
//
// Specification (implement the same on the GPU):
//   * input/output: 8 bits per channel, channels = 3 (RGB) or 4 (RGBA), rows
//     tightly packed (stride = width * channels); all channels, alpha included,
//     are blurred the same way; no color space conversion;
//   * kernel radius r = ceil(3 * sigma), weights w[i] = exp(-i^2 / (2 sigma^2)),
//     i = -r..r, normalized so that they sum to 1;
//   * border: clamp-to-edge (coordinates are clamped to [0, width-1] / [0, height-1]);
//   * the filter is separable: horizontal pass first, then vertical;
//     the intermediate result is kept in float (no rounding to 8 bits);
//   * final rounding: out = clamp(floor(v + 0.5), 0, 255).
//   A naive 2D convolution with weights w[i]*w[j] is mathematically equivalent;
//   differences caused by the float operation order stay within tolerance 1.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bhs {

// Normalized kernel weights, 2r+1 values, center at index r.
std::vector<float> GaussianKernel(float sigma);

// Kernel radius for sigma: ceil(3 * sigma).
int GaussianRadius(float sigma);

// src and dst are different buffers of width*height*channels bytes.
void GaussianBlurRef(const uint8_t* src, uint8_t* dst, int width, int height, int channels, float sigma);

struct CompareResult {
    int maxAbsDiff = 0;
    double meanAbsDiff = 0.0;
    size_t pixelsOverTolerance = 0;  // pixels where at least one channel differs by more than tolerance
};

CompareResult CompareImages(const uint8_t* a, const uint8_t* b, int width, int height, int channels,
                            int tolerance = 1);

}  // namespace bhs
