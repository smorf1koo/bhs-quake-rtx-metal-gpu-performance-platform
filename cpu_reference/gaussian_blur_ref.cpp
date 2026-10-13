// gaussian_blur_ref.cpp — CPU reference implementation (see the specification in the header).
// Deliberately simple and single-threaded: correctness matters here, not speed.
#include "gaussian_blur_ref.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace bhs {

int GaussianRadius(float sigma) {
    return std::max(1, (int)std::ceil(3.0f * sigma));
}

std::vector<float> GaussianKernel(float sigma) {
    const int r = GaussianRadius(sigma);
    std::vector<double> w(2 * r + 1);
    double sum = 0.0;
    for (int i = -r; i <= r; ++i) {
        w[i + r] = std::exp(-(double)i * i / (2.0 * (double)sigma * sigma));
        sum += w[i + r];
    }
    std::vector<float> out(w.size());
    for (size_t i = 0; i < w.size(); ++i) out[i] = (float)(w[i] / sum);
    return out;
}

void GaussianBlurRef(const uint8_t* src, uint8_t* dst, int width, int height, int channels, float sigma) {
    const std::vector<float> k = GaussianKernel(sigma);
    const int r = GaussianRadius(sigma);
    const size_t n = (size_t)width * height * channels;
    std::vector<float> tmp(n);

    // Horizontal pass: src (uint8) -> tmp (float).
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                float acc = 0.0f;
                for (int i = -r; i <= r; ++i) {
                    const int xx = std::clamp(x + i, 0, width - 1);
                    acc += k[i + r] * (float)src[((size_t)y * width + xx) * channels + c];
                }
                tmp[((size_t)y * width + x) * channels + c] = acc;
            }
        }
    }
    // Vertical pass: tmp (float) -> dst (uint8).
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                float acc = 0.0f;
                for (int i = -r; i <= r; ++i) {
                    const int yy = std::clamp(y + i, 0, height - 1);
                    acc += k[i + r] * tmp[((size_t)yy * width + x) * channels + c];
                }
                const float v = std::floor(acc + 0.5f);
                dst[((size_t)y * width + x) * channels + c] = (uint8_t)std::clamp(v, 0.0f, 255.0f);
            }
        }
    }
}

CompareResult CompareImages(const uint8_t* a, const uint8_t* b, int width, int height, int channels,
                            int tolerance) {
    CompareResult res;
    double sum = 0.0;
    const size_t pixels = (size_t)width * height;
    for (size_t p = 0; p < pixels; ++p) {
        bool over = false;
        for (int c = 0; c < channels; ++c) {
            const int d = std::abs((int)a[p * channels + c] - (int)b[p * channels + c]);
            res.maxAbsDiff = std::max(res.maxAbsDiff, d);
            sum += d;
            over |= d > tolerance;
        }
        res.pixelsOverTolerance += over ? 1 : 0;
    }
    res.meanAbsDiff = sum / (double)(pixels * channels);
    return res;
}

}  // namespace bhs
