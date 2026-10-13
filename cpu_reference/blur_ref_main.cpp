// blur_ref_main.cpp — command-line tool for the reference blur.
//   blur_ref input.ppm output.ppm [sigma]          (sigma defaults to 4.0)
//   blur_ref --compare a.ppm b.ppm [tolerance]      (tolerance defaults to 1)
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "gaussian_blur_ref.h"
#include "ppm_io.h"

int main(int argc, char** argv) {
    if (argc >= 4 && std::strcmp(argv[1], "--compare") == 0) {
        bhs::ImageRGB8 a, b;
        if (!bhs::ReadPPM(argv[2], a) || !bhs::ReadPPM(argv[3], b)) return 2;
        if (a.width != b.width || a.height != b.height) { std::fprintf(stderr, "size mismatch\n"); return 2; }
        const int tol = argc >= 5 ? std::atoi(argv[4]) : 1;
        auto r = bhs::CompareImages(a.pixels.data(), b.pixels.data(), a.width, a.height, 3, tol);
        std::printf("max_abs_diff=%d mean_abs_diff=%.5f pixels_over_tol=%zu (tolerance %d)\n",
                    r.maxAbsDiff, r.meanAbsDiff, r.pixelsOverTolerance, tol);
        return r.maxAbsDiff <= tol ? 0 : 1;
    }
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s input.ppm output.ppm [sigma]\n"
                             "       %s --compare a.ppm b.ppm [tolerance]\n", argv[0], argv[0]);
        return 2;
    }
    const float sigma = argc >= 4 ? std::strtof(argv[3], nullptr) : 4.0f;
    bhs::ImageRGB8 in;
    if (!bhs::ReadPPM(argv[1], in)) return 2;
    bhs::ImageRGB8 out = in;
    auto t0 = std::chrono::steady_clock::now();
    bhs::GaussianBlurRef(in.pixels.data(), out.pixels.data(), in.width, in.height, 3, sigma);
    auto t1 = std::chrono::steady_clock::now();
    std::printf("%dx%d sigma=%.3f radius=%d cpu_ms=%.1f (single-threaded reference, not a benchmark)\n",
                in.width, in.height, sigma, bhs::GaussianRadius(sigma),
                std::chrono::duration<double, std::milli>(t1 - t0).count());
    return bhs::WritePPM(argv[2], out) ? 0 : 2;
}
