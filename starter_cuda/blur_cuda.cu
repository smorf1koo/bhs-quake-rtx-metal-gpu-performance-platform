// blur_cuda.cu — CUDA skeleton for test task 03 ("no Mac" track).
//
// Already done:
//   * argument parsing, device selection, error checking (CUDA_CHECK);
//   * image loading (binary PPM) into RGBA8, GPU buffers
//     (input, output, float4 intermediate for the two-pass variant, weights);
//   * CPU reference (cpu_reference/, bhs::GaussianBlurRef) and comparison;
//   * launch of the single-pass variant (RunOnce), cudaEvent objects,
//     copy of the result back to the CPU and one results.csv row.
//
// Your part:
//   * kernel "naive" below;
//   * ONE optimized variant: "separable" (kernels separable_h + separable_v,
//     launch in RunOnce, TODO(optimized)) or your own tiled kernel in shared memory;
//   * GPU timing: warm-up, N measured runs, median / min / max (TODO(timing)).
//
// Build (from the package root; works the same in Google Colab):
//   python3 -I tools/image_tools.py to-ppm assets/test_4k.png test_4k.ppm
//   nvcc -O3 -std=c++17 -Icpu_reference -o blur_cuda \
//        starter_cuda/blur_cuda.cu cpu_reference/gaussian_blur_ref.cpp
//   ./blur_cuda --variant naive --csv results.csv
//   ./blur_cuda --variant separable --csv results.csv --output out.ppm
// Measure only a build without -G (device debug info disables optimizations).

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/utsname.h>
#endif

#include "gaussian_blur_ref.h"
#include "ppm_io.h"

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call,    \
                         cudaGetErrorString(err_));                               \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)

struct BlurParams {
    int width;
    int height;
    int radius;
};

// ---- Kernels ----------------------------------------------------------------
// The correct result is defined by cpu_reference/gaussian_blur_ref.h:
// clamp-to-edge, r = ceil(3 * sigma), normalized weights (2r+1 floats, center
// at index r), rounding floor(v + 0.5), tolerance max |diff| <= 1 per channel.

__global__ void naive(BlurParams p, const uchar4* src, uchar4* dst, const float* weights)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.width || y >= p.height) return;
    // TODO(naive): direct 2D convolution, (2r+1)^2 reads per pixel, one pass,
    //   weight of (i, j) = weights[r + i] * weights[r + j]; clamp coordinates
    //   to the image; round and write the result. Placeholder: copy the input.
    (void)weights;
    dst[y * p.width + x] = src[y * p.width + x];
}

__global__ void separable_h(BlurParams p, const uchar4* src, float4* tmp, const float* weights)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.width || y >= p.height) return;
    // TODO(optimized): horizontal pass, src -> tmp (float, no rounding).
    (void)weights;
    const uchar4 s = src[y * p.width + x];
    tmp[y * p.width + x] = make_float4(s.x, s.y, s.z, s.w);
}

__global__ void separable_v(BlurParams p, const float4* tmp, uchar4* dst, const float* weights)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.width || y >= p.height) return;
    // TODO(optimized): vertical pass, tmp -> dst, rounding floor(v + 0.5).
    (void)weights;
    const float4 t = tmp[y * p.width + x];
    dst[y * p.width + x] = make_uchar4((unsigned char)t.x, (unsigned char)t.y,
                                       (unsigned char)t.z, (unsigned char)t.w);
}

// ---- Host -------------------------------------------------------------------

namespace {

struct Options {
    std::string input = "test_4k.ppm";
    std::string variant = "naive";  // naive | separable
    std::string csv;                 // empty: do not write CSV
    std::string output;              // save the GPU result as PPM (for tools/image_tools.py compare)
    std::string notes;               // free text for the CSV "notes" column
    float sigma = 4.0f;
    int warmup = 5;
    int runs = 50;
    int bx = 16, by = 16;            // block size (threads per block)
};

void Usage(const char* argv0) {
    std::fprintf(stderr,
        "usage: %s [--input file.ppm] [--variant naive|separable] [--sigma s] [--warmup n] [--runs n]\n"
        "          [--block X Y] [--csv results.csv] [--output out.ppm] [--notes text]\n", argv0);
}

bool ParseArgs(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", name); std::exit(2); }
            return argv[++i];
        };
        if (a == "--input") o.input = next("--input");
        else if (a == "--variant") o.variant = next("--variant");
        else if (a == "--csv") o.csv = next("--csv");
        else if (a == "--output") o.output = next("--output");
        else if (a == "--notes") o.notes = next("--notes");
        else if (a == "--sigma") o.sigma = std::strtof(next("--sigma"), nullptr);
        else if (a == "--warmup") o.warmup = std::atoi(next("--warmup"));
        else if (a == "--runs") o.runs = std::atoi(next("--runs"));
        else if (a == "--block") { o.bx = std::atoi(next("--block")); o.by = std::atoi(next("--block")); }
        else { Usage(argv[0]); return false; }
    }
    return true;
}

std::string OsName() {
#if defined(__unix__) || defined(__APPLE__)
    utsname u{};
    if (uname(&u) == 0) return std::string(u.sysname) + " " + u.release;
#endif
#ifdef _WIN32
    return "Windows";
#else
    return "unknown";
#endif
}

// Measurement result for one variant. Filled in TODO(timing).
struct Timing {
    std::vector<double> samplesMs;  // one value per measured run
    double medianMs = 0, minMs = 0, maxMs = 0, stddevMs = 0;
    std::string method = "TODO";    // e.g. "cudaEventElapsedTime"
};

void AppendCsv(const std::string& path, const std::string& device, const Options& o, int width, int height,
               const Timing& t, int maxAbsDiff, size_t mismatched) {
    FILE* probe = std::fopen(path.c_str(), "r");
    bool needHeader = probe == nullptr;
    if (probe) {
        std::fseek(probe, 0, SEEK_END);
        needHeader = std::ftell(probe) == 0;
        std::fclose(probe);
    }
    FILE* f = std::fopen(path.c_str(), "a");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return; }
    if (needHeader)
        std::fprintf(f, "device,os,api,variant,width,height,sigma,radius,threadgroup,warmup_runs,measured_runs,"
                        "median_ms,min_ms,max_ms,stddev_ms,timing_method,max_abs_diff,mismatched_pixels,notes\n");
    std::fprintf(f, "\"%s\",\"%s\",cuda,%s,%d,%d,%.3f,%d,%dx%d,%d,%d,%.4f,%.4f,%.4f,%.4f,\"%s\",%d,%zu,\"%s\"\n",
                 device.c_str(), OsName().c_str(), o.variant.c_str(), width, height, o.sigma,
                 bhs::GaussianRadius(o.sigma), o.bx, o.by, o.warmup, (int)t.samplesMs.size(), t.medianMs,
                 t.minMs, t.maxMs, t.stddevMs, t.method.c_str(), maxAbsDiff, mismatched, o.notes.c_str());
    std::fclose(f);
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    if (!ParseArgs(argc, argv, opt)) return 2;
    const bool twoPass = opt.variant == "separable";
    if (opt.variant != "naive" && !twoPass) {
        std::fprintf(stderr, "unknown variant '%s' (naive | separable)\n", opt.variant.c_str());
        return 2;
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    CUDA_CHECK(cudaSetDevice(0));
    std::printf("device: %s (sm_%d%d), warpSize=%d, maxThreadsPerBlock=%d\nvariant: %s (%s)\n",
                prop.name, prop.major, prop.minor, prop.warpSize, prop.maxThreadsPerBlock,
                opt.variant.c_str(), twoPass ? "two-pass" : "single-pass");

    // ---- Data ---------------------------------------------------------------
    bhs::ImageRGB8 rgb;
    if (!bhs::ReadPPM(opt.input, rgb)) {
        std::fprintf(stderr, "convert the PNG first: python3 -I tools/image_tools.py to-ppm "
                             "assets/test_4k.png test_4k.ppm\n");
        return 1;
    }
    const int w = rgb.width, h = rgb.height;
    const size_t pixels = (size_t)w * h;
    std::vector<uint8_t> image(pixels * 4);  // RGBA8, alpha = 255
    for (size_t i = 0; i < pixels; ++i) {
        std::memcpy(&image[i * 4], &rgb.pixels[i * 3], 3);
        image[i * 4 + 3] = 255;
    }
    const std::vector<float> weights = bhs::GaussianKernel(opt.sigma);  // 2r+1 values, center at r
    const BlurParams params{w, h, bhs::GaussianRadius(opt.sigma)};
    std::printf("image: %dx%d RGBA8, sigma=%.2f, radius=%d\n", w, h, opt.sigma, params.radius);

    uchar4 *dSrc = nullptr, *dDst = nullptr;
    float4* dTmp = nullptr;
    float* dWeights = nullptr;
    CUDA_CHECK(cudaMalloc(&dSrc, pixels * sizeof(uchar4)));
    CUDA_CHECK(cudaMalloc(&dDst, pixels * sizeof(uchar4)));
    CUDA_CHECK(cudaMalloc(&dTmp, pixels * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&dWeights, weights.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dSrc, image.data(), pixels * sizeof(uchar4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dWeights, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice));

    const dim3 block((unsigned)opt.bx, (unsigned)opt.by);
    const dim3 grid((unsigned)((w + opt.bx - 1) / opt.bx), (unsigned)((h + opt.by - 1) / opt.by));

    // Enqueues one run on the default stream (asynchronous, does not wait).
    auto RunOnce = [&]() {
        if (!twoPass) {
            naive<<<grid, block>>>(params, dSrc, dDst, dWeights);
        } else {
            // TODO(optimized): launch separable_h (src -> tmp), then separable_v (tmp -> dst).
            // Does the second kernel see the results of the first one? Why?
        }
        CUDA_CHECK(cudaGetLastError());
    };

    // ---- TODO(timing): measurement --------------------------------------------
    // * opt.warmup runs that are NOT measured, then opt.runs measured runs;
    // * GPU time of one run: cudaEventRecord(start) before RunOnce(),
    //   cudaEventRecord(stop) after it, cudaEventSynchronize(stop),
    //   cudaEventElapsedTime(&ms, start, stop);
    //   do not use a CPU timer around the launch as "GPU time";
    // * fill timing.samplesMs, median / min / max / stddev and timing.method.
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    Timing timing;
    RunOnce();  // placeholder: a single unmeasured run so that the check below has data
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Correctness (already wired) ------------------------------------------
    std::vector<uint8_t> gpu(pixels * 4);
    CUDA_CHECK(cudaMemcpy(gpu.data(), dDst, pixels * sizeof(uchar4), cudaMemcpyDeviceToHost));
    std::vector<uint8_t> ref(pixels * 4);
    bhs::GaussianBlurRef(image.data(), ref.data(), w, h, 4, opt.sigma);
    const auto cmp = bhs::CompareImages(gpu.data(), ref.data(), w, h, 4, 1);
    std::printf("check vs reference: max|diff|=%d, mean|diff|=%.4f, pixels over tolerance=%zu -> %s\n",
                cmp.maxAbsDiff, cmp.meanAbsDiff, cmp.pixelsOverTolerance, cmp.maxAbsDiff <= 1 ? "OK" : "FAIL");

    if (!opt.output.empty()) {
        bhs::ImageRGB8 out;
        out.width = w; out.height = h; out.pixels.resize(pixels * 3);
        for (size_t i = 0; i < pixels; ++i) std::memcpy(&out.pixels[i * 3], &gpu[i * 4], 3);
        bhs::WritePPM(opt.output, out);
    }
    if (!opt.csv.empty())
        AppendCsv(opt.csv, prop.name, opt, w, h, timing, cmp.maxAbsDiff, cmp.pixelsOverTolerance);
    std::printf("median %.4f ms (min %.4f, max %.4f) over %zu runs [%s]\n", timing.medianMs, timing.minMs,
                timing.maxMs, timing.samplesMs.size(), timing.method.c_str());

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dSrc));
    CUDA_CHECK(cudaFree(dDst));
    CUDA_CHECK(cudaFree(dTmp));
    CUDA_CHECK(cudaFree(dWeights));
    return 0;
}
