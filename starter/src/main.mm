// main.mm — Metal compute skeleton for test task 03 ("Mac" track).
//
// Already done:
//   * argument parsing, MTLDevice / MTLCommandQueue, runtime compilation of
//     kernels.metal (Command Line Tools are enough), pipeline by variant name;
//   * image loading (PNG via ImageIO or PPM) into RGBA8, GPU buffers
//     (input, output, float4 intermediate for two-pass variants, weights);
//   * CPU reference (cpu_reference/, bhs::GaussianBlurRef) and comparison;
//   * dispatch of a single-pass variant (RunOnce) and one results.csv row.
//
// Your part:
//   * naive kernel in kernels.metal;
//   * one optimized variant (separable two-pass OR threadgroup-memory tiling);
//     a two-pass variant needs its own encoding in RunOnce (TODO(optimized));
//   * GPU timing: warm-up, N measured runs, median / min / max (TODO(timing));
//   * the comparison with the reference and the CSV row are already wired:
//     once the above works, check that they report what you expect.
//
// Example (from the package root, after building):
//   ./build/blur_metal --variant naive --csv results.csv
//   ./build/blur_metal --variant separable --csv results.csv --output out.ppm
// You may rewrite the skeleton completely (metal-cpp, Swift, Xcode project).

#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gaussian_blur_ref.h"
#include "ppm_io.h"

#ifndef BHS_KERNELS_PATH
#define BHS_KERNELS_PATH "starter/src/kernels.metal"  // relative to the package root
#endif
#ifndef BHS_ASSET_DIR
#define BHS_ASSET_DIR "assets"
#endif

namespace {

struct Options {
    std::string input = BHS_ASSET_DIR "/test_4k.png";
    std::string variant = "naive";  // kernel function name in kernels.metal (see --list)
    std::string csv;                 // empty: do not write CSV
    std::string output;              // save the GPU result as PPM (for tools/image_tools.py compare)
    std::string notes;               // free text for the CSV "notes" column
    float sigma = 4.0f;
    int warmup = 5;
    int runs = 50;
    int tgx = 16, tgy = 16;          // threadgroup size
    bool list = false;
};

void Usage(const char* argv0) {
    std::fprintf(stderr,
        "usage: %s [--input file.png|file.ppm] [--variant name] [--sigma s] [--warmup n] [--runs n]\n"
        "          [--tg X Y] [--csv results.csv] [--output out.ppm] [--notes text] [--list]\n", argv0);
}

bool ParseArgs(int argc, const char** argv, Options& o) {
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
        else if (a == "--tg") { o.tgx = std::atoi(next("--tg")); o.tgy = std::atoi(next("--tg")); }
        else if (a == "--list") o.list = true;
        else { Usage(argv[0]); return false; }
    }
    return true;
}

struct ImageRGBA8 {
    int width = 0, height = 0;
    std::vector<uint8_t> pixels;  // width * height * 4, rows top to bottom, alpha = 255
};

// PNG through ImageIO (raw decoded bytes, no color management) or binary PPM.
bool LoadImage(const std::string& path, ImageRGBA8& img) {
    if (path.size() > 4 && path.substr(path.size() - 4) == ".ppm") {
        bhs::ImageRGB8 rgb;
        if (!bhs::ReadPPM(path, rgb)) return false;
        img.width = rgb.width; img.height = rgb.height;
        img.pixels.resize((size_t)img.width * img.height * 4);
        for (size_t i = 0; i < (size_t)img.width * img.height; ++i) {
            std::memcpy(&img.pixels[i * 4], &rgb.pixels[i * 3], 3);
            img.pixels[i * 4 + 3] = 255;
        }
        return true;
    }
    NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
    CGImageSourceRef src = CGImageSourceCreateWithURL((__bridge CFURLRef)url, nullptr);
    if (!src) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    CGImageRef cg = CGImageSourceCreateImageAtIndex(src, 0, nullptr);
    CFRelease(src);
    if (!cg) { std::fprintf(stderr, "cannot decode %s\n", path.c_str()); return false; }
    const size_t bpc = CGImageGetBitsPerComponent(cg), bpp = CGImageGetBitsPerPixel(cg);
    const size_t stride = CGImageGetBytesPerRow(cg);
    img.width = (int)CGImageGetWidth(cg);
    img.height = (int)CGImageGetHeight(cg);
    CFDataRef data = CGDataProviderCopyData(CGImageGetDataProvider(cg));
    CGImageRelease(cg);
    if (!data || bpc != 8 || (bpp != 24 && bpp != 32)) {
        std::fprintf(stderr, "%s: unsupported pixel format (%zu bpc, %zu bpp)\n", path.c_str(), bpc, bpp);
        if (data) CFRelease(data);
        return false;
    }
    const uint8_t* p = CFDataGetBytePtr(data);
    img.pixels.resize((size_t)img.width * img.height * 4);
    for (int y = 0; y < img.height; ++y)
        for (int x = 0; x < img.width; ++x) {
            const uint8_t* s = p + (size_t)y * stride + (size_t)x * (bpp / 8);
            uint8_t* d = &img.pixels[((size_t)y * img.width + x) * 4];
            d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = 255;
        }
    CFRelease(data);
    return true;
}

// Must match BlurParams in kernels.metal.
struct BlurParams {
    uint32_t width;
    uint32_t height;
    int32_t radius;
};

// Measurement result for one variant. Filled in TODO(timing).
struct Timing {
    std::vector<double> samplesMs;  // one value per measured run
    double medianMs = 0, minMs = 0, maxMs = 0, stddevMs = 0;
    std::string method = "TODO";    // e.g. "MTLCommandBuffer.GPUStartTime/GPUEndTime"
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
    NSString* os = [[NSProcessInfo processInfo] operatingSystemVersionString];
    std::fprintf(f, "\"%s\",\"%s\",metal,%s,%d,%d,%.3f,%d,%dx%d,%d,%d,%.4f,%.4f,%.4f,%.4f,\"%s\",%d,%zu,\"%s\"\n",
                 device.c_str(), os.UTF8String, o.variant.c_str(), width, height, o.sigma,
                 bhs::GaussianRadius(o.sigma), o.tgx, o.tgy, o.warmup, (int)t.samplesMs.size(), t.medianMs,
                 t.minMs, t.maxMs, t.stddevMs, t.method.c_str(), maxAbsDiff, mismatched, o.notes.c_str());
    std::fclose(f);
}

}  // namespace

int main(int argc, const char** argv) {
    @autoreleasepool {
        Options opt;
        if (!ParseArgs(argc, argv, opt)) return 2;

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { std::fprintf(stderr, "Metal is not available\n"); return 1; }
        id<MTLCommandQueue> queue = [device newCommandQueue];

        NSError* error = nil;
        NSString* src = [NSString stringWithContentsOfFile:@BHS_KERNELS_PATH encoding:NSUTF8StringEncoding error:&error];
        if (!src) { NSLog(@"cannot read %s: %@", BHS_KERNELS_PATH, error); return 1; }
        MTLCompileOptions* copts = [MTLCompileOptions new];
        id<MTLLibrary> lib = [device newLibraryWithSource:src options:copts error:&error];
        if (!lib) { NSLog(@"kernels.metal failed to compile: %@", error); return 1; }
        if (opt.list) {
            for (NSString* name in lib.functionNames) std::printf("  %s\n", name.UTF8String);
            return 0;
        }

        // A two-pass variant "foo" is expected as two kernels "foo_h" and "foo_v" (see TODO(optimized)).
        auto makePso = [&](const std::string& name) -> id<MTLComputePipelineState> {
            id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
            if (!fn) return nil;
            NSError* e = nil;
            id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:fn error:&e];
            if (!p) NSLog(@"pipeline %s: %@", name.c_str(), e);
            return p;
        };
        id<MTLComputePipelineState> pso = makePso(opt.variant);
        id<MTLComputePipelineState> psoH = makePso(opt.variant + "_h");
        id<MTLComputePipelineState> psoV = makePso(opt.variant + "_v");
        const bool twoPass = !pso && psoH && psoV;
        if (!pso && !twoPass) {
            std::fprintf(stderr, "no kernel '%s' (or '%s_h' + '%s_v'); use --list\n", opt.variant.c_str(),
                         opt.variant.c_str(), opt.variant.c_str());
            return 1;
        }
        id<MTLComputePipelineState> anyPso = twoPass ? psoH : pso;
        std::printf("device: %s\nvariant: %s (%s), threadExecutionWidth=%lu, maxTotalThreadsPerThreadgroup=%lu\n",
                    device.name.UTF8String, opt.variant.c_str(), twoPass ? "two-pass" : "single-pass",
                    (unsigned long)anyPso.threadExecutionWidth, (unsigned long)anyPso.maxTotalThreadsPerThreadgroup);

        // ---- Data -----------------------------------------------------------
        ImageRGBA8 image;
        if (!LoadImage(opt.input, image)) return 1;
        const int w = image.width, h = image.height;
        const size_t bytes = image.pixels.size();
        const std::vector<float> weights = bhs::GaussianKernel(opt.sigma);  // 2r+1 values, center at r
        BlurParams params{(uint32_t)w, (uint32_t)h, bhs::GaussianRadius(opt.sigma)};
        std::printf("image: %dx%d RGBA8, sigma=%.2f, radius=%d\n", w, h, opt.sigma, params.radius);

        id<MTLBuffer> inBuf = [device newBufferWithBytes:image.pixels.data() length:bytes
                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> outBuf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> tmpBuf = [device newBufferWithLength:(size_t)w * h * 4 * sizeof(float)
                                                   options:MTLResourceStorageModePrivate];  // float4 per pixel
        id<MTLBuffer> wBuf = [device newBufferWithBytes:weights.data() length:weights.size() * sizeof(float)
                                                options:MTLResourceStorageModeShared];

        // Buffer indices used by kernels.metal: 0 params, 1 src, 2 dst, 3 weights, 4 tmp.
        const MTLSize grid = MTLSizeMake((NSUInteger)w, (NSUInteger)h, 1);
        const MTLSize tg = MTLSizeMake((NSUInteger)opt.tgx, (NSUInteger)opt.tgy, 1);
        auto RunOnce = [&]() -> id<MTLCommandBuffer> {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setBytes:&params length:sizeof params atIndex:0];
            [enc setBuffer:inBuf offset:0 atIndex:1];
            [enc setBuffer:outBuf offset:0 atIndex:2];
            [enc setBuffer:wBuf offset:0 atIndex:3];
            [enc setBuffer:tmpBuf offset:0 atIndex:4];
            if (!twoPass) {
                [enc setComputePipelineState:pso];
                [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            } else {
                // TODO(optimized, two-pass variant): dispatch psoH (src -> tmp), then psoV (tmp -> dst).
                // Does the second dispatch see the results of the first one? Check the
                // Metal docs for compute encoders (dispatch type, memory barriers).
            }
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            if (cmd.error) NSLog(@"GPU error: %@", cmd.error);
            return cmd;
        };

        // ---- TODO(timing): measurement ------------------------------------------
        // * opt.warmup runs that are NOT measured, then opt.runs measured runs;
        // * GPU time of one run: cmd.GPUEndTime - cmd.GPUStartTime (seconds);
        //   do not use a CPU timer around commit/wait as "GPU time";
        // * fill timing.samplesMs, median / min / max / stddev and timing.method.
        Timing timing;
        RunOnce();  // placeholder: a single unmeasured run so that the check below has data

        // ---- correctness (already wired) -----------------------------------
        std::vector<uint8_t> ref(bytes);
        bhs::GaussianBlurRef(image.pixels.data(), ref.data(), w, h, 4, opt.sigma);
        const auto cmp = bhs::CompareImages((const uint8_t*)outBuf.contents, ref.data(), w, h, 4, 1);
        std::printf("check vs reference: max|diff|=%d, mean|diff|=%.4f, pixels over tolerance=%zu -> %s\n",
                    cmp.maxAbsDiff, cmp.meanAbsDiff, cmp.pixelsOverTolerance, cmp.maxAbsDiff <= 1 ? "OK" : "FAIL");

        if (!opt.output.empty()) {
            bhs::ImageRGB8 out;
            out.width = w; out.height = h; out.pixels.resize((size_t)w * h * 3);
            const uint8_t* gpu = (const uint8_t*)outBuf.contents;
            for (size_t i = 0; i < (size_t)w * h; ++i) std::memcpy(&out.pixels[i * 3], &gpu[i * 4], 3);
            bhs::WritePPM(opt.output, out);
        }
        if (!opt.csv.empty())
            AppendCsv(opt.csv, device.name.UTF8String, opt, w, h, timing, cmp.maxAbsDiff, cmp.pixelsOverTolerance);
        std::printf("median %.4f ms (min %.4f, max %.4f) over %zu runs [%s]\n", timing.medianMs, timing.minMs,
                    timing.maxMs, timing.samplesMs.size(), timing.method.c_str());
    }
    return 0;
}
