// ppm_io.h — binary PPM (P6, 8-bit, RGB) reading and writing. Header-only.
// PNG -> PPM: python3 -I tools/image_tools.py to-ppm assets/test_4k.png test_4k.ppm
// On a Mac you may also read the PNG directly with ImageIO / MTKTextureLoader.
#pragma once

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace bhs {

struct ImageRGB8 {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> pixels;  // width * height * 3, rows top to bottom
};

inline bool ReadPPM(const std::string& path, ImageRGB8& img) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "ReadPPM: cannot open %s\n", path.c_str()); return false; }
    auto token = [&](std::string& out) {
        out.clear();
        int ch = std::fgetc(f);
        while (ch != EOF && (std::isspace(ch) || ch == '#')) {
            if (ch == '#') while (ch != EOF && ch != '\n') ch = std::fgetc(f);
            ch = std::fgetc(f);
        }
        while (ch != EOF && !std::isspace(ch)) { out.push_back((char)ch); ch = std::fgetc(f); }
        return !out.empty();  // the single whitespace after the token is already consumed
    };
    std::string magic, w, h, maxval;
    bool ok = token(magic) && token(w) && token(h) && token(maxval) && magic == "P6" && maxval == "255";
    if (ok) {
        img.width = std::stoi(w);
        img.height = std::stoi(h);
        img.pixels.resize((size_t)img.width * img.height * 3);
        ok = std::fread(img.pixels.data(), 1, img.pixels.size(), f) == img.pixels.size();
    }
    std::fclose(f);
    if (!ok) std::fprintf(stderr, "ReadPPM: %s is not a valid binary P6/255 file\n", path.c_str());
    return ok;
}

inline bool WritePPM(const std::string& path, const ImageRGB8& img) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "WritePPM: cannot open %s\n", path.c_str()); return false; }
    std::fprintf(f, "P6\n%d %d\n255\n", img.width, img.height);
    bool ok = std::fwrite(img.pixels.data(), 1, img.pixels.size(), f) == img.pixels.size();
    std::fclose(f);
    return ok;
}

}  // namespace bhs
