#include "libobsensor/ObSensor.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

void write_ppm(const std::string &path, const uint8_t *rgb, uint32_t width, uint32_t height) {
    std::ofstream out(path, std::ios::binary);
    if(!out) throw std::runtime_error("cannot open color output: " + path);
    out << "P6\n" << width << " " << height << "\n255\n";
    out.write(reinterpret_cast<const char *>(rgb), static_cast<std::streamsize>(width) * height * 3);
}

void write_raw(const std::string &path, const uint16_t *depth, uint32_t width, uint32_t height) {
    std::ofstream out(path, std::ios::binary);
    if(!out) throw std::runtime_error("cannot open depth output: " + path);
    out.write(reinterpret_cast<const char *>(depth), static_cast<std::streamsize>(width) * height * sizeof(uint16_t));
}

std::shared_ptr<ob::ColorFrame> to_rgb(std::shared_ptr<ob::ColorFrame> frame, ob::FormatConvertFilter &filter) {
    if(frame->format() == OB_FORMAT_RGB) return frame;
    if(frame->format() == OB_FORMAT_MJPG) filter.setFormatConvertType(FORMAT_MJPG_TO_RGB);
    else if(frame->format() == OB_FORMAT_UYVY) filter.setFormatConvertType(FORMAT_UYVY_TO_RGB);
    else if(frame->format() == OB_FORMAT_YUYV) filter.setFormatConvertType(FORMAT_YUYV_TO_RGB);
    else throw std::runtime_error("unsupported color format: " + std::to_string(frame->format()));
    auto converted = filter.process(frame);
    if(!converted) throw std::runtime_error("color conversion returned null");
    return converted->as<ob::ColorFrame>();
}

} // namespace

int main(int argc, char **argv) try {
    if(argc != 3) {
        std::cerr << "usage: orbbec_v1_capture OUTPUT_PREFIX FRAME_COUNT\n";
        return 2;
    }
    const std::string prefix = argv[1];
    const int requested = std::stoi(argv[2]);
    if(requested < 1 || requested > 100) throw std::runtime_error("FRAME_COUNT must be 1..100");

    ob::Pipeline pipeline;
    auto config = std::make_shared<ob::Config>();
    auto color_profiles = pipeline.getStreamProfileList(OB_SENSOR_COLOR);
    auto depth_profiles = pipeline.getStreamProfileList(OB_SENSOR_DEPTH);

    // DaBai DCW2 legacy firmware exposes calibrated D2C at 640x360.  This is an
    // exact 1/3 scale of the 1920x1080 RGB calibration used by the Python code.
    auto color = color_profiles->getVideoStreamProfile(640, 360, OB_FORMAT_ANY, 30);
    auto depth = std::const_pointer_cast<ob::StreamProfile>(
        depth_profiles->getProfile(OB_PROFILE_DEFAULT))->as<ob::VideoStreamProfile>();
    config->enableStream(color);
    config->enableStream(depth);
    config->setAlignMode(ALIGN_D2C_SW_MODE);
    try { pipeline.enableFrameSync(); } catch(...) {}
    pipeline.start(config);

    ob::FormatConvertFilter converter;
    int saved = 0;
    int attempts = 0;
    while(saved < requested && attempts++ < requested * 20 + 30) {
        auto frames = pipeline.waitForFrames(1500);
        if(!frames || attempts <= 8) continue;
        auto color_frame = frames->colorFrame();
        auto depth_frame = frames->depthFrame();
        if(!color_frame || !depth_frame) continue;
        color_frame = to_rgb(color_frame, converter);
        if(color_frame->width() != 640 || color_frame->height() != 360)
            throw std::runtime_error("color frame is not the required 640x360 D2C profile");
        if(depth_frame->width() != color_frame->width() || depth_frame->height() != color_frame->height())
            throw std::runtime_error("D2C depth dimensions do not match color dimensions");

        const std::string stem = prefix + "_" + std::to_string(saved);
        write_ppm(stem + ".ppm", static_cast<const uint8_t *>(color_frame->data()), color_frame->width(), color_frame->height());
        write_raw(stem + ".depth16", static_cast<const uint16_t *>(depth_frame->data()), depth_frame->width(), depth_frame->height());
        std::ofstream meta(stem + ".txt");
        meta << "width=" << depth_frame->width() << "\n"
             << "height=" << depth_frame->height() << "\n"
             << "depth_scale_mm=" << depth_frame->getValueScale() << "\n"
             << "color_timestamp_ms=" << color_frame->timeStamp() << "\n"
             << "depth_timestamp_ms=" << depth_frame->timeStamp() << "\n";
        ++saved;
    }
    pipeline.stop();
    if(saved != requested) throw std::runtime_error("timed out before requested aligned frames were captured");
    std::cout << "captured=" << saved << " color=" << color->width() << "x" << color->height()
              << " depth_aligned=640x360\n";
    return 0;
}
catch(const ob::Error &e) {
    std::cerr << "Orbbec error in " << e.getName() << ": " << e.getMessage() << " (" << e.getArgs() << ")\n";
    return 1;
}
catch(const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
}
