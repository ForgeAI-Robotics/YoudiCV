#include "libobsensor/ObSensor.hpp"
#include "exposure_saturation_recovery.hpp"

#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

volatile std::sig_atomic_t running = 1;

void stop_running(int) { running = 0; }

template <typename T>
void write_value(const T &value) {
    std::cout.write(reinterpret_cast<const char *>(&value), sizeof(T));
}

std::shared_ptr<ob::ColorFrame> to_rgb(std::shared_ptr<ob::ColorFrame> frame,
                                       ob::FormatConvertFilter &filter) {
    if(frame->format() == OB_FORMAT_RGB) return frame;
    if(frame->format() == OB_FORMAT_MJPG)
        filter.setFormatConvertType(FORMAT_MJPG_TO_RGB);
    else if(frame->format() == OB_FORMAT_UYVY)
        filter.setFormatConvertType(FORMAT_UYVY_TO_RGB);
    else if(frame->format() == OB_FORMAT_YUYV)
        filter.setFormatConvertType(FORMAT_YUYV_TO_RGB);
    else
        throw std::runtime_error("unsupported color format: " +
                                 std::to_string(frame->format()));
    auto converted = filter.process(frame);
    if(!converted) throw std::runtime_error("color conversion returned null");
    return converted->as<ob::ColorFrame>();
}

struct RoiExposureController {
    std::shared_ptr<ob::Device> device;
    int exposure = 157;
    int gain = 0;
    int exposure_min = 1;
    int exposure_max = 260;  // 约 26 ms：避免曝光造成明显的感知延迟。
    int gain_min = 0;
    int gain_max = 48;
    int settle_frames = 0;

    explicit RoiExposureController(std::shared_ptr<ob::Device> camera)
        : device(std::move(camera)) {
        if(!device->isPropertySupported(OB_PROP_COLOR_EXPOSURE_INT,
                                        OB_PERMISSION_WRITE) ||
           !device->isPropertySupported(OB_PROP_COLOR_GAIN_INT,
                                        OB_PERMISSION_WRITE)) {
            std::cerr << "ROI exposure disabled: manual exposure/gain unsupported\n";
            device.reset();
            return;
        }
        const auto er = device->getIntPropertyRange(OB_PROP_COLOR_EXPOSURE_INT);
        const auto gr = device->getIntPropertyRange(OB_PROP_COLOR_GAIN_INT);
        exposure_min = er.min;
        exposure_max = std::min<int>(er.max, exposure_max);
        gain_min = gr.min;
        gain_max = std::min<int>(gr.max, gain_max);
        exposure = std::clamp<int>(er.def / 4, exposure_min, exposure_max);
        gain = gain_min;
        if(device->isPropertySupported(OB_PROP_COLOR_AUTO_EXPOSURE_BOOL,
                                       OB_PERMISSION_WRITE))
            device->setBoolProperty(OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, false);
        device->setIntProperty(OB_PROP_COLOR_EXPOSURE_INT, exposure);
        device->setIntProperty(OB_PROP_COLOR_GAIN_INT, gain);
        verify_properties("startup");
    }

    void verify_properties(const char *reason) {
        const int actual_exposure = device->getIntProperty(OB_PROP_COLOR_EXPOSURE_INT);
        const int actual_gain = device->getIntProperty(OB_PROP_COLOR_GAIN_INT);
        const bool actual_auto = device->getBoolProperty(OB_PROP_COLOR_AUTO_EXPOSURE_BOOL);
        std::cerr << "ROI exposure readback: reason=" << reason
                  << " requested=" << exposure << " actual=" << actual_exposure
                  << " gain=" << actual_gain << " auto=" << actual_auto << "\n";
        if(actual_exposure != exposure || actual_gain != gain || actual_auto)
            throw std::runtime_error("camera exposure settings did not take effect");
    }

    void update(const std::shared_ptr<ob::ColorFrame> &frame) {
        if(!device || settle_frames-- > 0) return;
        // The lift panel occupies this central-right band at the ready pose.  It
        // remains in the band while aiming; the two bright windows stay outside.
        const int width = static_cast<int>(frame->width());
        const int height = static_cast<int>(frame->height());
        const int x0 = static_cast<int>(0.46 * width);
        const int x1 = static_cast<int>(0.80 * width);
        const int y0 = static_cast<int>(0.03 * height);
        const int y1 = static_cast<int>(0.88 * height);
        const auto *rgb = static_cast<const std::uint8_t *>(frame->data());
        std::vector<std::uint8_t> luma;
        std::size_t sampled = 0;
        std::size_t saturated = 0;
        luma.reserve(((x1 - x0) / 4 + 1) * ((y1 - y0) / 4 + 1));
        for(int y = y0; y < y1; y += 4) {
            for(int x = x0; x < x1; x += 4) {
                const auto *p = rgb + (y * width + x) * 3;
                const int value = (77 * p[0] + 150 * p[1] + 29 * p[2]) >> 8;
                ++sampled;
                if(value >= 250) ++saturated;
                // Ignore dead pixels and saturated reflections; keep the dark
                // panel itself because it is exactly what must drive metering.
                if(value > 2 && value < 250)
                    luma.push_back(static_cast<std::uint8_t>(value));
            }
        }
        const int recovered = saturation_recovery_exposure(
            exposure, exposure_min, saturated, sampled);
        if(sampled > 0 && saturated * 5 >= sampled * 4) {
            // Do not return just because clipping removed every meter pixel.
            exposure = recovered;
            gain = gain_min;
            device->setIntProperty(OB_PROP_COLOR_EXPOSURE_INT, exposure);
            device->setIntProperty(OB_PROP_COLOR_GAIN_INT, gain);
            std::cerr << "ROI saturation recovery: saturated=" << saturated
                      << "/" << sampled << " exposure=" << exposure << "\n";
            verify_properties("saturation_recovery");
            settle_frames = 3;
            return;
        }
        if(luma.size() < 100) return;
        const std::size_t percentile = luma.size() * 65 / 100;
        std::nth_element(luma.begin(), luma.begin() + percentile, luma.end());
        const int measured = luma[percentile];
        constexpr int target = 92;
        constexpr int deadband = 8;
        if(std::abs(measured - target) <= deadband) {
            settle_frames = 8;
            return;
        }

        int next_exposure = exposure;
        int next_gain = gain;
        if(measured < target - deadband) {
            if(exposure < exposure_max) {
                const double ratio = std::clamp(target / std::max(8.0, double(measured)),
                                                1.10, 1.55);
                next_exposure = std::min(exposure_max,
                                         int(std::lround(exposure * ratio)));
            }
            else {
                next_gain = std::min(gain_max, gain + (measured < 55 ? 5 : 2));
            }
        }
        else {
            if(gain > gain_min)
                next_gain = std::max(gain_min, gain - (measured > 140 ? 5 : 2));
            else
                next_exposure = std::max(exposure_min,
                                         int(std::lround(exposure * 0.82)));
        }
        if(next_exposure != exposure) {
            exposure = next_exposure;
            device->setIntProperty(OB_PROP_COLOR_EXPOSURE_INT, exposure);
        }
        if(next_gain != gain) {
            gain = next_gain;
            device->setIntProperty(OB_PROP_COLOR_GAIN_INT, gain);
        }
        verify_properties("meter_update");
        std::cerr << "ROI exposure: p65=" << measured << " exposure=" << exposure
                  << " gain=" << gain << "\n";
        settle_frames = 3;
    }
};

}  // namespace

int main(int argc, char **argv) try {
    if(argc != 1) {
        std::cerr << "usage: orbbec_v1_stream\n";
        return 2;
    }
    std::signal(SIGTERM, stop_running);
    std::signal(SIGINT, stop_running);
    std::ios::sync_with_stdio(false);

    // The vendor SDK defaults to a verbose file logger.  A resident camera
    // process would otherwise grow OrbbecSDK.log.txt continuously.
    ob::Context::setLoggerSeverity(OB_LOG_SEVERITY_ERROR);

    ob::Pipeline pipeline;
    auto device = pipeline.getDevice();
    // Gemini 330 depth discontinuities can contain laterally extended flying
    // pixels.  They are stable across adjacent frames, so temporal debounce in
    // the robot safety layer cannot distinguish them from a real obstacle.
    // Apply the camera SDK's own edge-noise filter before publishing depth.
    // Keep the device-recommended parameters: they are calibrated for the
    // connected camera and avoid hard-coding thresholds in disparity units.
    std::shared_ptr<ob::EdgeNoiseRemovalFilter> edge_noise_filter;
    try {
        auto filters = device->getSensor(OB_SENSOR_DEPTH)->getRecommendedFilters();
        for(std::uint32_t i = 0; i < filters->count(); ++i) {
            auto filter = filters->getFilter(i);
            if(filter->is<ob::EdgeNoiseRemovalFilter>()) {
                edge_noise_filter = filter->as<ob::EdgeNoiseRemovalFilter>();
                edge_noise_filter->enable(true);
                const auto params = edge_noise_filter->getFilterParams();
                std::cerr << "Edge depth filter enabled: type=" << params.type
                          << " margins=" << params.marginLeftTh << ","
                          << params.marginRightTh << "," << params.marginTopTh
                          << "," << params.marginBottomTh << "\n";
                break;
            }
        }
    }
    catch(const ob::Error &e) {
        std::cerr << "Edge depth filter unavailable: " << e.getMessage() << "\n";
        edge_noise_filter.reset();
    }
    if(!edge_noise_filter) {
        // Some Gemini 330 firmware revisions expose the processing block in
        // the SDK but omit disabled blocks from getRecommendedFilters().
        // Constructing it directly still uses the SDK defaults and lets the
        // first processed frame prove whether this device/format supports it.
        try {
            edge_noise_filter = std::make_shared<ob::EdgeNoiseRemovalFilter>();
            edge_noise_filter->enable(true);
            const auto params = edge_noise_filter->getFilterParams();
            std::cerr << "Edge depth filter enabled directly: type=" << params.type
                      << " margins=" << params.marginLeftTh << ","
                      << params.marginRightTh << "," << params.marginTopTh
                      << "," << params.marginBottomTh << "\n";
        }
        catch(const ob::Error &e) {
            std::cerr << "Edge depth filter construction failed: "
                      << e.getMessage() << "\n";
            edge_noise_filter.reset();
        }
    }
    // The elevator panel is between two bright windows. Backlight compensation
    // helps, while the ROI controller below prevents the windows from dominating
    // exposure metering.
    if(device->isPropertySupported(OB_PROP_COLOR_AUTO_EXPOSURE_BOOL,
                                   OB_PERMISSION_WRITE))
        device->setBoolProperty(OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, true);
    if(device->isPropertySupported(OB_PROP_COLOR_BACKLIGHT_COMPENSATION_INT,
                                   OB_PERMISSION_WRITE))
        device->setIntProperty(OB_PROP_COLOR_BACKLIGHT_COMPENSATION_INT, 2);
    auto config = std::make_shared<ob::Config>();
    auto color_profiles = pipeline.getStreamProfileList(OB_SENSOR_COLOR);
    auto depth_profiles = pipeline.getStreamProfileList(OB_SENSOR_DEPTH);
    auto color = color_profiles->getVideoStreamProfile(
        640, 360, OB_FORMAT_ANY, 30);
    auto depth = std::const_pointer_cast<ob::StreamProfile>(
                     depth_profiles->getProfile(OB_PROFILE_DEFAULT))
                     ->as<ob::VideoStreamProfile>();
    config->enableStream(color);
    config->enableStream(depth);
    config->setAlignMode(ALIGN_D2C_SW_MODE);
    try {
        pipeline.enableFrameSync();
    }
    catch(...) {
    }
    pipeline.start(config);

    ob::FormatConvertFilter converter;
    RoiExposureController roi_exposure(device);
    std::uint64_t sequence = 0;
    int warmup = 0;
    while(running) {
        auto frames = pipeline.waitForFrames(1500);
        if(!frames || warmup++ < 8) continue;
        auto color_frame = frames->colorFrame();
        auto depth_frame = frames->depthFrame();
        if(!color_frame || !depth_frame) continue;
        if(edge_noise_filter) {
            auto filtered = edge_noise_filter->process(depth_frame);
            if(!filtered) continue;
            depth_frame = filtered->as<ob::DepthFrame>();
        }
        color_frame = to_rgb(color_frame, converter);
        roi_exposure.update(color_frame);
        const std::uint32_t width = color_frame->width();
        const std::uint32_t height = color_frame->height();
        if(width != 640 || height != 360)
            throw std::runtime_error("color frame is not 640x360");
        if(depth_frame->width() != width || depth_frame->height() != height)
            throw std::runtime_error("aligned depth dimensions do not match color");

        const float depth_scale_mm = depth_frame->getValueScale();
        const std::uint64_t color_timestamp = color_frame->timeStamp();
        const std::uint64_t depth_timestamp = depth_frame->timeStamp();
        const std::uint32_t color_bytes = width * height * 3;
        const std::uint32_t depth_bytes = width * height * sizeof(std::uint16_t);
        const char magic[4] = {'O', 'R', 'B', 'D'};

        std::cout.write(magic, sizeof(magic));
        write_value(width);
        write_value(height);
        write_value(depth_scale_mm);
        write_value(color_timestamp);
        write_value(depth_timestamp);
        write_value(++sequence);
        write_value(color_bytes);
        write_value(depth_bytes);
        std::cout.write(static_cast<const char *>(color_frame->data()), color_bytes);
        std::cout.write(static_cast<const char *>(depth_frame->data()), depth_bytes);
        std::cout.flush();
        if(!std::cout) break;
    }
    pipeline.stop();
    return 0;
}
catch(const ob::Error &e) {
    std::cerr << "Orbbec error in " << e.getName() << ": " << e.getMessage()
              << " (" << e.getArgs() << ")\n";
    return 1;
}
catch(const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
}
