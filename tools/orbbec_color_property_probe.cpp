#include "libobsensor/ObSensor.hpp"

#include <iostream>
#include <memory>
#include <string>

void print_bool(const std::shared_ptr<ob::Device> &device, OBPropertyID id,
                const char *name) {
    bool readable = device->isPropertySupported(id, OB_PERMISSION_READ);
    bool writable = device->isPropertySupported(id, OB_PERMISSION_WRITE);
    std::cout << name << " supported=" << readable << " writable=" << writable;
    if(readable) std::cout << " value=" << device->getBoolProperty(id);
    std::cout << "\n";
}

void print_int(const std::shared_ptr<ob::Device> &device, OBPropertyID id,
               const char *name) {
    bool readable = device->isPropertySupported(id, OB_PERMISSION_READ);
    bool writable = device->isPropertySupported(id, OB_PERMISSION_WRITE);
    std::cout << name << " supported=" << readable << " writable=" << writable;
    if(readable) {
        auto range = device->getIntPropertyRange(id);
        std::cout << " value=" << device->getIntProperty(id)
                  << " min=" << range.min << " max=" << range.max
                  << " step=" << range.step << " default=" << range.def;
    }
    std::cout << "\n";
}

int main() try {
    ob::Context::setLoggerSeverity(OB_LOG_SEVERITY_ERROR);
    ob::Context context;
    auto devices = context.queryDeviceList();
    if(devices->deviceCount() != 1)
        throw std::runtime_error("expected exactly one Orbbec device");
    auto device = devices->getDevice(0);
    auto info = device->getDeviceInfo();
    std::cout << "device_name=" << info->name()
              << " vid=" << info->vid()
              << " pid=" << info->pid()
              << " firmware=" << info->firmwareVersion() << "\n";
    print_bool(device, OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, "color_auto_exposure");
    print_int(device, OB_PROP_COLOR_EXPOSURE_INT, "color_exposure");
    print_int(device, OB_PROP_COLOR_GAIN_INT, "color_gain");
    print_int(device, OB_PROP_COLOR_MAXIMAL_GAIN_INT, "color_maximal_gain");
    print_int(device, OB_PROP_COLOR_MAXIMAL_SHUTTER_INT, "color_maximal_shutter");
    print_int(device, OB_PROP_COLOR_BACKLIGHT_COMPENSATION_INT, "color_backlight_compensation");
    print_int(device, OB_PROP_COLOR_BRIGHTNESS_INT, "color_brightness");
    print_int(device, OB_PROP_COLOR_CONTRAST_INT, "color_contrast");
    print_int(device, OB_PROP_COLOR_GAMMA_INT, "color_gamma");
    print_int(device, OB_PROP_COLOR_SHARPNESS_INT, "color_sharpness");
    print_int(device, OB_PROP_COLOR_FOCUS_INT, "color_focus");
    return 0;
}
catch(const ob::Error &error) {
    std::cerr << error.getName() << ": " << error.getMessage() << "\n";
    return 1;
}
catch(const std::exception &error) {
    std::cerr << error.what() << "\n";
    return 1;
}
