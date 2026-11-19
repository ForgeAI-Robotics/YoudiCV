#pragma once
#include <algorithm>
#include <cstddef>

// A nearly saturated ROI has no usable non-highlight percentile. Recover
// before applying the normal trimmed-percentile meter.
inline int saturation_recovery_exposure(int exposure, int minimum,
                                        std::size_t saturated, std::size_t sampled) {
    if(sampled == 0 || saturated * 5 < sampled * 4) return exposure;
    return std::max(minimum, exposure / 2);
}
