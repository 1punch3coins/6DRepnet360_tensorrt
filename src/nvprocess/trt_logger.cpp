#include <iostream>
#include <iomanip>
#include <ctime>
#include <assert.h>
#include <sstream>
#include "trt_logger.hpp"

namespace trtLogger{
    const char* get_severity_str(Severity& severity) {
        switch (severity) {
            case Severity::kINTERNAL_ERROR: return "[F] ";
            case Severity::kERROR:          return "[E] ";
            case Severity::kWARNING:        return "[W] ";
            case Severity::kINFO:           return "[I] ";
            case Severity::kVERBOSE:        return "[V] ";
            default: assert(0);             return "";
        }
    }
    void update_os_str(std::string& os_str) {
        std::time_t timestamp = std::time(nullptr);
        tm tm_local;
    
        // Use localtime_s for MSVC and localtime_r for POSIX compatibility
    #ifdef _MSC_VER
        localtime_s(&tm_local, &timestamp);
    #else
        localtime_r(&timestamp, &tm_local);
    #endif

        std::ostringstream buf;
        buf << "[" 
            << std::setw(2) << std::setfill('0') << 1 + tm_local.tm_mon << "/"
            << std::setw(2) << std::setfill('0') << tm_local.tm_mday << "/"
            << std::setw(4) << std::setfill('0') << 1900 + tm_local.tm_year << "-"
            << std::setw(2) << std::setfill('0') << tm_local.tm_hour << ":"
            << std::setw(2) << std::setfill('0') << tm_local.tm_min << ":"
            << std::setw(2) << std::setfill('0') << tm_local.tm_sec << "] ";
        os_str = buf.str();
    }

    void ColoredLogger::log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept {
        update_os_str(os_str_);
        if (severity <= severity_) {
            if (severity == Severity::kWARNING) {
                printf("%s\033[33m%s[TRT] %s\033[0m\n", os_str_.c_str(), get_severity_str(severity), msg);
            }
            else if (severity <= Severity::kERROR) {
                printf("%s\033[31m%s[TRT] %s\033[0m\n", os_str_.c_str(), get_severity_str(severity), msg);
            }
            else {
                printf("%s%s[TRT] %s\n", os_str_.c_str(), get_severity_str(severity), msg);
            }
        }
    }
    ColoredLogger clogger{Severity::kINFO};
}