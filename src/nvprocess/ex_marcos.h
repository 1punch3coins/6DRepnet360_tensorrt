#pragma once
#ifdef API_EXPORT
    #if defined(_WIN32)
        #define API __declspec(dllexport)   // set exported symbols visibility
    #else
        #define API __attribute__((visibility("default")))  // set exported symbols visibility
    #endif
#else
    #if defined(_WIN32)
        #define API __declspec(dllimport)
    #else
        #define API
    #endif
#endif
