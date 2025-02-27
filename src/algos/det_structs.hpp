#ifndef __DET_STRUCTS_HPP__
#define __DET_STRUCTS_HPP__
#include <iostream>

struct Bbox2D {
    std::string cls_name;
    int32_t cls_id;
    float conf;

    int32_t x;
    int32_t y;
    int32_t w;
    int32_t h;

    Bbox2D():
        cls_id(0), conf(0), x(0), y(0), w(0), h(0)
    {}
    Bbox2D(int32_t cls_id_, float conf_, int32_t x_, int32_t y_, int32_t w_, int32_t h_):
        cls_id(cls_id_), conf(conf_), x(x_), y(y_), w(w_), h(h_)
    {}
    Bbox2D(std::string cls_name_, int32_t cls_id_, float conf_, int32_t x_, int32_t y_, int32_t w_, int32_t h_):
        cls_name(std::move(cls_name_)), cls_id(cls_id_), conf(conf_), x(x_), y(y_), w(w_), h(h_)
    {}
};

#endif
