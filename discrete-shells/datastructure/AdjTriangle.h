#pragma once

#include "Vertex.h"
#include "Edge.h"
#include "Triangle.h"

class AdjTriangle {
public:
    Vertex v0;
    Vertex v1;
    Vertex v2;
    Vertex v3;
    Edge e;
    Triangle t0;
    Triangle t1;
    double e_rest_length;
    double rest_height;
    double rest_angle;

    AdjTriangle(const Triangle& _t0, const Triangle& _t1, const Edge& _e) {
        t0 = _t0;
        t1 = _t1;
        e = _e;
        rest_angle = dihedralAngle();
        rest_height = height();
    }

    double height() const {
        //1/6 * (height_t0 + height_t1) sharing e
        double height_t0 = 2 * t0.area() / e.length();
        double height_t1 = 2 * t1.area() / e.length();
        return (1/6) * (height_t0 + height_t1);
    }

    double dihedralAngle() const{

        Eigen::Vector3d normal0 = t0.normal();
        Eigen::Vector3d normal1 = t1.normal();

        double dotProduct = normal0.dot(normal1);

        // Ensure the dot product is within valid range [-1, 1] to avoid NaN in arccos
        dotProduct = std::max(-1.0, std::min(1.0, dotProduct));

        double angleRad = std::acos(dotProduct);

        double angleDeg = angleRad * 180.0 / M_PI;

        return angleRad;
    }
};
