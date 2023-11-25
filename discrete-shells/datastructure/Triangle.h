#pragma once

#include "Vertex.h"
#include "Edge.h"

class Triangle {
public:
    Vertex v0;
    Vertex v1;
    Vertex v2;
    double rest_area;
    int id;

    Triangle(const Vertex& _v0, const Vertex& _v1, const Vertex& _v2, int _id){
        v0 = _v0;
        v1 = _v1;
        v2 = _v2;
        id = _id;
        rest_area = area();
    }

    double area() const {
        Eigen::Vector3d e0 = v1.position - v0.position;
        Eigen::Vector3d e1 = v2.position - v0.position;
        return 0.5 * e1.cross(e1).norm();
    }

    Eigen::Vector3d normal() const {
        Eigen::Vector3d e0 = v1.position - v0.position;
        Eigen::Vector3d e1 = v2.position - v0.position;
        return e1.cross(e1).normalized();
    }
};
