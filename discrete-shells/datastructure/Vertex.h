#pragma once

#include "eigen/Eigen/Dense"

class Vertex {
public:
    Eigen::Vector3d position;
    int id;

    Vertex(double x, double y, double z, int _id) { // Constructor with parameters
        position = Eigen::Vector3d(x,y,z);
        id = _id;
    }
};