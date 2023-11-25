#pragma once

#include "Vertex.h"

class Edge {
public:
    Vertex v0;
    Vertex v1;
    double rest_length;
    int id;

    Edge(const Vertex& _v0, const Vertex& _v1, int _id){
        v0 = _v0;
        v1 = _v1;
        id = _id;
        rest_length = length();
    }

    double length() const {
        return (v0.position - v1.position).norm();
    }
};