#pragma once

#include <vector>
#include "Vertex.h"
#include "Edge.h"
#include "Triangle.h"

class ShellObject {
public:
    std::vector<Vertex> vertices;
    std::vector<Edge> edges;
    std::vector<Triangle> triangles;

    // Constructors
    ShellObject();  // Default constructor

    // Add functions to manipulate vertices, edges, and triangles as needed
    void addVertex(const Vertex& vertex);
    void addEdge(const Edge& edge);
    void addTriangle(const Triangle& triangle);
    // Add more functions as needed

    // Example functions to calculate overall properties
    double totalArea() const;
    double totalVolume() const;
};
