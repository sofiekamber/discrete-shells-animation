#include "ShellObject.h"

// Default constructor
ShellObject::ShellObject() {
    // Initialization of vectors can be done here if needed
}

void ShellObject::addVertex(const Vertex& vertex) {
    vertices.push_back(vertex);
}

void ShellObject::addEdge(const Edge& edge) {
    edges.push_back(edge);
}

void ShellObject::addTriangle(const Triangle& triangle) {
    triangles.push_back(triangle);
}
// Add more functions as needed

