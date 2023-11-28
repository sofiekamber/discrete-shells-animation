import taichi as ti
ti.init(arch=ti.gpu, debug=False)

import flex_H

@ti.kernel
def test_felx_H():
    print("before call")
    result = flex_H.felx_H(0.0,0.0,0.0, 1.0,0.0,0.0, 1.0,1.0,0.0, 0.0,0.0,1.0, 1.0, 1.0, 1.0)
    print(result)

import flex_J
@ti.kernel
def test_felx_J() -> ti.types.vector(12, ti.f32):
   result =  flex_J.felx_J(0.0,0.0,0.0, 1.0,0.0,0.0, 1.0,1.0,0.0, 0.0,0.0,1.0, 1.0, 1.0, 1.0)
   return result

import edge_H
@ti.kernel
def test_edge_H() -> ti.types.vector(36, ti.f32):
    result =  edge_H.edge_H(0.0,0.0,0.0, 1.0,0.0,0.0, 1.0)
    return result

import edge_J
@ti.kernel
def test_edge_J() -> ti.types.vector(6, ti.f32):
    result = edge_J.edge_J(0.0,0.0,0.0, 1.0,0.0,0.0, 1.0)
    return result
