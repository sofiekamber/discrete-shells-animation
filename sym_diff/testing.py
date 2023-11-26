import taichi as ti
ti.init(arch=ti.cpu, debug=True)

import flex_H

@ti.kernel
def test_felx_H():
    print("before call")
    result = flex_H.felx_H(0.0,0.0,0.0, 1.0,0.0,0.0, 1.0,1.0,0.0, 0.0,0.0,1.0, 1.0, 1.0, 1.0)
    print(result)
