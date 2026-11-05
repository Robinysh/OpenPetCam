#!/usr/bin/env python3
"""Write a copy of the YOLOX detector whose input height/width are free (the graph is fully convolutional; only the
exported shape was fixed at 416x416). The watcher then runs it at 256x416, the 16:9 frame without the grey padding
a square input needs, and on small crops around where she is (sim/world/detcost.py measures both).
    python3 tools/make_dyn_detector.py models/yolox_nano.onnx models/yolox_nano_dyn.onnx"""
import sys

import onnx

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
d = m.graph.input[0].type.tensor_type.shape.dim
for k, name in ((2, "h"), (3, "w")):
    d[k].Clear()
    d[k].dim_param = name
for o in m.graph.output:
    od = o.type.tensor_type.shape.dim
    if len(od) >= 2:
        od[1].Clear()
        od[1].dim_param = "anchors"
onnx.checker.check_model(m)
onnx.save(m, dst)
print("wrote", dst)
