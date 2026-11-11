#!/usr/bin/env python3
"""Export YOLO26 weights (Ultralytics, AGPL-3.0) to the ONNX the watcher reads when DETECTOR=yolo26.

    make_yolo26.py yolo26s.pt models/yolo26s.onnx

Free input height and width (any multiple of 32), so DETECT_WIDTH still sets the size. Needs the `ultralytics`,
`onnx` and `onnxruntime` packages (tools/fetch_models.sh builds a cached virtualenv for it; it is far too heavy for
the service's own venv). The output is checked: a head of shape (1, 84, N) is what server/detector.py decodes."""
import sys

import numpy as np


def main(pt, out):
    from ultralytics import YOLO

    path = YOLO(pt).export(format="onnx", dynamic=True, simplify=False, half=False, opset=17)
    import shutil

    shutil.move(path, out)
    import onnxruntime as ort

    s = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    y = s.run(None, {s.get_inputs()[0].name: np.zeros((1, 3, 384, 640), np.float32)})[0]
    if y.ndim != 3 or y.shape[1] != 84:
        sys.exit(f"unexpected head {y.shape}: server/detector.py decodes (1, 84, N)")
    print(f"wrote {out}: head {y.shape} at 640x384")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
