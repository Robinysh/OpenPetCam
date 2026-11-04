"""A camera that is NOT the reference one, with hidden parameters, for testing that self-calibration recovers a new
camera's behaviour. It answers the same two calls sim/sysid_camera.py makes of a real camera: a shell script (the
motor trace, in the camera's own output format) and a snapshot (a crop of a textured panorama at the OPTICAL pose, so
gear play shows in the picture but not in the step counter).
"""
import re

import numpy as np


class VirtualCamera:
    def __init__(self, speed=140.0, fast_ratio=3.2, start_s=0.25, frame_steps=(900.0, 520.0), backlash=60.0,
                 frame_px=(640, 360), seed=0, noise=2.0):
        self.v, self.fast, self.start, self.b = speed, fast_ratio, start_s, backlash
        self.fs, self.px = frame_steps, frame_px
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        self.C = [1850.0, 500.0]                     # step counter (motors -j)
        self.O = [1850.0, 500.0]                     # optical pointing (follows the counter with gear play)
        self.t = 100.0
        h, w = frame_px[1] * 6, frame_px[0] * 8      # a smooth random panorama, bigger than the travel
        f = np.fft.rfft2(self.rng.normal(size=(h, w)))
        ky, kx = np.meshgrid(np.fft.fftfreq(h), np.fft.rfftfreq(w), indexing="ij")
        f /= (1 + (np.hypot(kx, ky) * 60) ** 2)
        img = np.fft.irfft2(f, s=(h, w))
        self.pano = (img - img.min()) / (img.max() - img.min()) * 255
        self.q = []                                  # queued moves: (axis_dx, axis_dy, speed)

    # physics
    def _run_to(self, t_end, out=None, poll=0.064):
        while self.t < t_end:
            dt = min(poll, t_end - self.t)
            if self.q:
                dx, dy, sp, t_go = self.q[0]
                if self.t >= t_go:
                    step = sp * dt
                    for k, d in ((0, dx), (1, dy)):
                        if d:
                            m = max(-step, min(step, d))
                            self.C[k] += m
                            d -= m
                            half = self.b / 2
                            if self.C[k] - self.O[k] > half:
                                self.O[k] = self.C[k] - half
                            elif self.O[k] - self.C[k] > half:
                                self.O[k] = self.C[k] + half
                        if k == 0:
                            dx = d
                        else:
                            dy = d
                    self.q[0] = (dx, dy, sp, t_go)
                    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                        self.q.pop(0)
                        if self.q:                   # the next queued move starts after the start delay
                            nx, ny, ns, _ = self.q[0]
                            self.q[0] = (nx, ny, ns, self.t + self.start)
            self.t += dt
            if out is not None:
                out.append(f"{self.t:.2f} status:{1 if self.q and self.t >= self.q[0][3] else 0} "
                           f"xpos:{int(round(self.C[0]))} ypos:{int(round(self.C[1]))} speed:250 invert:2")

    def _cmd(self, dx, dy, speed):
        sp = self.v if speed <= 300 else self.v * self.fast
        self.q.append((float(dx), float(dy), sp, self.t + self.start if not self.q else 1e18))

    # the two calls sysid_camera makes
    def __call__(self, script):
        out = []
        if "== A:" in script:                         # the motor characterisation script
            for line in script.splitlines():
                m = re.search(r'echo "(== [^"]+)"', line)
                if m:
                    out.append(m.group(1))
                for cmd in re.split(r";\s*", line):
                    mm = re.search(r"motors -d g -s (\d+) -x (-?\d+) -y (-?\d+)", cmd)
                    if 'echo "CMD' in cmd:
                        out.append(f"{'CMD2' if 'CMD2' in cmd else 'CMD'} {self.t:.2f}")
                    elif mm:
                        self._cmd(int(mm.group(2)), int(mm.group(3)), int(mm.group(1)))
                    else:
                        pm = re.search(r"poll ([\d.]+)", cmd)
                        if pm:
                            self._run_to(self.t + float(pm.group(1)), out)
                        elif cmd.strip() == "idle":
                            n = 0
                            while self.q and n < 3000:
                                self._run_to(self.t + 0.1)
                                n += 1
            return "\n".join(out) + "\n"
        mm = re.search(r"motors -d g -s (\d+) -x (-?\d+) -y (-?\d+)", script)
        if mm:
            self._cmd(int(mm.group(2)), int(mm.group(3)), int(mm.group(1)))
        if "while [" in script:                       # an idle-wait loop: run until the queued moves finished
            n = 0
            while self.q and n < 3000:
                self._run_to(self.t + 0.1)
                n += 1
        for sl in re.findall(r"sleep ([\d.]+)", script.split("done")[-1] if "done" in script else script):
            self._run_to(self.t + float(sl))
        if "motors -j" in script:
            return '{"status":"0","xpos":"%d","ypos":"%d","invert":"0"}' % (self.C[0], self.C[1])
        return ""

    def snapshot(self):
        w, h = self.px
        cx = self.O[0] * w / self.fs[0]              # optical pointing -> panorama pixels
        cy = self.O[1] * h / self.fs[1]
        x0 = int(round(cx - w / 2 + self.pano.shape[1] / 2 - 1850 * w / self.fs[0]))
        y0 = int(round(cy - h / 2 + self.pano.shape[0] / 2 - 500 * h / self.fs[1]))
        crop = self.pano[y0:y0 + h, x0:x0 + w]
        return (crop + self.rng.normal(0, self.noise, crop.shape)).astype(np.float32)
