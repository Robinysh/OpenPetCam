"""Radio: 2.4 and 5 GHz WiFi (wavelengths 12.5 cm and 6 cm).

WifiLink: the camera's own received signal strength from its access point, polled twice a second through one
wpa_cli session (`signal_poll`). One antenna and no channel state information, so it cannot image the room; what it
can show is the radio channel changing: people and the dog crossing the path between camera and router perturb it.
WifiEnvironment: one passive scan by this computer (CoreWLAN via system_profiler): every network in range with its
channel, band and signal. Network names are dropped. Positions of the access points are not observable from one scan."""
import json
import os
import re
import subprocess
import time

import numpy as np

from ..layers import Band, Layer
from .base import Sensor


class WifiLink(Sensor):
    name = "radio.link"
    band = Band("radio", 0.1220, 0.1250, "camera WiFi, 2.4 GHz channel (from wpa_cli FREQUENCY)")

    def layers(self):
        p = self.ctx.path("rssi.log")
        if not os.path.exists(p):
            return []
        t, v = [], []
        for line in open(p):
            m = re.match(r"([\d.]+) (?:> )?RSSI=(-?\d+)", line.strip())
            if m:
                t.append(float(m.group(1))); v.append(int(m.group(2)))
        if not t:
            return []
        t, v = np.array(t), np.array(v, float)
        # per-minute variability of the link, against the camera's own motion measure in that minute
        mins = {}
        for ti, vi in zip(t, v):
            mins.setdefault(int(ti // 60 * 60), []).append(vi)
        minute = {m["ts"]: m for m in self.ctx.json("minute.json")}
        pairs = [(float(np.std(vs)), minute[k].get("motion") or 0, minute[k].get("present") or 0)
                 for k, vs in mins.items() if k in minute and len(vs) > 60]
        r = float(np.corrcoef([p[0] for p in pairs], [p[1] for p in pairs])[0, 1]) if len(pairs) > 5 else None
        step = max(1, len(t) // 20000)
        return [Layer("radio.rssi", self.name, self.band, "series", "WiFi: camera signal strength",
                      "The camera's received signal from its router twice a second (dBm). Bodies moving through the "
                      "radio path change it." + (f" Per-minute spread vs. picture motion: r = {r:.2f} over {len(pairs)} minutes."
                                                 if r is not None else ""),
                      float(t[0]), float(t[-1]),
                      {"unit": "dBm", "t": np.round(t[::step], 1).tolist(), "v": v[::step].tolist(),
                       "minute_spread": {str(k): round(float(np.std(vs)), 2) for k, vs in mins.items()},
                       "motion_corr": r, "minutes": len(pairs)})]


class WifiEnvironment(Sensor):
    name = "radio.scan"
    band = Band("radio", 0.05, 0.125, "this computer's WiFi radio, passive scan, 2.4 and 5 GHz")

    def collect(self):
        out = subprocess.run(["system_profiler", "-json", "SPAirPortDataType"], capture_output=True, text=True,
                             timeout=60).stdout
        js = json.loads(out or "{}")
        nets = []
        for itf in js.get("SPAirPortDataType", [{}])[0].get("spairport_airport_interfaces", []):
            cur = itf.get("spairport_current_network_information")
            for n, home in ([(cur, True)] if cur else []) + [(x, False) for x in itf.get("spairport_airport_other_local_wireless_networks", [])]:
                ch = str(n.get("spairport_network_channel", ""))
                sn = str(n.get("spairport_signal_noise", ""))
                m = re.findall(r"-?\d+", sn)
                nets.append({"home": home, "channel": ch, "band": "5 GHz" if "5GHz" in ch else ("6 GHz" if "6GHz" in ch else "2.4 GHz"),
                             "signal_dbm": int(m[0]) if m else None, "noise_dbm": int(m[1]) if len(m) > 1 else None,
                             "security": n.get("spairport_security_mode", "")})
        os.makedirs(self.ctx.path("raw", "radio"), exist_ok=True)
        json.dump({"t": time.time(), "networks": nets}, open(self.ctx.path("raw", "radio", "scan.json"), "w"), indent=1)

    def layers(self):
        p = self.ctx.path("raw", "radio", "scan.json")
        if not os.path.exists(p):
            self.collect()
        s = json.load(open(p))
        return [Layer("radio.scan", self.name, self.band, "table", "WiFi: networks in range",
                      f"One passive scan from this computer: {len(s['networks'])} networks (names dropped). Signal is "
                      "at the computer, not the camera; direction and position are not observable from one antenna.",
                      s["t"], s["t"], s)]
