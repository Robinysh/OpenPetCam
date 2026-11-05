"""What a sensor produces. Every layer carries its band and its provenance, so the viewer can say what it is showing
and how it was measured. Times are unix seconds; positions are metres in the world frame of camera.py."""
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class Band:
    kind: str                 # "visible" | "infrared" | "acoustic" | "radio"
    lo_m: float               # wavelength range, metres
    hi_m: float
    note: str = ""


@dataclass
class Layer:
    id: str
    sensor: str
    band: Band
    type: str                 # "panorama" | "frames" | "series" | "echoes" | "points"
    title: str
    method: str               # how it was measured, in one or two sentences
    t0: Optional[float] = None
    t1: Optional[float] = None
    data: dict = field(default_factory=dict)   # type-specific: asset paths or inline arrays

    def to_json(self):
        return asdict(self)
