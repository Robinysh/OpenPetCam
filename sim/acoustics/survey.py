"""Create immutable, balanced probe trials and fixed-source receiver surveys.

Planning is offline. A source means both position and orientation; pan is fixed
through each gather. Tilt changes the receiver, including its directivity.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

from .geometry import Rig


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def make_plan(packet_ids, calibration, *, mode="comparison", pans=(1850,),
              tilts=(520, 610, 699, 790, 880), reference_tilt=699,
              blocks=3, seed=731, volume=30, volumes=None, rig=None):
    """Repetition blocks in comparison; one contiguous block per fixed source.

    The returned plan is geometry/config only. Source timing is NOT inferred from
    motor coordinates or an acoustic pilot. Capture supplies actual poses later.
    """
    if mode not in ("comparison", "gather", "volume"):
        raise ValueError("mode must be comparison, gather or volume")
    if mode != "volume" and volumes is not None:
        raise ValueError("volumes apply only to volume-study mode")
    levels = tuple((20, 30, 40) if volumes is None else volumes) if mode == "volume" else ()
    if mode == "volume" and (len(levels) < 2 or any(type(v) is not int or not 1 <= v <= 40 for v in levels)
                             or len(set(levels)) != len(levels)):
        raise ValueError("volume study requires at least two unique integer levels in 1..40")
    values = (*pans, *tilts, reference_tilt, blocks, volume, seed)
    if any(type(v) is not int for v in values):
        raise ValueError("motor positions, blocks, volume and seed must be integers")
    if any(not isinstance(v, str) or not v for v in packet_ids):
        raise ValueError("packet IDs must be nonempty strings")
    if not packet_ids or len(set(packet_ids)) != len(packet_ids):
        raise ValueError("unique packet IDs required")
    if not pans or len(set(pans)) != len(pans) or not tilts or len(set(tilts)) != len(tilts):
        raise ValueError("unique nonempty pan and tilt coordinates required")
    if any(not 0 <= p <= 3700 for p in pans) or any(not 40 <= t <= 960 for t in (*tilts, reference_tilt)):
        raise ValueError("motor positions outside conservative reference-device range")
    if not 1 <= blocks <= 20 or not 1 <= volume <= 40:
        raise ValueError("blocks must be 1..20; volume 1..40")
    if mode == "volume" and volume not in levels:
        raise ValueError("reference volume must also appear among the randomized trial levels")
    if mode == "comparison" and len(pans) != 1:
        raise ValueError("probe comparison must use one fixed source/receiver pose")
    if mode == "gather" and len(packet_ids) != 1:
        raise ValueError("select one probe before a gather; compare probes separately")
    if mode == "volume" and (len(pans) != 1 or len(packet_ids) != 1):
        raise ValueError("volume study requires one fixed source/receiver pose and one exact probe")
    rig = rig or Rig()
    rnd = random.Random(seed)
    entries = []

    def append(pan, tilt, packet, block, role, level=None):
        pose = rig.pose_motor(calibration, pan, tilt)
        entries.append({"capture_id": f"capture_{len(entries):04d}", "packet_id": packet,
                        "source_id": f"pan_{pan}", "gather_id": f"pan_{pan}_block_{block}",
                        "receiver_id": f"pan_{pan}_tilt_{tilt}", "pan": pan, "tilt": tilt,
                        "block": block, "role": role, "volume": volume if level is None else level,
                        "source_m_assumed": pose.source_m.tolist(),
                        "receiver_m_assumed": pose.receiver_m.tolist(),
                        "source_axis_assumed": pose.source_axis.tolist(),
                        "receiver_axis_assumed": pose.receiver_axis.tolist()})

    if mode == "comparison":
        for block in range(blocks):
            order = list(packet_ids)
            rnd.shuffle(order)
            for packet in order:
                append(pans[0], reference_tilt, packet, block, "probe_comparison")
    elif mode == "volume":
        for block in range(blocks):
            append(pans[0], reference_tilt, packet_ids[0], block, "reference_before")
            order = list(levels)
            rnd.shuffle(order)
            for level in order:
                append(pans[0], reference_tilt, packet_ids[0], block, "volume_trial", level)
            append(pans[0], reference_tilt, packet_ids[0], block, "reference_after")
    else:
        for block in range(blocks):
            source_order = list(pans)
            rnd.shuffle(source_order)
            for pan in source_order:
                append(pan, reference_tilt, packet_ids[0], block, "reference_before")
                receiver_order = list(tilts)
                rnd.shuffle(receiver_order)
                for tilt in receiver_order:
                    append(pan, tilt, packet_ids[0], block, "receiver")
                append(pan, reference_tilt, packet_ids[0], block, "reference_after")
    plan = {"schema": "openpetcam_acquisition_v1", "mode": mode, "seed": seed,
            "entries": entries, "calibration": calibration,
            "timing_requirement": "hardware_reference_or_independently_calibrated_direct_path",
            "timing_verified": False,
            "assumptions": ["Rig acoustic centres are uncalibrated.",
                            "Scene must remain static within a sequential gather.",
                            "Receiver orientation changes with tilt; directional response needs calibration."],
            "processing": {"cross_pose_alignment": False, "per_shot_peak_normalization": False,
                           "stack_before_qc": False, "inversion_allowed": False}}
    if mode == "volume":
        plan.update(purpose="noisy_room_volume_robustness", reference_volume=volume,
                    volumes=list(levels), blocks=blocks)
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("comparison", "gather", "volume"), default="comparison")
    p.add_argument("--packet", action="append", help="packet ID; all suite packets for comparison")
    p.add_argument("--pans", nargs="+", type=int, default=[1850])
    p.add_argument("--tilts", nargs="+", type=int, default=[520, 610, 699, 790, 880])
    p.add_argument("--reference-tilt", type=int, default=699)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--volume", type=int, default=30)
    p.add_argument("--volumes", type=int, nargs="+", help="volume-study trial levels; defaults to 20 30 40")
    p.add_argument("--seed", type=int, default=731)
    a = p.parse_args()
    suite = json.loads(a.suite.read_text())
    packets = suite["packets"]
    ids = a.packet or [q["packet_id"] for q in packets]
    known = {q["packet_id"] for q in packets}
    if not set(ids) <= known:
        p.error("packet ID absent from suite")
    plan = make_plan(ids, json.loads(a.calibration.read_text()), mode=a.mode, pans=a.pans,
                     tilts=a.tilts, reference_tilt=a.reference_tilt, blocks=a.blocks,
                     volume=a.volume, volumes=a.volumes, seed=a.seed)
    plan.update(suite_sha256=digest(a.suite), calibration_sha256=digest(a.calibration))
    with a.output.open("x") as f:
        json.dump(plan, f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"{len(plan['entries'])} captures planned; no hardware actions")


if __name__ == "__main__":
    main()
