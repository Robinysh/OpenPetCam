"""A world model of the room, built from every sensor the camera (and the computer next to it) has.

    camera.py    the camera as an instrument: calibrated rays for any pixel at any pan/tilt
    geometry.py  directions, panoramas, floor intersection
    layers.py    what a sensor produces: Panorama, Frames, Series, Echoes, Points
    sensors/     one module per sensor; each declares its band and how it was measured
    room.py      the fused geometry: planes read from the panorama, ranges from sound
    build.py     run every sensor, write eval/world/model/ (manifest + assets) and the viewer

Everything it writes stays in eval/ (git-ignored): it is pictures and times from inside a home.
"""
