"""Open the fused room in Blender: floor, walls, the countertop overhang, the lens as an empty, and the calibrated
panorama projected from the lens onto every surface. Each vertex gets the panorama coordinate of its direction from the
lens (the same formula as the web viewer), on finely divided meshes, so the projection needs no Blender convention.

    blender -b --python sim/world/blender_export.py -- eval/world/model
        -> eval/world/model/room.blend, room.glb, blender_check.png (perspective), blender_plan.png (top)

Blender is z-up; the model is y-up: model (x, y, z) -> Blender (x, z, y): the model's frame is left-handed, so this mirror is what keeps left and right."""
import json
import math
import os
import sys

import bpy  # noqa: E402  (runs inside Blender)

model = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else "eval/world/model"
M = json.load(open(os.path.join(model, "manifest.json")))
R = M["room"]
H, CEIL = R["floor_height_m"], R["ceiling_m"]


def B(x, y, z):
    return (x, z, y)          # model frame is left-handed (x right, y up, z forward); Blender's is right-handed, z up


bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene

img = bpy.data.images.load(os.path.abspath(os.path.join(model, "pano_visible_sweep.jpg")))
mat = bpy.data.materials.new("lens projection")
mat.use_nodes = True
nt = mat.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)
out = nt.nodes.new("ShaderNodeOutputMaterial")
emit = nt.nodes.new("ShaderNodeEmission")
texn = nt.nodes.new("ShaderNodeTexImage")
texn.image = img
texn.extension = "REPEAT"
lum = nt.nodes.new("ShaderNodeRGBToBW"); seen = nt.nodes.new("ShaderNodeMath"); seen.operation = "GREATER_THAN"
seen.inputs[1].default_value = 0.01
mix = nt.nodes.new("ShaderNodeMix"); mix.data_type = "RGBA"; mix.inputs["A"].default_value = (0.13, 0.15, 0.18, 1)
nt.links.new(texn.outputs["Color"], lum.inputs[0]); nt.links.new(lum.outputs[0], seen.inputs[0])
nt.links.new(seen.outputs[0], mix.inputs["Factor"]); nt.links.new(texn.outputs["Color"], mix.inputs["B"])
nt.links.new(mix.outputs["Result"], emit.inputs["Color"])
nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])


def pano_uv(v):
    """The panorama pixel for a point seen from the lens, exactly as the web viewer computes it (model coordinates)."""
    x, y, z = v
    r = math.sqrt(x * x + y * y + z * z) or 1
    return (math.atan2(x, z) / (2 * math.pi) + 0.5, 0.5 + math.asin(max(-1, min(1, y / r))) / math.pi)


def grid(p0, du, dv, nu, nv):
    """A quad p0 + s*du + t*dv split into nu x nv cells, so the projected texture bends with the surface."""
    vs = [tuple(p0[k] + du[k] * i / nu + dv[k] * j / nv for k in range(3)) for j in range(nv + 1) for i in range(nu + 1)]
    fs = [(j * (nu + 1) + i, j * (nu + 1) + i + 1, (j + 1) * (nu + 1) + i + 1, (j + 1) * (nu + 1) + i) for j in range(nv) for i in range(nu)]
    return vs, fs


def mesh(name, verts, faces):
    me = bpy.data.meshes.new(name)
    me.from_pydata([B(*v) for v in verts], [], faces)
    me.update()
    uv = me.uv_layers.new(name="pano")
    for poly in me.polygons:
        us = [pano_uv(verts[me.loops[li].vertex_index]) for li in poly.loop_indices]
        wrap = max(u for u, _ in us) - min(u for u, _ in us) > 0.5            # the face straddles azimuth 180
        for li, (u, w) in zip(poly.loop_indices, us):
            uv.data[li].uv = (u + 1 if wrap and u < 0.5 else u, w)
    ob = bpy.data.objects.new(name, me)
    ob.data.materials.append(mat)
    sc.collection.objects.link(ob)
    return ob


OL = [o for o in R["outline"] if o[1] is not None]
RINGS = 16
verts, faces = [(0, -H, 0)], []
for a, d, _l in OL:
    for k in range(1, RINGS + 1):
        verts.append((math.sin(math.radians(a)) * d * k / RINGS, -H, math.cos(math.radians(a)) * d * k / RINGS))
for i in range(len(OL)):
    j = (i + 1) % len(OL)
    if ((OL[j][0] - OL[i][0]) % 360) > 2:
        continue
    base_i, base_j = 1 + i * RINGS, 1 + j * RINGS
    faces.append((0, base_j, base_i))
    for k in range(RINGS - 1):
        faces.append((base_i + k, base_j + k, base_j + k + 1, base_i + k + 1))
mesh("floor", verts, faces)
for w in R["walls"]:
    (ax, az), (bx, bz) = w["a"], w["b"]
    vs, fs = grid((ax, -H, az), (bx - ax, 0, bz - az), (0, CEIL, 0), 24, 16)
    mesh(w["label"][:60], vs, fs)
if R.get("overhead_m"):
    y = R["overhead_m"]
    vs, fs = grid((-1.25, y, -0.8), (1.6, 0, 0), (0, 0, 1.0), 16, 10)
    mesh("countertop overhang", vs, fs)
lens = bpy.data.objects.new("lens", None)
sc.collection.objects.link(lens)

cam = bpy.data.cameras.new("view")
camo = bpy.data.objects.new("view", cam)
sc.collection.objects.link(camo)
camo.location = B(-5.5, 4.5, -4.5)
tgt = bpy.data.objects.new("target", None); sc.collection.objects.link(tgt); tgt.location = B(0, -0.5, 1.2)
c = camo.constraints.new("TRACK_TO"); c.target = tgt; c.track_axis = "TRACK_NEGATIVE_Z"; c.up_axis = "UP_Y"
sc.camera = camo
sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items] else "BLENDER_EEVEE"
sc.render.resolution_x, sc.render.resolution_y = 1280, 720
sc.world = bpy.data.worlds.new("w"); sc.world.color = (0.05, 0.06, 0.08)
cam.lens = 20
sc.render.filepath = os.path.abspath(os.path.join(model, "blender_check.png"))
bpy.ops.render.render(write_still=True)
camo.constraints.clear(); camo.location = (0, 1.2, 9); camo.rotation_euler = (0, 0, 0)
cam.type = "ORTHO"; cam.ortho_scale = 12
sc.render.filepath = os.path.abspath(os.path.join(model, "blender_plan.png"))
bpy.ops.render.render(write_still=True)
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(os.path.join(model, "room.blend")))
bpy.ops.export_scene.gltf(filepath=os.path.abspath(os.path.join(model, "room.glb")), export_format="GLB")
print("wrote room.blend, room.glb, blender_check.png")
