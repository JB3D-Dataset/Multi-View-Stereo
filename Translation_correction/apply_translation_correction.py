"""Apply the JB3D translation correction (<scene>/transform.txt of this folder) to a scene's reference model.

The published models of the 14 video scenes with LiDAR are offset from the LiDAR point clouds by a constant
translation per scene. transform.txt is the 4x4 matrix [I | delta] (delta = E, N, U in metres, in the frame of
enu_observer.txt / the LiDAR point clouds). This script moves
  points3D.txt, points3D.ply:          X -> X + delta
  images.txt, images_30fps.txt:        camera centre C -> C + delta, i.e. TX TY TZ -> t - R delta (quaternion unchanged)
and copies every other line unchanged. Requires numpy only.

usage: python apply_translation_correction.py SCENE_DIR TRANSFORM_TXT OUT_DIR
  e.g. python apply_translation_correction.py Entrance_III JB3D_translation_correction/Entrance_III/transform.txt Entrance_III_corrected
"""
import os
import sys

import numpy as np


def quat_to_rotmat(qw, qx, qy, qz):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)]])


def fix_images_txt(src, dst, delta):
    with open(src) as f, open(dst, "w") as g:
        for line in f:
            t = line.split()
            # image line: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME (a POINTS2D line has 3k entries)
            if not line.startswith("#") and len(t) == 10:
                R = quat_to_rotmat(*map(float, t[1:5]))
                tvec = np.array(t[5:8], float) - R @ delta
                line = " ".join(t[:5] + [repr(float(v)) for v in tvec] + t[8:]) + "\n"
            g.write(line)


def fix_points3d_txt(src, dst, delta):
    with open(src) as f, open(dst, "w") as g:
        for line in f:
            if not line.startswith("#") and line.strip():
                t = line.rstrip("\n").split(" ")   # POINT3D_ID X Y Z R G B ERROR TRACK[]
                t[1:4] = [repr(float(v) + d) for v, d in zip(t[1:4], delta)]
                line = " ".join(t) + "\n"
            g.write(line)


def fix_points3d_ply(src, dst, delta):
    with open(src, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            header += f.readline()
        body = f.read()
    lines = header.decode().splitlines()
    assert "format binary_little_endian 1.0" in lines
    types = {"float": "<f4", "double": "<f8", "uchar": "u1"}
    dtype = np.dtype([(l.split()[2], types[l.split()[1]]) for l in lines if l.startswith("property")])
    v = np.frombuffer(body, dtype).copy()
    for k, d in zip("xyz", delta):
        v[k] = (v[k].astype(np.float64) + d).astype(v[k].dtype)
    with open(dst, "wb") as g:
        g.write(header)
        g.write(v.tobytes())


if __name__ == "__main__":
    scene, transform, out = sys.argv[1:4]
    T = np.loadtxt(transform)
    assert T.shape == (4, 4) and np.allclose(T[:3, :3], np.eye(3)) and np.allclose(T[3], [0, 0, 0, 1]), \
        "expects a pure translation [I | delta]"
    delta = T[:3, 3]
    os.makedirs(out, exist_ok=True)
    for name, fn in (("images.txt", fix_images_txt), ("images_30fps.txt", fix_images_txt),
                     ("points3D.txt", fix_points3d_txt), ("points3D.ply", fix_points3d_ply)):
        if os.path.exists(os.path.join(scene, name)):
            fn(os.path.join(scene, name), os.path.join(out, name), delta)
            print(f"{name}: shifted by {delta.tolist()} m -> {os.path.join(out, name)}")
