# Translation correction

The reference reconstructions of the 14 video scenes with LiDAR ground truth (`images.txt`, `images_30fps.txt`,
`points3D.txt`, `points3D.ply`) are offset from the LiDAR point clouds by a constant translation per scene: the
registration to the LiDAR was applied about the wrong centre while exporting the dataset. Rotation and scale are correct, so the correction is a
pure translation `delta` (metres, E / N / U in the frame of `enu_observer.txt` and the LiDAR point clouds), stored as
the 4x4 matrix `[I | delta]` in `<scene>/transform.txt`. The same correction applies to all four model files of a
scene. 

Not affected: `cameras.txt`, the videos, the LiDAR point clouds, `All_RGB_images_day1/2` and the scenes without LiDAR
(Main_stage_area_I/II, Flight_to_tower, Tower, Park_area). Evaluations that align the reconstruction to the LiDAR
by ICP / a similarity transform are not affected.

## Python (numpy only)

```
python apply_translation_correction.py Entrance_III JB3D_translation_correction/Entrance_III/transform.txt Entrance_III_corrected
```

Writes corrected `images.txt`, `images_30fps.txt`, `points3D.txt` and `points3D.ply` to the output folder
(points `X + delta`, camera centres `C + delta`, i.e. `t - R delta`; quaternions and all other lines unchanged).

## COLMAP

`model_transformer` reads the `images.txt` of the input folder; for the 30 fps poses use a folder in which
`images_30fps.txt` is named `images.txt`.

```
S=Entrance_III
mkdir -p ${S}_corrected
colmap model_transformer --input_path $S --output_path ${S}_corrected \
    --transform_path JB3D_translation_correction/$S/transform.txt
colmap model_converter --input_path ${S}_corrected --output_path ${S}_corrected --output_type TXT
colmap model_converter --input_path ${S}_corrected --output_path ${S}_corrected/points3D.ply --output_type PLY

# 30 fps poses
mkdir -p ${S}_30fps ${S}_30fps_corrected
cp $S/cameras.txt $S/points3D.txt ${S}_30fps/ && cp $S/images_30fps.txt ${S}_30fps/images.txt
colmap model_transformer --input_path ${S}_30fps --output_path ${S}_30fps_corrected \
    --transform_path JB3D_translation_correction/$S/transform.txt
colmap model_converter --input_path ${S}_30fps_corrected --output_path ${S}_30fps_corrected --output_type TXT
```

| Scene | delta E [m] | delta N [m] | delta U [m] |
|---|---|---|---|
| Main_stage_I | 0.237 | 0.478 | 0.195 |
| Main_stage_II | 0.053 | 0.107 | 0.059 |
| Flight_to_entrance | 0.163 | -0.411 | 0.240 |
| Entrance_I | 0.212 | -0.555 | 0.245 |
| Second_stage_I | -0.073 | -0.241 | 0.041 |
| Garden_I | 0.058 | -0.054 | -0.022 |
| Food_stalls_I | -0.002 | 0.010 | 0.004 |
| Garden_II | 0.054 | -0.166 | 0.067 |
| Main_stage_III | 0.051 | -0.015 | 0.151 |
| Entrance_II | 0.038 | -0.067 | 0.696 |
| Entrance_III | -0.005 | -0.009 | 0.710 |
| Second_stage_II | -0.023 | 0.021 | 0.019 |
| Food_stalls_II | -0.047 | -0.019 | 0.015 |
| Gardens_III | -0.117 | -0.075 | -0.037 |