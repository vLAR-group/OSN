# Data preparation

First specify a `${DATASET}` path for your dataset.
Then put the captured training and testing images into `${DATASET}/images_train` and `${DATASET}/images_test` respectively.


## 1. Process training set

### (1) 2D object segmentation and tracking

Use [Track-Anything](https://github.com/gaomingqi/Track-Anything) to process the monocular video from your `${DATASET}/images_train`.
You may prompt the segmentation and tracking by manual click in the 1st frame to ensure the accuracy.
After tracking, you can get the segmentation (`.npy` format) for each frame in your training set.
Put them into `${DATASET}/segm_sam`.

### (2) Optical flow and pixel correspondence

```shell script
cd data_prepare/raft
# Estimate optical flow
python generate_flow.py --dataset_path ${DATASET} --model weights/raft-things.pth
# Extract pixel correspondence
python generate_trajectory.py --data_root ${DATASET} --preproc_path sam_raft --downsample 1.0
```
When extracting pixel correspondence, you can set `--downsample` as `0.1` or `0.01` to use sparse pixel correspondence in the later SfM for efficiency.
Our method is robust to the preprocessing from such sparse pixel correspondence.

**(Optionally)** Extract co-visibility masks, if you want to follow [DyCheck](https://hangg7.com/dycheck/) to evaluate on co-visible pixels only.
```shell script
cd data_prepare/raft
python generate_covis_mask.py --dataset_path_src ${DATASET} --dataset_path_target ${DATASET} --model weights/raft-things.pth
```

### (3) Per-object SfM

Make sure you have installed [COLMAP](https://colmap.github.io/).
```shell script
cd data_prepare
# Initialize COLMAP database
bash colmap_init.sh
# Save previously extracted pixel correspondence into COLMAP database
python colmap_database.py --data_root ${DATASET} --preproc_path sam_raft
# Run SfM for each object
bash colmap_sfm.sh
```

**Note:**
* The SfM fails sometimes, and we find it a good solution to manually specify the initial image ids (`--Mapper.init_image_id1` and `--Mapper.init_image_id2`) for the SfM.
* We assume the known camera intrinsics in this pipeline.
Otherwise, you can choose not to specify them (`--ImageReader.camera_params`) in `colmap_init.sh`, and let the `colmap_sfm.sh` optimize them for you (`--Mapper.ba_refine_focal_length 1`).
It is suggested that you pick the static background in your scene to optimize the camera intrinsics, and then fix the optimized intrinsics for other objects.

### (4) Process for training

```shell script
cd data_prepare
python process_colmap.py ${CONFIG_FILE}
```


## (Optionally) 2. Process testing camera poses

Unless your testing views share the camera poses with the training views (like in NVIDIA Dynamic Scene dataset), you need to process the testing camera poses separately.
The overall solution is to merge the training and testing sets into a whole sequence, and estimate the camera poses along this sequence.
Then the estimated training + testing camera poses here can be aligned into the coordinate frame of training camera poses from the previous step.

### (1) Collect training and testing sets into a whole sequence

We suggest that you specify another `${DATASET_STEREO}` path, and put both training and testing images into `${DATASET_STEREO}/images_train`.
Please carefully arrange the order of these images, which is important for the later SfM (camera pose estimation).
If your training and testing images are captured by a stereo camera (like in Oxford Multimotion dataset), It is suggested that you interleave the training (left camera) and testing (right camera) images in the sequence.

### (2) Process the merged sequence

Follow the steps in 1.(1)~(4) above to process the merged sequence in `${DATASET_STEREO}/images_train`.
Here you only need to run per-object SfM for the static background only.
For 1.(4), remember to temporarily replace `{DATASET}` with `${DATASET_STEREO}` for `data_root` in your `${CONFIG_FILE}`.

### (3) Align the testing camera poses

Align the estimated testing camera poses into the coordinate frame of training camera poses estimated above.
```shell script
cd data_prepare
python align_pose.py
```