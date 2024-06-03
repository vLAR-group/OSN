#DATA_ROOT='/media/SSD/ziyang/Datasets_NeRF/indoor/data/chessboard++'
#PREPROC_PATH='colmap_init'
#
#mkdir -p ${DATA_ROOT}/${DATASET_NAME}/${PREPROC_PATH}
#colmap feature_extractor \
#  --database_path ${DATA_ROOT}/${DATASET_NAME}/${PREPROC_PATH}/database.db \
#  --image_path ${DATA_ROOT}/${DATASET_NAME}/images_train \
#  --ImageReader.single_camera 1 \
#  --ImageReader.camera_model SIMPLE_PINHOLE \
#  --ImageReader.camera_params "444.44441923584253, 320.0, 240.0"
#
#colmap exhaustive_matcher \
#  --database_path ${DATA_ROOT}/${DATASET_NAME}/${PREPROC_PATH}/database.db



DATA_ROOT='/media/SSD/ziyang/Datasets_NeRF/iPhone/data/dragons'
PREPROC_PATH='colmap_init'

mkdir -p ${DATA_ROOT}/${DATASET_NAME}/${PREPROC_PATH}
colmap feature_extractor \
  --database_path ${DATA_ROOT}/${DATASET_NAME}/${PREPROC_PATH}/database.db \
  --image_path ${DATA_ROOT}/${DATASET_NAME}/images_train \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model SIMPLE_PINHOLE \
  --ImageReader.camera_params "719.0, 320.0, 180.0"

colmap exhaustive_matcher \
  --database_path ${DATA_ROOT}/${DATASET_NAME}/${PREPROC_PATH}/database.db