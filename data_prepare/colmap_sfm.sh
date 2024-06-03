#DATASET_PATH='/media/SSD/ziyang/Datasets_NeRF/indoor/data/chessboard++'
#MOSEG_DIR='sam_raft'
#
##for OBJ_ID in 0 1 2 3
##for OBJ_ID in 0
#for OBJ_ID in 1 2 3 4 5 6 7 8
#do
#  colmap exhaustive_matcher \
#  --database_path ${DATASET_PATH}/${MOSEG_DIR}/database${OBJ_ID}.db
#
#  mkdir ${DATASET_PATH}/${MOSEG_DIR}/sparse${OBJ_ID}
#  colmap mapper \
#      --database_path ${DATASET_PATH}/${MOSEG_DIR}/database${OBJ_ID}.db \
#      --image_path ${DATASET_PATH}/images_train \
#      --output_path ${DATASET_PATH}/${MOSEG_DIR}/sparse${OBJ_ID} \
#      --Mapper.num_threads 16 \
#      --Mapper.init_min_tri_angle 4 \
#      --Mapper.multiple_models 0 \
#      --Mapper.extract_colors 0 \
#      --Mapper.ba_refine_focal_length 0 \
#      --Mapper.ba_refine_extra_params 0 \
##      --Mapper.init_image_id1 14 --Mapper.init_image_id2 10
#done



DATASET_PATH='/media/SSD/ziyang/Datasets_NeRF/iPhone/data/dragons'
MOSEG_DIR='sam_raft01'

for OBJ_ID in 3
do
  colmap exhaustive_matcher \
  --database_path ${DATASET_PATH}/${MOSEG_DIR}/database${OBJ_ID}.db

  mkdir ${DATASET_PATH}/${MOSEG_DIR}/sparse${OBJ_ID}
  colmap mapper \
      --database_path ${DATASET_PATH}/${MOSEG_DIR}/database${OBJ_ID}.db \
      --image_path ${DATASET_PATH}/images_train \
      --output_path ${DATASET_PATH}/${MOSEG_DIR}/sparse${OBJ_ID} \
      --Mapper.num_threads 16 \
      --Mapper.init_min_tri_angle 4 \
      --Mapper.multiple_models 0 \
      --Mapper.extract_colors 0 \
      --Mapper.ba_refine_focal_length 0 \
      --Mapper.ba_refine_extra_params 0 \
      --Mapper.init_image_id1 1 --Mapper.init_image_id2 9
done