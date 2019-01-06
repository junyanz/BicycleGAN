set -ex
# models
RESULTS_DIR='./videos/edges2shoes'
G_PATH='./pretrained_models/edges2shoes_net_G.pth'

# dataset
CLASS='edges2shoes'
DIRECTION='AtoB'
LOAD_SIZE=256
CROP_SIZE=256
INPUT_NC=1

# misc
GPU_ID=0
NUM_TEST=5 # number of input images duirng test
NUM_SAMPLES=20 # number of samples per input images

# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./video.py \
  --dataroot ./datasets/${CLASS} \
  --results_dir ${RESULTS_DIR} \
  --checkpoints_dir ./pretrained_models/ \
  --name ${CLASS} \
  --direction ${DIRECTION} \
  --load_size ${LOAD_SIZE} --crop_size ${CROP_SIZE} \
  --input_nc ${INPUT_NC} \
  --num_test ${NUM_TEST} \
  --n_samples ${NUM_SAMPLES} \
  --center_crop \
  --no_flip
