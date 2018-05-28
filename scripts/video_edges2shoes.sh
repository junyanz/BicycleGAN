set -ex
# models
RESULTS_DIR='./videos/edges2shoes'
G_PATH='./pretrained_models/edges2shoes_net_G.pth'

# dataset
CLASS='edges2shoes'
DIRECTION='AtoB'
LOAD_SIZE=256
FINE_SIZE=256
INPUT_NC=1

# misc
GPU_ID=0
HOW_MANY=5 # number of input images duirng test
NUM_SAMPLES=20 # number of samples per input images

# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./video.py \
  --dataroot ./datasets/${CLASS} \
  --results_dir ${RESULTS_DIR} \
  --checkpoints_dir ./pretrained_models/ \
  --name ${CLASS} \
  --which_direction ${DIRECTION} \
  --loadSize ${FINE_SIZE} --fineSize ${FINE_SIZE} \
  --input_nc ${INPUT_NC} \
  --how_many ${HOW_MANY} \
  --n_samples ${NUM_SAMPLES} \
  --center_crop \
  --no_flip
