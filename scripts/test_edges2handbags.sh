set -ex
# models
RESULTS_DIR='./results/edges2handbags'
G_PATH='./pretrained_models/edges2handbags_net_G.pth'
E_PATH='./pretrained_models/edges2handbags_net_E.pth'

# dataset
CLASS='edges2handbags'
DIRECTION='AtoB' # from domain A to domain B
LOAD_SIZE=256 # scale images to this size
FINE_SIZE=256 # then crop to this size
INPUT_NC=1  # number of channels in the input image

# misc
GPU_ID=0   # gpu id
HOW_MANY=10 # number of input images duirng test
NUM_SAMPLES=10 # number of samples per input images

# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./test.py \
  --dataroot ./datasets/${CLASS} \
  --results_dir ${RESULTS_DIR} \
  --checkpoints_dir ./pretrained_models/ \
  --name ${CLASS} \
  --which_direction ${DIRECTION} \
  --loadSize ${FINE_SIZE} \
  --fineSize ${FINE_SIZE} \
  --input_nc ${INPUT_NC} \
  --how_many ${HOW_MANY} \
  --n_samples ${NUM_SAMPLES} \
  --center_crop \
  --no_flip
