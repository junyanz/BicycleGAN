set -ex
# models
RESULTS_DIR='./results/night2day'
G_PATH='./pretrained_models/night2day_net_G.pth'
E_PATH='./pretrained_models/night2day_net_E.pth'

# dataset
CLASS='night2day'
DIRECTION='AtoB' # from domain A to domain B
LOAD_SIZE=286 # scale images to this size
CROP_SIZE=256 # then crop to this size
INPUT_NC=3  # number of channels in the input image
ASPECT_RATIO=1.4
 # change aspect ratio for the test images

# misc
GPU_ID=0   # gpu id
NUM_TEST=10 # number of input images duirng test
NUM_SAMPLES=10 # number of samples per input images


# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./test.py \
  --dataroot ./datasets/${CLASS} \
  --results_dir ${RESULTS_DIR} \
  --checkpoints_dir ./pretrained_models/ \
  --name ${CLASS} \
  --direction ${DIRECTION} \
  --load_size ${LOAD_SIZE} \
  --crop_size ${CROP_SIZE} \
  --input_nc ${INPUT_NC} \
  --num_test ${NUM_TEST} \
  --n_samples ${NUM_SAMPLES} \
  --aspect_ratio ${ASPECT_RATIO} \
  --center_crop \
  --no_flip
