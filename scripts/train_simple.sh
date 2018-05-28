set -ex
MODEL='bicycle_gan_simple'
CLASS=${1}      # facades, day2night, edges2shoes, edges2handbags, maps
GPU_ID=${2}
DISPLAY_ID=$((GPU_ID*10+1))
NZ=8


CHECKPOINTS_DIR=../checkpoints/${CLASS}/
DATE=`date '+%d_%m_%Y_%H'`
NAME=${CLASS}_${MODEL}_${DATE}


# dataset
NO_FLIP=''
DIRECTION='AtoB'
LOAD_SIZE=286
FINE_SIZE=256
INPUT_NC=3

# dataset parameters
case ${CLASS} in
'facades')
  NITER=100
  NITER_DECAY=100
  SAVE_EPOCH=20
  DIRECTION='BtoA'
  ;;
'edges2shoes')
  NITER=15
  NITER_DECAY=15
  LOAD_SIZE=256
  SAVE_EPOCH=5
  INPUT_NC=1
  NO_FLIP='--no_flip'
  ;;
'edges2handbags')
  NITER=10
  NITER_DECAY=5
  LOAD_SIZE=256
  SAVE_EPOCH=5
  INPUT_NC=1
  ;;
'maps')
  NITER=100
  NITER_DECAY=100
  LOAD_SIZE=600
  SAVE_EPOCH=25
  DIRECTION='BtoA'
  ;;
'day2night')
  NITER=25
  NITER_DECAY=25
  SAVE_EPOCH=10
  ;;
*)
  echo 'WRONG category'${CLASS}
  ;;
esac



# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train.py \
  --display_id ${DISPLAY_ID} \
  --dataroot ./datasets/${CLASS} \
  --name ${NAME} \
  --model ${MODEL} \
  --which_direction ${DIRECTION} \
  --checkpoints_dir ${CHECKPOINTS_DIR} \
  --loadSize ${LOAD_SIZE} \
  --fineSize ${FINE_SIZE} \
  --nz ${NZ} \
  --init_type kaiming \
  --input_nc ${INPUT_NC} \
  --niter ${NITER} \
  --niter_decay ${NITER_DECAY} \
  --use_dropout
