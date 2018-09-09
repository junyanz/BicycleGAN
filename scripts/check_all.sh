set -ex
DOWNLOAD_MODEL=${1}
# test code
echo 'test edges2handbags'
if [ ${DOWNLOAD_MODEL} -eq 1 ]
then
  bash ./pretrained_models/download_model.sh edges2handbags
fi
bash ./datasets/download_testset.sh edges2handbags
bash ./scripts/test_edges2handbags.sh

echo 'test edges2shoes'
if [ ${DOWNLOAD_MODEL} -eq 1 ]
then
  bash ./pretrained_models/download_model.sh edges2shoes
fi
bash ./datasets/download_testset.sh edges2shoes
bash ./scripts/test_edges2shoes.sh

echo 'test facades_label2image'
if [ ${DOWNLOAD_MODEL} -eq 1 ]
then
  bash ./pretrained_models/download_model.sh night2day
fi
bash ./datasets/download_testset.sh night2day
bash ./scripts/test_night2day.sh

echo 'test maps'
if [ ${DOWNLOAD_MODEL} -eq 1 ]
then
  bash ./pretrained_models/download_model.sh maps
fi
bash ./datasets/download_testset.sh maps
bash ./scripts/test_maps.sh

echo 'test facades'
if [ ${DOWNLOAD_MODEL} -eq 1 ]
then
  bash ./pretrained_models/download_model.sh facades
fi
bash ./datasets/download_testset.sh facades
bash ./scripts/test_facades.sh

echo 'test night2day'
if [ ${DOWNLOAD_MODEL} -eq 1 ]
then
  bash ./pretrained_models/download_model.sh night2day
fi
bash ./datasets/download_testset.sh night2day
bash ./scripts/test_night2day.sh

echo 'video edges2shoes'
bash ./scripts/video_edges2shoes.sh

echo "train a pix2pix model"
bash ./datasets/download_dataset.sh facades
python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix \
  --netG unet_256 --direction BtoA --lambda_L1 10 --dataset_mode aligned \
  --gan_mode lsgan --norm batch --niter 1 --niter_decay 0 --save_epoch_freq 1
echo "train a bicyclegan model"
python train.py --dataroot ./datasets/facades --name facades_bicycle --model bicycle_gan \
  --direction BtoA --dataset_mode aligned \
  --gan_mode lsgan --norm batch --niter 1 --niter_decay 0 --save_epoch_freq 1
