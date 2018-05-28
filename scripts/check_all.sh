set -ex

# test code
echo 'test edges2handbags'
bash ./pretrained_models/download_model.sh edges2handbags
bash ./datasets/download_testset.sh edges2handbags
bash ./scripts/test_edges2handbags.sh

echo 'test edges2shoes'
bash ./pretrained_models/download_model.sh edges2shoes
bash ./datasets/download_testset.sh edges2shoes
bash ./scripts/test_edges2shoes.sh

echo 'test facades_label2image'
bash ./pretrained_models/download_model.sh night2day
bash ./datasets/download_testset.sh night2day
bash ./scripts/test_night2day.sh

echo 'test maps'
bash ./pretrained_models/download_model.sh maps
bash ./datasets/download_testset.sh maps
bash ./scripts/test_maps.sh

echo 'test facades'
bash ./pretrained_models/download_model.sh facades
bash ./datasets/download_testset.sh facades
bash ./scripts/test_facades.sh

echo 'test night2day'
bash ./pretrained_models/download_model.sh night2day
bash ./datasets/download_testset.sh night2day
bash ./scripts/test_night2day.sh

echo 'video edges2shoes'
bash ./scripts/video_edges2shoes.sh

echo "train a pix2pix model"
bash ./datasets/download_dataset.sh facades
python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix \
  --which_model_netG unet_256 --which_direction BtoA --lambda_L1 10 --dataset_mode aligned \
  --gan_mode lsgan --norm batch --niter 1 --niter_decay 0 --save_epoch_freq 1
echo "train a bicyclegan model"
python train.py --dataroot ./datasets/facades --name facades_bicycle --model bicycle_gan \
  --which_direction BtoA --dataset_mode aligned \
  --gan_mode lsgan --norm batch --niter 1 --niter_decay 0 --save_epoch_freq 1
