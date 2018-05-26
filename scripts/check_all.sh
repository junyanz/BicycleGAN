set -ex
# bash ./datasets/download_dataset.sh facades
# echo "train a pix2pix model"
# python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix \
  # --which_model_netG unet_256 --which_direction BtoA --lambda_L1 10 --dataset_mode aligned \
  # --gan_mode lsgan --norm batch --niter 1 --niter_decay 0 --save_epoch_freq 1
# echo "train a bicyclegan model"
# python train.py --dataroot ./datasets/facades --name facades_bicycle --model bicycle_gan \
  # --which_direction BtoA --dataset_mode aligned \
  # --gan_mode lsgan --norm batch --niter 1 --niter_decay 0 --save_epoch_freq 1
