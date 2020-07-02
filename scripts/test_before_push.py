"""Simple script to make sure basic usage such as training, testing, saving and loading runs without errors."""
import os


def run(command):
    print(command)
    exit_status = os.system(command)
    if exit_status > 0:
        exit(1)


if __name__ == '__main__':
    if not os.path.exists('./datasets/mini_pix2pix'):
        run('bash ./datasets/download_mini_dataset.sh mini_pix2pix')

    # pix2pix train/test
    run('python train.py --model pix2pix --name temp_pix2pix --dataroot ./datasets/mini_pix2pix --niter 1 --niter_decay 0 --save_latest_freq 10 --display_id -1')

    # template train/test
    run('python train.py --model template --name temp2 --dataroot ./datasets/mini_pix2pix --niter 1 --niter_decay 0 --save_latest_freq 10 --display_id -1')

    run('bash ./scripts/test_edges2shoes.sh')
    run('bash ./scripts/test_edges2shoes.sh --sync')
    run('bash ./scripts/video_edges2shoes.sh')
    # run('bash ./scripts/train_facades.sh')
