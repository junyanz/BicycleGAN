"""Simple script to make sure basic usage such as training, testing, saving and loading runs without errors."""
import os


def run(command):
    print(command)
    exit_status = os.system(command)
    if exit_status > 0:
        exit(1)


if __name__ == '__main__':
    run('bash ./scripts/test_edges2shoes.sh')
    run('bash ./scripts/test_edges2shoes.sh --sync')
    run('bash ./scripts/video_edges2shoes.sh')
    run('bash ./scripts/train_facades.sh')
