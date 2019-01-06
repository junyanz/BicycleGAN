from options.video_options import VideoOptions
from data import create_dataset
from models import create_model
from itertools import islice
from util import util
import numpy as np
import moviepy.editor
import os
import torch


def get_random_z(opt):
    z_samples = np.random.normal(0, 1, (opt.n_samples + 1, opt.nz))
    return z_samples


def produce_frame(t):
    k = int(t * opt.fps)
    return np.concatenate(frame_rows[k], axis=1 - use_vertical)


# hard-code opt
opt = VideoOptions().parse()
opt.num_threads = 1   # test code only supports num_threads=1
opt.batch_size = 1   # test code only supports batch_size=1
opt.no_encode = True  # do not use encoder

dataset = create_dataset(opt)
model = create_model(opt)
model.setup(opt)
model.eval()
interp_mode = 'slerp'
use_vertical = 1 if opt.align_mode == 'vertical' else 0

print('Loading model %s' % opt.model)
# create website
results_dir = opt.results_dir
util.mkdir(results_dir)
total_frames = opt.num_frames * opt.n_samples


z_samples = get_random_z(opt)
frame_rows = [[] for n in range(total_frames)]

for i, data in enumerate(islice(dataset, opt.num_test)):
    print('process input image %3.3d/%3.3d' % (i, opt.num_test))
    model.set_input(data)
    real_A = util.tensor2im(model.real_A)
    wb = opt.border
    hb = opt.border
    h = real_A.shape[0]
    w = real_A.shape[1]   # border
    real_A_b = np.full((h + hb, w + wb, opt.output_nc), 255, real_A.dtype)
    real_A_b[hb:, wb:, :] = real_A
    frames = [[real_A_b] for n in range(total_frames)]

    for n in range(opt.n_samples):
        z0 = z_samples[n]
        z1 = z_samples[n + 1]
        zs = util.interp_z(z0, z1, num_frames=opt.num_frames, interp_mode=interp_mode)
        for k in range(opt.num_frames):
            zs_k = (torch.Tensor(zs[[k]])).to(model.device)
            _, fake_B_device, _ = model.test(zs_k, encode=False)
            fake_B = util.tensor2im(fake_B_device)
            fake_B_b = np.full((h + hb, w + wb, opt.output_nc), 255, fake_B.dtype)
            fake_B_b[hb:, wb:, :] = fake_B
            frames[k + opt.num_frames * n].append(fake_B_b)

    for k in range(total_frames):
        frame_row = np.concatenate(frames[k], axis=use_vertical)
        frame_rows[k].append(frame_row)

# compile it to a vdieo
images_dir = os.path.join(results_dir, 'frames_seed%4.4d' % opt.seed)
util.mkdir(images_dir)


for k in range(total_frames):
    final_frame = np.concatenate(frame_rows[k], axis=1 - use_vertical)
    util.save_image(final_frame, os.path.join(
        images_dir, 'frame_%4.4d.jpg' % k))


video_file = os.path.join(
    results_dir, 'morphing_video_seed%4.4d_fps%d.mp4' % (opt.seed, opt.fps))
video = moviepy.editor.VideoClip(
    produce_frame, duration=float(total_frames) / opt.fps)
video.write_videofile(video_file, fps=30, codec='libx264', bitrate='16M')
