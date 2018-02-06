import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import save_images
from itertools import islice
from util import html
import numpy as np


# helper function
def get_random_z(opt):
    z_samples = np.random.normal(0, 1, (opt.n_samples + 1, opt.nz))
    return z_samples


# options
opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads=1
opt.batchSize = 1   # test code only supports batchSize=1
opt.serial_batches = True  # no shuffle

# create dataset
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
model.eval()
print('Loading model %s' % opt.model)

# create website
web_dir = os.path.join(opt.results_dir, opt.phase +
                       '_sync' if opt.sync else opt.phase)
webpage = html.HTML(web_dir, 'Training = %s, Phase = %s, G = %s, E = %s' % (
    opt.name, opt.phase, opt.G_path, opt.E_path))

# sample random z
if opt.sync:
    z_samples = get_random_z(opt)

# test stage
for i, data in enumerate(islice(dataset, opt.how_many)):
    model.set_input(data)
    print('process input image %3.3d/%3.3d' % (i, opt.how_many))
    if not opt.sync:
        z_samples = get_random_z(opt)
    for nn in range(opt.n_samples + 1):
        encode_B = nn == 0 and not opt.no_encode
        _, real_A, fake_B, real_B, _ = model.test_simple(
            z_samples[nn], encode_real_B=encode_B)
        if nn == 0:
            all_images = [real_A, real_B, fake_B]
            all_names = ['input', 'ground truth', 'encoded']
        else:
            all_images.append(fake_B)
            all_names.append('random sample%2.2d' % nn)

    img_path = 'input image%3.3i' % i
    save_images(webpage, all_images, all_names, img_path, None,
                width=opt.fineSize, aspect_ratio=opt.aspect_ratio)

webpage.save()
