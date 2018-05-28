from .base_options import BaseOptions


class VideoOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--results_dir', type=str, default='../video/', help='saves results here.')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--phase', type=str, default='val', help='train, val, test, etc')
        self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')
        self.parser.add_argument('--n_samples', type=int, default=5, help='#samples')
        self.parser.add_argument('--num_frames', type=int, default=4, help='number of the frames used in the morphing sequence')
        self.parser.add_argument('--align_mode', type=str, default='horizontal', help='ways of aligning the input images')
        self.parser.add_argument('--border', type=int, default='0', help='border between results')
        self.parser.add_argument('--seed', type=int, default=50, help='random seed for latent vectors')
        self.parser.add_argument('--fps', type=int, default=8, help='speed of the generated video')
        self.isTrain = False
