from .base_options import BaseOptions


class VideoOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='../video/', help='saves results here.')
        parser.add_argument('--phase', type=str, default='val', help='train, val, test, etc')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        parser.add_argument('--n_samples', type=int, default=5, help='#samples')
        parser.add_argument('--num_frames', type=int, default=4, help='number of the frames used in the morphing sequence')
        parser.add_argument('--align_mode', type=str, default='horizontal', help='ways of aligning the input images')
        parser.add_argument('--border', type=int, default='0', help='border between results')
        parser.add_argument('--seed', type=int, default=50, help='random seed for latent vectors')
        parser.add_argument('--fps', type=int, default=8, help='speed of the generated video')
        self.isTrain = False
        return parser
