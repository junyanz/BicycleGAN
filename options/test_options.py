from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--results_dir', type=str, default='../results/', help='saves results here.')
        self.parser.add_argument('--phase', type=str, default='val', help='train, val, test, etc')
        self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')
        self.parser.add_argument('--n_samples', type=int, default=5, help='#samples')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--no_encode', action='store_true', help='do not produce encoded image')
        self.parser.add_argument('--sync', action='store_true', help='use the same latent code for different input images')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio for the results')
        self.isTrain = False
