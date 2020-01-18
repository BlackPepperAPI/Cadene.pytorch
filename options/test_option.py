from .base_option import BaseCLIOption


class TestCLIOptions(BaseCLIOption):
    """Test options parser from CLI/Argv

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = super().initialize(self, parser)

        parser.add_argument('--model_checkpoint', type=str, required=True, help='path to the checkpoint model file')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results artifact e.g. plotting, images/videos here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')

        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(crop_size=parser.get_default('load_size'))

        self.isTrain = False
        self.phase = 'test'

        return parser
