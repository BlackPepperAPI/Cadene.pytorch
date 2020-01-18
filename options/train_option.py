from .base_option import BaseCLIOption


class TrainCLIOptions(BaseCLIOption):
    """Training options parser from CLI/Argv

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        # network saving and loading parameters
        parser.add_argument('--model_checkpoint', type=str, help='continue training: load the latest model')
        # training parameters
        parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--momentum', type=float, default=0.5, help='momentum term for optimizer')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_iters', type=int, default=50, help='frequency of multiply by a gamma every lr_iters iterations')

        self.isTrain = True
        self.phase = 'train'

        return parser
