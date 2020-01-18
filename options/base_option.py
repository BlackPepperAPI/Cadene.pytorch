import argparse


class BaseCLIOption:
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    """

    def __init__(self):
        self.phase = None
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--task_name', type=str, default='experiment_1', help='name of the experiment. It decides where to store samples and models (default: experiment_1')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU (default: 0)')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here (default: ./checkpoints)')
        # model parameters
        parser.add_argument('--model', type=str, choices=('fastdvdnet', 'multifiber3dnet'), help='name of model to user. [fastdvdnet | multifiber3dnet]')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal] (default: kaiming)')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal. (default: 0.02)')
        # dataloading parameters
        parser.add_argument('--dataset', type=str, default='UCF101', choices=('UCF101', 'HMDB51', 'Kinetics'), help='name of the dataset to give.')
        parser.add_argument('--dataloader', type=str, default='dali_video', help='chooses how datasets are loaded. [dali_video | cv2_video | ] (default: dali_video)')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=64, help='input batch size (default: 64)')
        parser.add_argument('--load_size', type=int, default=240, help='scale images to this size (default: 240)')
        parser.add_argument('--crop_size', type=int, default=224, help='crop to this size after reshaping (default: 224)')
        parser.add_argument('--sequence_length', type=int default=16, help='clip length for each loading (default: 16)')
        parser.add_argument('--frame_step', type=int, default=-1, help='frame interval between each sequence. use -1 to set step = sequence_length')
        parser.add_argument('--temp_stride', type=int, default=1, help='distance between consecutive frames in sequence (default:1)')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information (default: False)')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opts.name = opts.name + suffix: e.g., {model}_{dataset}')

        self.initialized = True

    def print_options(self, opts):
        """ Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opts.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opts).items()):
            message += '.. {:>25}: {:<30}\n'.format(str(k), str(v))
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opts.checkpoints_dir, opts.task_name)
        # util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opts.txt'.format(self.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse_args(self):
        """ Parse our options, create checkpoints directory suffix, and set up gpu device.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # process opts.suffix
        if opts.suffix:
            suffix = ('_' + opts.suffix.format(**vars(opts))) if opts.suffix != '' else ''
            opts.name = opts.name + suffix

        self.print_options(opts)

        # set gpu ids
        str_ids = opts.gpu_ids.split(',')
        opts.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opts.gpu_ids.append(id)
        if len(opts.gpu_ids) > 0:
            torch.cuda.set_device(opts.gpu_ids[0])

        self.opts = opts
        return self.opts
