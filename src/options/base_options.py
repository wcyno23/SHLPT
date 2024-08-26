import argparse
import os
import re
import torch
import time

from utils import utils
from io import BytesIO
from options.pre_args import *


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initialized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument("--comment", default='', type=str, help="The comment")
        parser.add_argument('--name', type=str, default="test")
        parser.add_argument('--output_dir', type=str, default="../ablation")
        parser.add_argument('--cache_dir', type=str, default="../../cache")
        parser.add_argument("--continual_train", action='store_true',
                            help="Whether restore from the last checkpoint, is no checkpoints, start from scratch")
        parser.add_argument("--block_size", default=512, type=int,
                            help="ptional input sequence length after tokenization."
                                 "The training dataset will be truncated in block of this size for training."
                                 "Default to the model max input length for single sentence inputs (take into account special tokens)")

        parser.add_argument("--dataset_set", default=None, type=str,
                            help="The setting of dataset")

        # downstream dataset (labeled) parameters
        parser.add_argument("--data_cache_dir", default="../../dataset", type=str,
                            help="The tasks' names")
        parser.add_argument("--train_tasks", default=None, type=str,
                            help="The input training data file(s)")
        parser.add_argument("--pri_task", default="seq_cla", type=str,
                            help="The task of primary task(s)")
        parser.add_argument("--max_datalen", default=-1, type=int,
                            help="Optional input sequence length after tokenization.")
        parser.add_argument("--test_max_datalen", default=-1, type=int,
                            help="Optional input sequence length after tokenization.")

        # task generator parameters
        parser.add_argument("--task_set", default=None, type=str,
                            help="The setting of training task(s)")
        parser.add_argument("--task_order", default=None, type=str,
                            help="The order of training task(s)")
        parser.add_argument("--task_gen_type", default=0, type=int,
                            help="0 for block stream, 1 for sample stream, 2 for mix stream")
        parser.add_argument("--task_split", default=None, type=str,
                            help="The split part of tasks to be mixed")
        parser.add_argument("--task_chang_freq", default=1, type=int,
                            help="The freq to change task in the same part")
        parser.add_argument("--task_num", default=1, type=int,
                            help="The number of majority datasets in each task")
        parser.add_argument("--major_num", default=1, type=int,
                            help="The number of majority datasets in each task")
        parser.add_argument("--major_step", default=-1, type=int,
                            help="The number of majority datasets in each task")
        parser.add_argument("--major_prob", default=0.8, type=float,
                            help="The number of majority datasets in each task")
        parser.add_argument("--task_boundary", default=1, type=int,
                            help="The number of majority datasets in each task")
        parser.add_argument("--corpus_involved", default=0, type=int,
                            help="The number of majority datasets in each task")
        parser.add_argument("--cover_width", default=3, type=int,
                            help="The number of majority datasets in each task")
        parser.add_argument("--scenario", default='til', type=str,
                            help="the problem setting for continual learning ")
        parser.add_argument("--sample_method", default=None, type=str,
                            help="the problem setting for continual learning ")

        parser.add_argument("--task_definition", default="dataset", type=str,
                            help="the problem setting for continual learning ")

        # model parameters
        parser.add_argument("--model_set", default=None, type=str,
                            help="The setting of training task(s)")
        parser.add_argument("--model_name_or_path", default='roberta-base', type=str,
                            help="Path to pre-trained model or shortcut name selected in the list: ")
        parser.add_argument("--model_type", default='roberta-base', type=str,
                            help="Model type selected in the list")
        parser.add_argument("--config_name", default='roberta-base', type=str,
                            help="Path to pre-trained model or shortcut name selected in the list: ")
        parser.add_argument("--tokenizer_name", default='roberta-base', type=str,
                            help="Path to pre-trained model or shortcut name selected in the list: ")
        parser.add_argument("--init_type", default='normal', type=str,
                            help="Path to pre-trained model or shortcut name selected in the list: ")
        parser.add_argument("--max_seq_len", type=int, default=256,
                            help="max length of token sequence")
        parser.add_argument("--betas", type=str, default="(0.9,0.999)",
                            help="The initial beta for Adam.")
        parser.add_argument("--learning_rate", default=3e-5, type=float,
                            help="The initial learning rate for Adam.")
        parser.add_argument("--base_wd", type=float, default=0.01)
        parser.add_argument("--weight_decay", default=0.0, type=float,
                            help="Weight decay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                            help="Epsilon for Adam optimizer.")
        parser.add_argument('--froze_base', action='store_true',
                            help="froze the base model in parameter optimization")
        parser.add_argument("--clf_head", default='linear', type=str,
                            help="the type of classification head ")

        # optimize parameters
        parser.add_argument("--train_set", default=None, type=str,
                            help="The setting of training task(s)")
        parser.add_argument("--per_gpu_train_batch_size", default=2, type=int,
                            help="Batch size per GPU/CPU for training.")
        parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
                            help="Batch size per GPU/CPU for evaluation.")
        parser.add_argument("--warmup_frac", type=float, default=0.1)
        parser.add_argument("--max_grad_norm", default=1.0, type=float,
                            help="Max gradient norm.")
        parser.add_argument("--epoch", default=1.0, type=float,
                            help="Total number of training epochs to perform.")
        parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
        parser.add_argument("--eval_every", type=int, default=300,
                            help="Frequency with which to evaluate the model")
        parser.add_argument('--eval_iter', action='store_true',
                            help="whether to evaluate each n iteration")
        parser.add_argument("--loss_every", type=int, default=100,
                            help="Frequency with which to display the loss")
        parser.add_argument('--seed', type=int, default=0,
                            help="random seed for initialization")
        parser.add_argument('--is_save_log_disc', action='store_false',
                            help="save log to the disc")
        parser.add_argument('--save_model', action='store_true',
                            help="save log to the disc")
        parser.add_argument('--save_model_online', action='store_true',
                            help="save log to the disc")
        parser.add_argument('--load_name', default=None, type=str,
                            help="the path to load model")

        # parameter for continual learning algo 
        parser.add_argument('--algo', type=str, default="basic")
        parser.add_argument('--load_algo_args', action='store_true',
                            help="load the args from the predefined args")
        parser.add_argument('--reg_on_base', action='store_true',
                            help="whether the basic feature extractor should be regularized")
        parser.add_argument('--reg_on_head', action='store_true',
                            help="whether the head should be regularized")
        parser.add_argument('--no_reg_on_unlabeled', action='store_true',
                            help="whether the training of unlabeled data be regularized")

        # parameter for l2 regularization 
        parser.add_argument("--l2w", default=0.0, type=float, help="l2 regularization weight")

        # parameter for ewc regularization 
        parser.add_argument("--ewc_gamma", default=0.9, type=float, help="the gamma weight for online ewc")
        parser.add_argument("--ewc_lamb", default=0.7, type=float, help="ewc regularization weight")

        # parameter for memory based    
        parser.add_argument('--memory_size', type=int, default=1000,
                            help="the size of memory can be used")
        parser.add_argument('--memory_mode', type=str, default="reservoir",
                            help="the mode to store and sample data")
        parser.add_argument("--m_beta", default=0.5, type=float, help="the gamma weight for online ewc")
        parser.add_argument("--m_alpha", default=0.5, type=float, help="ewc regularization weight")
        # parser.add_argument('--replay_freq', type=int, default=4, help="the frequency of memory replay")
        # parser.add_argument('--replay_size', type=int, default=1, help="the number of data used in one memory replay")


        parser.add_argument("--local_rank", type=int, default=-1,
                            help="For distributed training: local_rank")
        parser.add_argument("--use_pre_prompt", action="store_true", help="whether use previous prompt to initialize the first prompt")
        parser.add_argument("--prompt_path", type=str, help="the saved prompt ckpt path for initialize or progressive prompts test")
        parser.add_argument("--prompt_number", type=int, default=100, help="the number of prompt")
        parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient accumulation steps")

        # parameter for distributed deployment
        parser.add_argument("--distributed", action="store_true")
        parser.add_argument("--n_gpu", type=int, default=1, help="gpu number")
        parser.add_argument("--save_path", type=str, default="../../ckpt/conll_ckpt", help="model checkpoint save path")
        parser.add_argument("--use_scaler", action="store_true", help="whether use scaler")
        # parameter for Prompt tuning
        parser.add_argument("--bottleneck_size", type=int, default=1200, help="MLP bottleneck size")
        parser.add_argument("--mlp_layer_norm", action="store_true", help="Do layer norm in mlp")
        parser.add_argument("--prefix_mlp", action="store_true", help="whether to prefix a mlp before prepend prompt")
        parser.add_argument("--fix_mlp_prompt", action="store_true")
        parser.add_argument("--use_hat_mask", action="store_true")
        parser.add_argument("--hard_mask", action="store_true", help="use hard mask or soft mask")
        parser.add_argument("--init_from_previous", action="store_true", help="whether init prompt from previous prompt")
        parser.add_argument("--init_from_random_tokens", action="store_true", help="the way to init new task prompt")
        parser.add_argument("--k_train", type=int, default=-1, help='Select k examples each class from train set')
        parser.add_argument("--k_val", type=int, default=-1, help='Select k samples each class consists of validation set')
        parser.add_argument("--k_test", type=int, default=-1, help='Select k samples each class from test set')

        parser.add_argument("--first_k_split", action="store_true", help="Use first k samples as valid set")
        parser.add_argument("--split_test_file", action="store_true", help="whether split testset into valid set and test set")

        parser.add_argument("--split_train_file_to_valid_and_test", action="store_true", help="whether split trainset into train set, valid set and test set. This option is activated during hyperparameter search")
        parser.add_argument("--split_train_file_to_valid", action="store_true", help="whether split trainset into train set and valid set")
        parser.add_argument("--gpu_id", type=int, default=0, help="gpu id for setting device")
        parser.add_argument("--progressive_init", action="store_true", help="whether initialize like progressive prompts")
        # parameter for Progressive Prompts
        parser.add_argument("--progressive", action="store_true", help="whether prepend previous prompts in later task")
        # parameter for SHLPT
        parser.add_argument("--load_pre_prompts", action="store_true", help="load previous prompts for convenient testing")
        parser.add_argument("--pre_prompts_path", type=str, help="the path for loading previous prompts")
        parser.add_argument("--pre_prompt_number", type=int, default=0, help="the number of previous prompts")
        parser.add_argument("--estimator", type=str, default="attempt", help="task similarity estimator type")
        parser.add_argument("--softmax_temperature", type=float, default=2000)
        parser.add_argument("--involve_current_prompt", action="store_true", help="whether involve current optimized prompt when estimate similarity")
        parser.add_argument("--threshold", type=float, default=0.01, help="the threshold to determine dissimilar task")
        parser.add_argument("--involve_random_prompt", action="store_true", help="whether involve random prompt to better split similar and dissimilar task")
        parser.add_argument("--fixed_value", type=str, default="0.25,0.25,0.25,0.25", help="fixed similarity value")
        parser.add_argument("--add_prompt_lambda", action="store_true", help="add lambda for better composing previous prompts and current prompts")
        parser.add_argument("--add_kl_dis", action="store_true", help="add kl loss between current task and dissimilar task")
        parser.add_argument("--add_hsc_loss", action="store_true", help="add contrastive loss of output hidden states between current and dissimilar tasks")
        parser.add_argument("--hsc_lamb", type=float, default=1, help="the lambda weight for Hidden States Contrastive loss")
        parser.add_argument("--hsc_temperature", type=float, default=0.5)
        parser.add_argument("--add_asc_loss_unfiltered", action="store_true", help="add activation loss of ffn activated neurons between current and dissimilar tasks")
        parser.add_argument("--add_asc_loss_filtered", action="store_true", help="add activation loss of ffn activated neurons between current and dissimilar tasks. In addition, the neuron activated by instance X is filtered.")
        parser.add_argument("--layers", type=list, default=[23], help="the layers to calculate activation loss")
        parser.add_argument("--asc_lamb", type=float, default=1, help="the lambda weight for Activation States Contrastive loss")
        parser.add_argument("--asc_temperature", type=float, default=0.5)
        parser.add_argument("--all_dissimilar", action="store_true", help="previous tasks are all dissimilar to current task")
        parser.add_argument("--use_ste", action="store_true", help="whether use straight through estimator, if yes the activation states will be binarized")
        parser.add_argument("--accum_epoch", type=int, default=25, help="the epoch to early stop")
        parser.add_argument("--mean", action="store_true", help="whether use mean to calculate similarity")
        parser.add_argument("--add_not", action="store_true", help="whether add not to calulate similarity")
        parser.add_argument("--visualize", action="store_true", help="whether visualize the similarity")
        parser.add_argument("--act_path", type=str, help="the path to save activation states")
        parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
        self.initialized = True

        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            self.parser = self.initialize(parser)
        # save and return the parser
        args = parser.parse_args()
        return args

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        file_name = os.path.join(opt.output_dir, 'train_opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt = self.load_pre_args(opt)

        name_prefix = time.strftime('%m-%d-%H:%M:%S') + "_" + str(opt.name) + "_" + str(opt.algo) + "_" + str(
            opt.model_type)
        opt.model_name = name_prefix
        opt.output_dir = os.path.join(opt.output_dir, opt.model_name)
        utils.mkdirs(opt.output_dir)

        opt.is_save_log = opt.is_save_log_oss or opt.is_save_log_disc
        opt.save_model = True if opt.save_model_online else opt.save_model
        if opt.is_save_log_disc:
            opt.disc_log_path = os.path.join(opt.output_dir, 'train_log.txt')

        self.print_options(opt)

        opt.train_tasks = opt.train_tasks.split(":") if opt.train_tasks else []
        opt.task_order = opt.task_order.split(":")  # example: order = ["a1,d1,d2,"]
        opt.task_split = opt.task_split.split(":") if opt.task_split else []  # example: order = ["a1,d1,d2,"]
        opt.task_split.append("-1")
        if len(opt.task_order) != len(opt.train_tasks):
            raise ValueError("Unmatched order with the provided dataset")

        opt.model_name_or_path = os.path.join(opt.cache_dir, opt.model_type)
        self.opt = opt

        return self.opt

    def load_pre_args(self, opt):
        if opt.task_set is not None:
            args = task_args[opt.task_set]
            for key, value in args.items():
                setattr(opt, key, value)

        if opt.dataset_set is not None:
            args = dataset_args[opt.dataset_set]
            for key, value in args.items():
                setattr(opt, key, value)

        if opt.model_set is not None:
            args = model_args[opt.model_set]
            for key, value in args.items():
                setattr(opt, key, value)

        if opt.train_set is not None:
            args = train_args[opt.train_set]
            for key, value in args.items():
                setattr(opt, key, value)

        if opt.load_algo_args:
            args = algo_args[opt.algo]
            for key, value in args.items():
                setattr(opt, key, value)

        return opt
