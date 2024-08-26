import os
import random
import numpy as np
import torch
import glob
import re
import string
import collections
from collections import defaultdict
from torch.optim.optimizer import Optimizer
from typing import Dict, List, Tuple
from torch.nn import init
from torch.nn.utils.rnn import pad_sequence


# For collating text data into batches and padding appropriately
def collate(examples, pad_token_id):
    return pad_sequence(examples, batch_first=True, padding_value=pad_token_id)


def collate_fn(pad_token_id):
    def this_collate(examples):
        return collate(examples, pad_token_id)

    return this_collate


def f1_compute_fn(y_true, y_pred, average):
    try:
        from sklearn.metrics import f1_score
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")

    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    return f1_score(y_true, y_pred, average=average)


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (
                classname.find('Conv') != -1 or classname.find('Linear') != -1 or classname.find('Embedding') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'Norm') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)

    if not net:
        return

    print('initialize network with %s' % init_type)
    net.apply(init_func)


'''
	Misc setup code. 
'''


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    os.environ['PYTHONASSEED'] = str(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


'''
	Checkpointing related code. Inherited from Huggingface.
'''


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


'''
    Earlustop class
'''


class EarlyStopper(object):

    def __init__(self, num_trials, save_path, record):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path
        self.record = record

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            if self.record == 1:
                torch.save(model.state_dict(), self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


class offline_saver(object):
    def __init__(self, name="-", path="-", load_name=None):

        self.path = path
        self.save_dir = os.path.join(self.path, name)
        if load_name:
            self.load_dir = os.path.join(self.path, load_name)
        else:
            self.load_dir = self.save_dir

    def put_object(self, path, objects):
        pass

    def get_object(self, objects):
        pass

    def get_object_to_file(self, objects, path):
        pass

    def download_pretrain_model(self, model_name, path):
        pass

    def download_dataset(self, task_name, path):
        pass

    def save(self, model, id, name):
        save_filename = '%s_%s.pth' % (str(id), name)
        save_path = os.path.join(self.save_dir, save_filename)
        try:
            print(save_path)
            torch.save(model.state_dict(), save_path)
        except:
            try:
                torch.save(model.state_dict(), save_path)
            except:
                print("saving failed!")

    def load(self, model, id, name, device="0"):
        save_filename = '%s_%s.pth' % (str(id), name)
        save_path = os.path.join(self.load_dir, save_filename)
        print('loading the model from %s' % save_path)
        try:
            old_state_dict = torch.load(save_path, map_location=str(device))
        except:
            print("load failed!")
            exit()
        model.load_state_dict(old_state_dict)


def nf1_score(prediction, ground_truth):
    prediction_tokens = prediction.split()
    ground_truth_tokens = ground_truth.split()
    if prediction_tokens == ground_truth_tokens:
        return 1
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def normalize_text(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))