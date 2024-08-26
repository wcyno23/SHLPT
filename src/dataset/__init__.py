import torch
import json
from utils.utils import collate_fn

from dataset.config import *


def load_labeled_dataset(args, tokenizer, train_type="train", label_maps=None, label2task_id=None,
                         split_test_set=False, split_train_set_to_valid=False,
                         split_train_set_to_valid_and_test=False, ):
    task_names = args.train_tasks
    num_labels = []
    datasets = {}
    label_maps = label_maps if label_maps else {}
    label2task_id = label2task_id if label2task_id else {"num": -1, "split_type": args.task_definition}
    task_to_target_len = {
        'rte': 6,
        'mrpc': 5,
        'sst2': 2,
        'qqp': 6,
        'cola': 5,
        'qnli': 5,
        'mnli': 5,
        'stsb': 3,

        'wic': 2,
        'boolq': 2,
        'copa': 2,
        'wsc': 3,
        'wsc_bool': 2,
        'cb': 5,
        'multirc': 5,
        'record': 10,
        'rte_superglue': 5,

        'imdb': 2,

        'ag_news': 2,
        'yahoo_answers_topics': 5,
        'dbpedia_14': 5,
        'amazon_new': 2,
        'yelp': 2,
        'squad': 76,
        'srl': 146,

        'AGNews': 2,
        'yahoo': 5,
        'dbpedia': 5,
        'amazon': 2,
    }
    T5_TASK_LIST = set(T5_JSON_TASK_NAME_LIST + T5_JSONL_TASK_NAME_LIST + T5_CSV_TASK_NAME_LIST + T5_TSV_TASK_NAME_LIST)
    # Dividing a portion of samples from the test set as validation set.
    if split_test_set:
        validDatasets = {}
        testDatasets = {}
        for idx, task_name in enumerate(task_names):
            if task_name in VALID_FIlE_ONLY_TASK_LIST:
                target_len = task_to_target_len[task_name]
                if args.algo in ["t5er", "t5derpp"]:
                    if args.dataset_set == "dis_seq1":
                        target_len = 6
                    elif args.dataset_set == "dis_seq2":
                        target_len = 76
                    elif args.dataset_set == "dis_seq3":
                        target_len = 76
                    elif args.dataset_set in ["scl_seq1", "scl_seq2", "scl_seq3"]:
                        target_len = 5
                    elif args.dataset_set in ['long_seq1', 'long_seq2', 'long_seq3']:
                        target_len = 6
                    elif args.dataset_set in ['decanlp1', 'decanlp2', 'decanlp3']:
                        target_len = 146
                from dataset.t5_dataset import T5Dataset
                Dataset = T5Dataset(args, tokenizer, task_name, idx, args.max_seq_len, train_type, lazy=False,
                                    split=True, target_len=target_len)
                validDataset = T5Dataset(args, tokenizer, task_name, idx, args.max_seq_len, 'valid',
                                         examples=Dataset.dev_examples, target_len=target_len)
                testDataset = T5Dataset(args, tokenizer, task_name, idx, args.max_seq_len, 'test',
                                        examples=Dataset.test_examples, target_len=target_len)
                label_maps[task_name] = validDataset.label_map
                label2task_id = validDataset.label2task_id
                validDatasets[task_name] = CustomDatasetDataLoader(args, validDataset, args.per_gpu_eval_batch_size,
                                                                   args.pri_task,
                                                                   task_name, shuffle=False)
                testDatasets[task_name] = CustomDatasetDataLoader(args, testDataset, args.per_gpu_eval_batch_size,
                                                                  args.pri_task,
                                                                  task_name, shuffle=False)
                del Dataset
            elif task_name in T5_TSV_TASK_NAME_LIST:
                target_len = task_to_target_len[task_name]
                if args.algo in ["t5er", "t5derpp"]:
                    if args.dataset_set == "dis_seq1":
                        target_len = 6
                    elif args.dataset_set == "dis_seq2":
                        target_len = 76
                    elif args.dataset_set == "dis_seq3":
                        target_len = 76
                    elif args.dataset_set in ["scl_seq1", "scl_seq2", "scl_seq3"]:
                        target_len = 5
                    elif args.dataset_set in ['long_seq1', 'long_seq2', 'long_seq3']:
                        target_len = 6
                    elif args.dataset_set in ['decanlp1', 'decanlp2', 'decanlp3']:
                        target_len = 146
                from dataset.t5_dataset import T5Dataset
                validDataset = T5Dataset(args, tokenizer, task_name, idx, args.max_seq_len, "valid",
                                         target_len=target_len)
                testDataset = T5Dataset(args, tokenizer, task_name, idx, args.max_seq_len, "test",
                                        target_len=target_len)
                label_maps[task_name] = validDataset.label_map
                label2task_id = validDataset.label2task_id
                validDatasets[task_name] = CustomDatasetDataLoader(args, validDataset, args.per_gpu_eval_batch_size,
                                                                   args.pri_task,
                                                                   task_name, shuffle=False)
                testDatasets[task_name] = CustomDatasetDataLoader(args, testDataset, args.per_gpu_eval_batch_size,
                                                                  args.pri_task,
                                                                  task_name, shuffle=False)
            else:
                raise ValueError("No such task %s !!!" % task_name)

        return validDatasets, testDatasets, label_maps, label2task_id

    # Dividing a portion of samples from the training set as validation set and test set.
    if split_train_set_to_valid_and_test:
        trainDatasets = {}
        validDatasets = {}
        testDatasets = {}
        for idx, task_name in enumerate(task_names):
            if task_name in T5_TASK_LIST:
                target_len = task_to_target_len[task_name]
                if args.algo in ["t5er", "t5derpp"]:
                    if args.dataset_set == "dis_seq1":
                        target_len = 6
                    elif args.dataset_set == "dis_seq2":
                        target_len = 76
                    elif args.dataset_set == "dis_seq3":
                        target_len = 76
                    elif args.dataset_set in ["scl_seq1", "scl_seq2", "scl_seq3"]:
                        target_len = 5
                    elif args.dataset_set in ['long_seq1', 'long_seq2', 'long_seq3']:
                        target_len = 6
                    elif args.dataset_set in ['decanlp1', 'decanlp2', 'decanlp3']:
                        target_len = 146
                from dataset.t5_dataset import T5Dataset
                Dataset = T5Dataset(args, tokenizer, task_name, idx, args.max_seq_len, train_type, lazy=False,
                                    split=True, target_len=target_len)
                validDataset = T5Dataset(args, tokenizer, task_name, idx, args.max_seq_len, 'valid',
                                         examples=Dataset.dev_examples, target_len=target_len)
                testDataset = T5Dataset(args, tokenizer, task_name, idx, args.max_seq_len, 'test',
                                        examples=Dataset.test_examples, target_len=target_len)
                label_maps[task_name] = validDataset.label_map
                label2task_id = validDataset.label2task_id
                trainDatasets[task_name] = CustomDatasetDataLoader(args, Dataset, args.per_gpu_train_batch_size,
                                                                   args.pri_task,
                                                                   task_name, shuffle=True)
                validDatasets[task_name] = CustomDatasetDataLoader(args, validDataset, args.per_gpu_eval_batch_size,
                                                                   args.pri_task,
                                                                   task_name, shuffle=False)
                testDatasets[task_name] = CustomDatasetDataLoader(args, testDataset, args.per_gpu_eval_batch_size,
                                                                  args.pri_task,
                                                                  task_name, shuffle=False)
            else:
                raise ValueError("No such task %s !!!" % task_name)
        return trainDatasets, testDatasets, validDatasets, label_maps, label2task_id

    # Dividing a portion of samples from the training set as validation set.
    if split_train_set_to_valid:
        trainDatasets = {}
        validDatasets = {}
        for idx, task_name in enumerate(task_names):
            if task_name in T5_TASK_LIST:
                target_len = task_to_target_len[task_name]
                if args.algo in ["t5er", "t5derpp"]:
                    if args.dataset_set == "dis_seq1":
                        target_len = 6
                    elif args.dataset_set == "dis_seq2":
                        target_len = 76
                    elif args.dataset_set == "dis_seq3":
                        target_len = 76
                    elif args.dataset_set in ["scl_seq1", "scl_seq2", "scl_seq3"]:
                        target_len = 5
                    elif args.dataset_set in ['long_seq1', 'long_seq2', 'long_seq3']:
                        target_len = 6
                    elif args.dataset_set in ['decanlp1', 'decanlp2', 'decanlp3']:
                        target_len = 146
                from dataset.t5_dataset import T5Dataset
                Dataset = T5Dataset(args, tokenizer, task_name, idx, args.max_seq_len, train_type, lazy=False,
                                    split=True, target_len=target_len)
                validDataset = T5Dataset(args, tokenizer, task_name, idx, args.max_seq_len, 'valid',
                                         examples=Dataset.dev_examples, target_len=target_len)
                label_maps[task_name] = validDataset.label_map
                label2task_id = validDataset.label2task_id
                trainDatasets[task_name] = CustomDatasetDataLoader(args, Dataset, args.per_gpu_train_batch_size,
                                                                   args.pri_task,
                                                                   task_name, shuffle=True)
                validDatasets[task_name] = CustomDatasetDataLoader(args, validDataset, args.per_gpu_eval_batch_size,
                                                                   args.pri_task,
                                                                   task_name, shuffle=False)
            else:
                raise ValueError("No such task %s !!!" % task_name)
        return trainDatasets, validDatasets, label_maps, label2task_id

    for idx, task_name in enumerate(task_names):
        # print(task_name)
        if train_type == "train":
            label_map = {}
        else:
            label_map = label_maps[task_name]

        if task_name in T5_TASK_LIST:
            from dataset.t5_dataset import T5Dataset
            target_len = task_to_target_len[task_name]
            if args.algo in ["t5er", "t5derpp"]:
                if args.dataset_set in ["scl_seq1", "scl_seq2", "scl_seq3"]:
                    target_len = 5
                elif args.dataset_set in ['long_seq1', 'long_seq2', 'long_seq3']:
                    target_len = 6
            dataset = T5Dataset(args, tokenizer, task_name, idx, args.max_seq_len, train_type, lazy=False,
                                target_len=target_len)
        else:
            raise ValueError("No such task %s !!!" % task_name)

        num_labels.append(len(dataset.id2label))
        if train_type == "train":
            datasets[task_name] = CustomDatasetDataLoader(args, dataset, args.per_gpu_train_batch_size, args.pri_task,
                                                          task_name)
        else:
            datasets[task_name] = CustomDatasetDataLoader(args, dataset, args.per_gpu_eval_batch_size, args.pri_task,
                                                          task_name, shuffle=False)
        label_maps[task_name] = dataset.label_map

        label2task_id = dataset.label2task_id

    return datasets, num_labels, label_maps, label2task_id


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, args, dataset, bsz, task_type=None, task_name=None, shuffle=True):
        """Initialize this class
        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.args = args
        self.bsz = bsz
        self.shuffle = shuffle
        self.dataset_len = len(dataset)
        self.task_info = {"task_type": task_type, "task_name": task_name}
        self.dataset = dataset
        if task_name in T5_TSV_TASK_NAME_LIST or task_name in T5_CSV_TASK_NAME_LIST or task_name in T5_JSON_TASK_NAME_LIST or task_name in T5_JSONL_TASK_NAME_LIST:
            self.task_info["target_len"] = dataset.target_len

    def load_data(self, sampler=None):
        if sampler:
            if self.args.distributed:
                self.dataloader = torch.utils.data.DataLoader(
                    self.dataset,
                    batch_size=self.bsz,
                    sampler=sampler,
                    pin_memory=True,
                    shuffle=False,
                )
            else:
                self.dataloader = torch.utils.data.DataLoader(
                    self.dataset,
                    batch_size=self.bsz,
                    sampler=sampler,
                    pin_memory=True,
                )
        else:
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.bsz,
                shuffle=self.shuffle)
        return self.dataloader

    def __len__(self):
        """Return the number of data in the dataset"""
        return self.dataset_len
