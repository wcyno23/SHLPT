import os
import logging
import numpy as np
import torch
import random
import json
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from dataset.config import *
from dataset import data_utils

logger = logging.getLogger(__name__)


class Features(object):
    """A set of features of data specifically for T5Dataset"""

    def __init__(self, c_ids, c_mask, t_ids, t_mask, dataset_id=None,
                 task_id=None, ex_index=None):
        # context
        self.c_ids = c_ids
        self.c_mask = c_mask
        # target
        self.t_ids = t_ids
        self.t_mask = t_mask

        self.dataset_id = dataset_id
        self.task_id = task_id
        self.ex_index = ex_index  # example id


class T5Example(object):
    """A single training/test example for T5."""

    def __init__(self, context, target, question=None):
        """construct a T5 example"""
        self.context = context
        self.target = target
        self.question = question


class T5Processor(data_utils.DataProcessor):
    """Processor for T5 Dataset."""

    def get_train_examples(self, data_dir, max_num=-1, fn="train.txt", task_name=None):
        if fn.endswith(".txt"):
            return self._create_examples(
                self._read_txt(os.path.join(data_dir, fn)), max_num)
        elif fn.endswith(".csv"):
            return self._read_csv(os.path.join(data_dir, fn), task_name)
        elif fn.endswith(".jsonl"):
            return self._read_jsonl(os.path.join(data_dir, fn), task_name)
        elif fn.endswith(".tsv"):
            return self._read_tsv(os.path.join(data_dir, fn), task_name)
        elif fn.endswith(".json"):
            return self._read_json_t5(os.path.join(data_dir, fn), task_name)

    def get_dev_examples(self, data_dir, max_num=-1, fn="valid.txt", task_name=None):
        if fn.endswith(".txt"):
            return self._create_examples(
                self._read_txt(os.path.join(data_dir, fn)), max_num)
        elif fn.endswith(".csv"):
            return self._read_csv(os.path.join(data_dir, fn), task_name)
        elif fn.endswith(".jsonl"):
            return self._read_jsonl(os.path.join(data_dir, fn), task_name)
        elif fn.endswith(".tsv"):
            return self._read_tsv(os.path.join(data_dir, fn), task_name)
        elif fn.endswith(".json"):
            return self._read_json_t5(os.path.join(data_dir, fn), task_name)

    def get_test_examples(self, data_dir, max_num=-1, fn="test.txt", task_name=None):
        if fn.endswith(".txt"):
            return self._create_examples(
                self._read_txt(os.path.join(data_dir, fn)), max_num)
        elif fn.endswith(".csv"):
            return self._read_csv(os.path.join(data_dir, fn), task_name)
        elif fn.endswith(".jsonl"):
            return self._read_jsonl(os.path.join(data_dir, fn), task_name)
        elif fn.endswith(".tsv"):
            return self._read_tsv(os.path.join(data_dir, fn), task_name)
        elif fn.endswith(".json"):
            return self._read_json_t5(os.path.join(data_dir, fn), task_name)

    def _create_examples(self, raw_datas, max_num=-1):
        examples = []

        for i, data in enumerate(raw_datas):
            data = data.strip()
            data_list = data.split("\t")
            examples.append(T5Example(context=data_list[0], target=data_list[1]))
        return examples


class T5Dataset(Dataset):
    """
        Dataset to load T5 dataset
    """

    def __init__(self, args,
                 tokenizer: PreTrainedTokenizer,
                 task_name, dataset_id,
                 max_seq_len, train_type,
                 lazy=False, extra_data=None, split=False, examples=None, target_len=5):
        self.tokenizer = tokenizer
        self.args = args
        self.target_len = target_len
        dir_path = os.path.join(args.data_cache_dir, DATA_DIR_MAP[task_name])

        if task_name in T5_CSV_TASK_NAME_LIST:
            fn = train_type + ".csv"
        elif task_name in T5_TSV_TASK_NAME_LIST:
            fn = train_type + ".tsv"
        elif task_name in T5_JSONL_TASK_NAME_LIST:
            fn = train_type + ".jsonl"
        elif task_name in T5_JSON_TASK_NAME_LIST:
            fn = train_type + ".json"
        else:
            fn = train_type + ".txt"

        if examples is not None:
            train_examples = examples
        else:
            logger.info("Creating features from %s dataset file at %s", train_type, dir_path)
            processor = T5Processor()
            if train_type == "train":
                train_examples = processor.get_train_examples(dir_path, args.max_datalen, fn=fn, task_name=task_name)
            elif train_type == "valid":
                train_examples = processor.get_dev_examples(dir_path, args.test_max_datalen, fn=fn, task_name=task_name)
            elif train_type == "test":
                train_examples = processor.get_test_examples(dir_path, args.test_max_datalen, fn=fn,
                                                             task_name=task_name)
            elif train_type == "extra":
                train_examples = self.get_examples_from_extra_data(extra_data, task_name)
            else:
                raise ValueError("train_type un fitted!")
            # For yahoo dataset we need to filter out empty rows
            # (i.e. where "question" field is empty)
            if task_name == "yahoo_answers_topics":
                train_examples = np.array(train_examples)
                logger.info("Filter out empty rows in yahoo dataset")
                if train_type == "train":
                    good_id = np.load('../dataset/good_id_yahoo/good_id_yahoo_train2.npy')
                else:
                    good_id = np.load('../dataset/good_id_yahoo/good_id_yahoo_test2.npy')
                train_examples = train_examples[good_id]

            if split:
                if args.k_val != -1:
                    k_val = args.k_val
                else:
                    k_val = args.k_train
                # Dividing a portion of samples from the training set as validation set and test set.
                if args.split_train_file_to_valid_and_test:
                    train_examples = self.select_subset_dataset(train_examples, args.k_train + k_val, task_name)
                    train_examples, val_and_test_examples = self.split_dataset(train_examples, args.k_train,
                                                                               args.first_k_split, task_name)
                    self.dev_examples, self.test_examples = self.split_dataset(val_and_test_examples, k_val,
                                                                               args.first_k_split, task_name)
                # Dividing a portion of samples from the training set as validation set.
                elif args.split_train_file_to_valid:
                    train_examples = self.select_subset_dataset(train_examples, args.k_train + k_val, task_name)
                    train_examples, self.dev_examples = self.split_dataset(train_examples, args.k_train,
                                                                           args.first_k_split, task_name)
                # Dividing a portion of samples from the test set as validation set.
                elif args.split_test_file:
                    if args.first_k_split:
                        train_examples = self.select_subset_dataset(train_examples, args.k_test + k_val, task_name)
                    else:
                        train_examples = self.select_subset_dataset(train_examples, args.k_test * 2, task_name)
                    self.dev_examples, self.test_examples = self.split_dataset(train_examples, k_val,
                                                                               args.first_k_split, task_name)
                else:
                    raise ValueError("split way not fitted!")
            else:
                if train_type == "train":
                    train_examples = self.select_subset_dataset(train_examples, args.k_train, task_name)
                else:
                    # small dataset not need to select subset
                    if task_name in ['cb']:
                        pass
                    else:
                        train_examples = self.select_subset_dataset(train_examples, args.k_test, task_name)


        self.train_features = self.convert_examples_to_feature(train_examples, max_seq_len, dataset_id)
        # The following variants were not used
        self.id2label = {}
        self.label_map = {}
        self.label2task_id = {}

    def get_examples_from_extra_data(self, extra_data, task_name):
        examples = []
        for data in extra_data:
            data = data.strip()
            data = data.split("__eos__")[0]
            data_list = data.split("__ans__")
            target = data_list[1].strip(' ')
            examples.append(T5Example(context=data_list[0], target=target))

        return examples

    def convert_examples_to_feature(self, examples, max_seq_length, dataset_id):
        features = []
        max = 0
        for ex_index, example in enumerate(examples):
            context = example.context
            target = example.target

            tokenized_c = self.tokenizer.encode_plus(context, truncation=True, add_special_tokens=True,
                                                     padding=True, return_token_type_ids=True,
                                                     max_length=max_seq_length)
            tokenized_t = self.tokenizer.encode_plus(target, truncation=True, add_special_tokens=True,
                                                     padding=True, return_token_type_ids=True,
                                                     max_length=max_seq_length)

            c_ids = tokenized_c["input_ids"]
            c_mask = tokenized_c["attention_mask"]
            t_ids = tokenized_t["input_ids"]
            t_mask = tokenized_t["attention_mask"]
            if max < len(c_ids):
                max = len(c_ids)
                # print(max)
            while len(c_ids) < max_seq_length:
                c_ids.append(self.tokenizer.pad_token_id)
                c_mask.append(0)
            while len(t_ids) < self.target_len:
                t_ids.append(self.tokenizer.pad_token_id)
                t_mask.append(0)

            assert len(c_ids) == max_seq_length
            assert len(c_mask) == max_seq_length
            assert len(t_ids) == self.target_len
            assert len(t_mask) == self.target_len


            features.append(
                Features(
                    c_ids=c_ids,
                    c_mask=c_mask,
                    t_ids=t_ids,
                    t_mask=t_mask,
                    dataset_id=dataset_id,
                    ex_index=ex_index
                )
            )
        return features

    def __len__(self):
        return len(self.train_features)

    def __getitem__(self, item):
        c_ids = torch.tensor(self.train_features[item].c_ids, dtype=torch.long)
        c_mask = torch.tensor(self.train_features[item].c_mask, dtype=torch.long)
        t_ids = torch.tensor(self.train_features[item].t_ids, dtype=torch.long)
        t_mask = torch.tensor(self.train_features[item].t_mask, dtype=torch.long)
        dataset_id = torch.tensor(self.train_features[item].dataset_id, dtype=torch.long)
        if self.args.add_hsc_loss or self.args.add_asc_loss_unfiltered or self.args.add_asc_loss_filtered:
            ex_index = self.train_features[item].ex_index
            return c_ids, c_mask, t_ids, t_mask, dataset_id, ex_index
        return c_ids, c_mask, t_ids, t_mask, dataset_id

    def select_subset_dataset(self, train_examples, k=-1, task_name=None):
        # select a subset of k samples per class for classification dataset while 4k samples for QA dataset
        if k == -1:
            return train_examples
        if task_name in QA_TASK_NAME_LIST:
            return random.sample(train_examples, k * 4)

        idx_total = np.array([], dtype='int64')
        label2idx = {}
        train_examples = np.array(train_examples)
        for idx, example in enumerate(train_examples):
            if example.target not in label2idx.keys():
                label2idx[example.target] = [idx]
            else:
                label2idx[example.target].append(idx)
        for label in label2idx.keys():
            idx_per_class = label2idx[label]
            idx_total = np.concatenate(
                [idx_total, np.random.choice(idx_per_class, min(k, len(idx_per_class)), replace=False)])
        np.random.shuffle(idx_total)
        return train_examples[idx_total]


    # split dataset into two subset
    def split_dataset(self, train_examples, k=-1, first_k_split=False, task_name=None):
        # QA dataset
        if task_name in QA_TASK_NAME_LIST:
            if k != -1:
                return train_examples[:k * 4], train_examples[k * 4:]
            else:
                half_size = len(train_examples) // 2
                return train_examples[:half_size], train_examples[half_size:]

        # Classification dataset
        idx_total_dev = np.array([], dtype='int64')
        idx_total_test = np.array([], dtype='int64')
        label2idx = {}
        train_examples = np.array(train_examples)
        for idx, example in enumerate(train_examples):
            if example.target not in label2idx.keys():
                label2idx[example.target] = [idx]
            else:
                label2idx[example.target].append(idx)
        for label in label2idx.keys():
            idx_per_class = np.array(label2idx[label])
            num = idx_per_class.shape[0]
            np.random.shuffle(idx_per_class)
            if k != -1 and first_k_split and num >= k:
                idx_total_dev = np.concatenate([idx_total_dev, idx_per_class[np.arange(0, k)]])
                idx_total_test = np.concatenate([idx_total_test, idx_per_class[np.arange(k, num)]])
            else:
                idx_total_dev = np.concatenate([idx_total_dev, idx_per_class[np.arange(0, num // 2)]])
                idx_total_test = np.concatenate([idx_total_test, idx_per_class[np.arange(num // 2, num)]])
        np.random.shuffle(idx_total_dev)
        np.random.shuffle(idx_total_test)
        return train_examples[idx_total_dev], train_examples[idx_total_test]

