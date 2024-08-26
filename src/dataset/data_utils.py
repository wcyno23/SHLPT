import json
import datasets
import jsonlines
import torch
import pandas as pd
from transformers import PreTrainedTokenizer
from dataset.config import *


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid=None, text_a=None, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids=None,
                 input_mask=None,
                 segment_ids=None,

                 tokens_term_ids=None,
                 tokens_sentence_ids=None,

                 label_id=None,

                 masked_lm_labels=None,
                 masked_pos=None,
                 masked_weights=None,

                 position_ids=None,

                 valid_ids=None,
                 label_mask=None,

                 dataset_id=None,
                 task_id=None,

                 cq_ids=None,
                 cq_len=None,
                 y_ids=None,
                 gen_x_ids=None,
                 gen_x_mask=None,
                 gen_x_segment=None,
                 gen_y_ids=None,
                 id=None,
                 ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

        self.label_id = label_id

        self.masked_lm_labels = masked_lm_labels,
        self.masked_pos = masked_pos,
        self.masked_weights = masked_weights

        self.tokens_term_ids = tokens_term_ids
        self.tokens_sentence_ids = tokens_sentence_ids

        self.position_ids = position_ids

        self.valid_ids = valid_ids
        self.label_mask = label_mask

        self.dataset_id = dataset_id
        self.task_id = task_id

        self.cq_ids = cq_ids
        self.cq_len = cq_len
        self.y_ids = y_ids
        self.gen_x_ids = gen_x_ids
        self.gen_x_mask = gen_x_mask
        self.gen_x_segment = gen_x_segment
        self.gen_y_ids = gen_y_ids
        self.id = id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file):
        """Reads a json file for tasks in sentiment analysis."""
        if input_file.endswith(".json"):
            with open(input_file) as f:
                return json.load(f)
        elif input_file.endswith(".jsonl"):
            lines = []
            with open(input_file) as f:
                for item in jsonlines.Reader(f):
                    lines.append(item)
                return lines

    @classmethod
    def _read_jsonl(cls, input_file, task_name=None):
        """Reads a jsonl file."""
        if input_file.endswith(".jsonl"):
            from dataset.t5_dataset import T5Example
            examples = []
            lines = []
            with open(input_file) as f:
                for item in jsonlines.Reader(f):
                    lines.append(item)
            labels = task_to_labels[task_name]
            keys = task_to_keys[task_name]
            i = 0
            # TODO: update for idx when use multirc
            idx = []
            for data in lines:
                context = ""
                target = ""
                if task_name == "multirc":
                    paragraph = data["passage"]
                    for question in paragraph["questions"]:
                        for answer in question["answers"]:
                            label = answer.get("label")
                            item = {"paragraph": paragraph["text"],
                                    "question": question["question"],
                                    "answer": answer["text"], }
                            for key in keys:
                                context += key + ": " + str(item[key]) + ' '
                            examples.append(
                                T5Example(context=context, target=labels[label]))
                            idx.append(i)
                        i += 1
                elif task_name == "qqp":
                    context = "question1: " + data["text1"] + " " + "question2: " + data["text2"] + " "
                    examples.append(T5Example(context=context, target=labels[data["label"]]))
                else:
                    if keys[1] != None:
                        for key in keys:
                            context += key + ": " + str(data[key]) + ' '
                    else:
                        key = keys[0]
                        context += key + ": " + str(data[key]) + ' '
                    # not need to convert id to label
                    if task_name in ['cb']:
                        target = data['label']
                    elif task_name in ['boolq']:
                        target = labels[1] if data['label'] else labels[0]
                    elif task_name in ['nqopen']:
                        target = ", ".join(data['answer'])
                    else:
                        target = labels[data['label']]
                    examples.append(T5Example(context=context, target=target))
            return examples

    @classmethod
    def _read_json_t5(cls, input_file, task_name):
        """"Reads a json file"""
        if input_file.endswith(".json"):
            from dataset.t5_dataset import T5Example
            examples = []
            raw_datas = None
            with open(input_file) as f:
                raw_datas = json.load(f)
            if task_name == "squad" or task_name == "srl" or task_name == "woz":
                raw_datas = map(lambda x: x["paragraphs"], raw_datas["data"])
                datas = []
                for raw_data in raw_datas:
                    datas.extend(raw_data)
                for data in datas:
                    context = "context: " + data['context']
                    for qa in data['qas']:
                        question = "question: " + qa['question'] + ' '
                        answer = []
                        raw_answers = qa['answers']
                        if len(raw_answers) == 0:
                            assert qa["is_impossible"]
                            raw_answers.append({"text": ""})
                        for raw_answer in raw_answers:
                            answer.append(raw_answer['text'])
                        examples.append(
                            T5Example(context=question + context, target=answer[0]))
            else:
                for data in raw_datas:
                    context = "sentence: " + data['sentence'] + ' '
                    target = 'positive' if data['label'] == "Good" else 'negative'
                    examples.append(T5Example(context=context, target=target))
            return examples

    @classmethod
    def _read_txt(cls, input_file):
        """Reads a txt file"""
        if input_file.endswith(".txt"):
            lines = []
            with open(input_file, 'r') as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    lines.append(line)
            return lines

    @classmethod
    def _read_csv(cls, input_file, task_name=None):
        """Reads a csv file"""
        if input_file.endswith(".csv"):
            if task_name in T5_CSV_TASK_NAME_LIST:
                from dataset.t5_dataset import T5Example
                examples = []

                df = pd.read_csv(input_file, header=None)
                if task_name == 'yahoo_answers_topics':
                    df = df.rename(columns={0: "label", 1: "question_title", 2: "question_content", 3: "best_answer"})
                elif task_name == 'yelp':
                    df = df.rename(columns={0: "label", 1: "content"})
                else:
                    df = df.rename(columns={0: "label", 1: "title", 2: "content"})
                df['label'] = df['label'] - 1
                dataset = datasets.Dataset.from_pandas(df)
                labels = task_to_labels[task_name]
                keys = task_to_keys[task_name]
                for data in dataset:
                    context = ""
                    if keys[1] != None:
                        for key in keys:
                            context += key + ": " + str(data[key]) + ' '
                    else:
                        key = keys[0]
                        context += key + ": " + str(data[key]) + ' '
                    examples.append(
                        T5Example(context=context, target=labels[data['label']]))
                return examples

    @classmethod
    def _read_tsv(cls, input_file, task_name=None):
        """Reads a tsv file"""
        if input_file.endswith(".tsv") and task_name in T5_TSV_TASK_NAME_LIST:
            from dataset.t5_dataset import T5Example
            examples = []
            df = pd.read_table(input_file, on_bad_lines='skip')
            if task_name == 'rte':
                # df = df.rename(columns={0:})
                pass
            else:
                df = df.rename(columns={0: "content", 1: "label"})
            dataset = datasets.Dataset.from_pandas(df)
            labels = task_to_labels[task_name]
            keys = task_to_keys[task_name]
            for data in dataset:
                context = ""
                if task_name == "mnli":
                    if data['sentence1'] != None and data['sentence2'] != None:
                        context = "premise: " + data['sentence1'] + ' ' + "hypothesis: " + data['sentence2'] + ' '
                    else:
                        continue
                elif keys[1] != None:
                    for key in keys:
                        context += key + ": " + str(data[key]) + ' '
                else:
                    key = keys[0]
                    context += key + ": " + str(data[key]) + ' '
                if task_name == 'rte':
                    # already converted to labels
                    target = data['label']
                elif task_name == 'mnli':
                    target = data['gold_label']
                    if target is None:
                        continue
                else:
                    target = labels[data['label']]
                examples.append(
                        T5Example(context=context, target=target))
            return examples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(examples,
                                 label_list,
                                 max_seq_length,
                                 tokenizer: PreTrainedTokenizer,
                                 mode,
                                 label_map={},
                                 dataset_id=-1,
                                 label2task_id=None,
                                 class_per_task=-1,
                                 task_name=None):
    """Loads a data file into a list of `InputBatch`s."""
    if label2task_id is not None:
        if dataset_id not in label2task_id.keys():
            label2task_id[dataset_id] = {}

    if mode == 'asc':  # for pair
        label_map = {'+': 0, 'positive': 0, '-': 1, 'negative': 1, 'neutral': 2}
    elif mode == 'nli':
        label_map = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
    elif mode == 'ae':
        label_map = {'B': 0, 'I': 1, 'O': 2}
    features = []

    id2label = {}
    for label in label_map.keys():
        id2label[label_map[label]] = label
    class_list = [[] for _ in range(len(label_map))]

    for (ex_index, example) in enumerate(examples):
        if example.text_b:
            tokenized = tokenizer.encode_plus((example.text_a, example.text_b), truncation=True,
                                              add_special_tokens=True, padding=True, return_token_type_ids=True,
                                              max_length=max_seq_length)
        else:
            if mode == 'ae':
                tokenized = tokenizer.encode_plus(example.text_a, truncation=True, add_special_tokens=True,
                                                  padding=True, return_token_type_ids=True, is_split_into_words=True,
                                                  max_length=max_seq_length)
            else:
                tokenized = tokenizer.encode_plus(example.text_a, truncation=True, add_special_tokens=True,
                                                  padding=True, return_token_type_ids=True, max_length=max_seq_length)

        input_ids = tokenized["input_ids"]
        input_mask = tokenized["attention_mask"]
        segment_ids = tokenized["token_type_ids"]

        while len(input_ids) < max_seq_length:
            input_ids.append(tokenizer.pad_token_id)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if mode in ["asc", "nli"]:
            label_id = label_map[example.label]
            # id2label[label_id] = example.label
        elif mode in ["ae"]:
            label_id = [-1] * len(input_ids)  # -1 is the index to ignore
            # truncate the label length if it exceeds the limit.
            lb = []
            for label in example.label:
                lb.append(label_map[label])
                # id2label[label_map[label]] = label
            if len(lb) > max_seq_length - 2:
                lb = lb[0:(max_seq_length - 2)]
            label_id[1:len(lb) + 1] = lb
        else:
            if example.label not in label_map.keys():
                label_map[example.label] = len(label_map)
                id2label[label_map[example.label]] = example.label
                class_list.append([])
                label2task_id["num"] += 1
                if label2task_id["split_type"] == "class":
                    label2task_id[dataset_id][label_map[example.label]] = label2task_id["num"] // class_per_task
                else:
                    label2task_id[dataset_id][label_map[example.label]] = dataset_id
            label_id = label_map[example.label]
            class_list[label_id].append(ex_index)
        task_id = label2task_id[dataset_id][label_id]

        # if ex_index<2:
        #     print(example.text_a, example.text_b, example.label)
        #     print(input_ids, input_mask, segment_ids, label_id)

        features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_id=label_id,
                    dataset_id=dataset_id,
                    task_id=task_id))

    return features, id2label, label_map, class_list, label2task_id
