import torch.utils.data.sampler as sampler
import torch.utils.data.distributed as distributed
from torch.utils.data import Sampler
import random
import numpy as np
import math

random.seed(10)


class SubsetSequentialSampler(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, indices) -> None:
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)




def generate_prob_matrix(args, datalen, train_tasks, task_split):
    task_boundary = args.task_boundary
    major_num = args.major_num
    major_step = args.major_step if args.major_step != -1 else major_num
    if major_step > major_num:
        raise ValueError(
            "the config major step should not larger than major num, which will result in some samples untrainining")
    # major_prob = args.major_prob
    cover_width = args.cover_width

    # find the unlabel task in task sequnece and map the label tasks to the unlabel tasks
    unlabel_pos = []
    label2corpus = {}
    part_id = 0
    for i in range(len(train_tasks)):
        if train_tasks[i][0] == "a":
            unlabel_pos.append(i)
    if len(unlabel_pos) != 0:
        for i in range(len(train_tasks)):
            if i == int(task_split[part_id]):
                part_id += 1
            if train_tasks[i][0] != "a":
                label2corpus[i] = unlabel_pos[part_id - 1]
    print(len(unlabel_pos))

    # set the prob matrix for labeled data 
    prob = []
    datalen -= len(unlabel_pos)
    for task_id in range(math.ceil(datalen / major_step)):
        prob.append(np.zeros(datalen))

        # random draw the prob for major samples
        if task_boundary == 0:
            major_prob = 1
            right_num = 0
            left_num = 0
        else:
            major_prob = np.clip(np.random.normal(args.major_prob, 0.07), args.major_prob - 0.1, args.major_prob + 0.1)

            # draw the available neighbour samples
            right_num = max(min(cover_width, datalen - (task_id * major_step + major_num)), 0)
            left_num = min(cover_width, task_id * major_step)

        # random draw the prob for non-major samples
        non_major_num = left_num + right_num
        non_major_prob = np.random.randint(0, 10, non_major_num)
        non_major_prob = [(1 - major_prob) * i / sum(non_major_prob) for i in non_major_prob]

        # set prob to the matrix
        for i in range(left_num):
            pos = task_id * major_step - i - 1
            prob[task_id][pos] = non_major_prob[left_num - 1 - i]
        for i in range(right_num):
            pos = task_id * major_step + major_num + i
            prob[task_id][pos] = non_major_prob[left_num + i]
        for i in range(major_num):
            pos = task_id * major_step + i
            if pos == datalen - 1 or i == major_num - 1:
                prob[task_id][pos] = 1 - sum(prob[task_id])
                break
            else:
                prob[task_id][pos] = major_prob / major_num

    # set the prob matrix for unlabeled data
    if len(unlabel_pos) != 0:
        probb = []
        datalen += len(unlabel_pos)
        if args.corpus_involved:
            # unlabeled corpus train with label data in the same task
            for task_id in range(len(prob)):
                probb.append(np.zeros(datalen))
                for i in label2corpus.keys():
                    offset = sum(np.array(unlabel_pos) < i)
                    probb[-1][i] = prob[task_id][i - offset]
                    probb[-1][label2corpus[i]] += prob[task_id][i - offset]
        else:
            # unlabeled corpus train as an external task
            last_corpus = []
            for task_id in range(len(prob)):
                probb.append(np.zeros(datalen))
                for i in label2corpus.keys():
                    offset = sum(np.array(unlabel_pos) < i)
                    probb[-1][i] = prob[task_id][i - offset]
                    if prob[task_id][i - offset] != 0 and label2corpus[i] not in last_corpus:
                        last_corpus.append(label2corpus[i])
                        probb.insert(-1, np.zeros(datalen))
                        probb[-2][label2corpus[i]] = 1
    else:
        probb = prob

    probb = np.array(probb)
    return probb


def sample_task_in_domain_level(args, datalen_list, valid_datalen_list, test_datalen_list):
    task_list = {'train': [], 'valid': [], 'test': []}

    task_list['train'].append([0, {}])
    for i, datalen in enumerate(datalen_list):
        task_list['train'][0][0] += math.ceil(datalen / (args.per_gpu_train_batch_size))
        task_list['train'][0][1][str(i)] = sampler.RandomSampler(list(range(datalen)))

    task_list['valid'].append([0, {}])
    for i, datalen in enumerate(valid_datalen_list):
        if datalen == 0:
            continue
        task_list['valid'][0][0] += math.ceil(datalen / (args.per_gpu_eval_batch_size))
        task_list['valid'][0][1][str(i)] = sampler.SequentialSampler(list(range(datalen)))

    task_list['test'].append([0, {}])
    for i, datalen in enumerate(test_datalen_list):
        if datalen == 0:
            continue
        task_list['test'][0][0] += math.ceil(datalen / (args.per_gpu_eval_batch_size))
        task_list['test'][0][1][str(i)] = sampler.SequentialSampler(list(range(datalen)))
    return task_list


def sample_task_in_dataset_level(args, datalen_list, valid_datalen_list, test_datalen_list):
    task_list = {'train': [], 'valid': [], 'test': []}

    prob = generate_prob_matrix(args, len(datalen_list), args.task_order, args.task_split)
    print(args.task_order)
    print(prob)

    for i in range(len(prob)):
        task_list['train'].append([0, {}])
        task_list['valid'].append([0, {}])
        task_list['test'].append([0, {}])

    for i, datalen in enumerate(datalen_list):
        indices = list(range(datalen))
        random.shuffle(indices)

        valid_datalen = valid_datalen_list[i]
        valid_indices = list(range(valid_datalen))

        test_datalen = test_datalen_list[i]
        test_indices = list(range(test_datalen))

        for j in range(len(prob)):
            left = int(np.sum(prob[:j, i]) / np.sum(prob[:, i]) * datalen)
            right = int(np.sum(prob[:j + 1, i]) / np.sum(prob[:, i]) * datalen)

            valid_left = int(np.sum(prob[:j, i]) / np.sum(prob[:, i]) * valid_datalen)
            valid_right = int(np.sum(prob[:j + 1, i]) / np.sum(prob[:, i]) * valid_datalen)

            test_left = int(np.sum(prob[:j, i]) / np.sum(prob[:, i]) * test_datalen)
            test_right = int(np.sum(prob[:j + 1, i]) / np.sum(prob[:, i]) * test_datalen)
            if math.ceil((right - left) / (args.per_gpu_train_batch_size)) != 0:
                task_list['train'][j][0] += math.ceil((right - left) / (args.per_gpu_train_batch_size))
                task_list['train'][j][1][str(i)] = sampler.SubsetRandomSampler(indices[left:right])

            if math.ceil((valid_right - valid_left) / (args.per_gpu_eval_batch_size)) != 0:
                task_list['valid'][j][0] += math.ceil((valid_right - valid_left) / (args.per_gpu_eval_batch_size))
                task_list['valid'][j][1][str(i)] = SubsetSequentialSampler(valid_indices[valid_left:valid_right])

            if math.ceil((test_right - test_left) / (args.per_gpu_eval_batch_size)) != 0:
                task_list['test'][j][0] += math.ceil((test_right - test_left) / (args.per_gpu_eval_batch_size))
                task_list['test'][j][1][str(i)] = SubsetSequentialSampler(test_indices[test_left:test_right])
    return task_list


def sample_task_in_class_level(args, dataset_list, valid_datalen_list, test_datalen_list):
    task_list = {'train': [], 'valid': [], 'test': []}

    class_len = 0
    # split the dataset to multi datasets by the label class
    # extend the train_order and task_split to the multi dataset version
    m_train_order, m_task_split = [], []
    for k, classlen_list in enumerate(dataset_list):
        if int(args.task_split[len(m_task_split)]) == k:
            m_task_split.append(class_len)
        class_len += len(classlen_list)
        m_train_order.extend([args.task_order[k] for i in range(len(classlen_list))])
    m_task_split.append(-1)
    print(m_train_order)
    print(m_task_split)

    prob = generate_prob_matrix(args, class_len, m_train_order, m_task_split)
    print(prob)
    for i in range(len(prob)):
        task_list['train'].append([0, {}])
        task_list['valid'].append([0, {}])
        task_list['test'].append([0, {}])
    index = 0
    for k, classlen_list in enumerate(dataset_list):
        valid_classlen_list = valid_datalen_list[k]
        test_classlen_list = test_datalen_list[k]
        if len(classlen_list) == 1:
            # unlabeled data
            datalen = classlen_list[0]
            indices = list(range(datalen))
            random.shuffle(indices)
            for j in range(len(prob)):
                left = int(np.sum(prob[:j, index]) / np.sum(prob[:, index]) * datalen)
                right = int(np.sum(prob[:j + 1, index]) / np.sum(prob[:, index]) * datalen)
                # print(i, datalen, left, right)
                part_ind = indices[left:right]
                if math.ceil((right - left) / (args.per_gpu_train_batch_size)) != 0:
                    task_list['train'][j][0] += (right - left)
                    task_list['train'][j][1][str(k)] = part_ind
        else:
            # labeled data
            for i, class_indices in enumerate(classlen_list):
                random.shuffle(class_indices)
                classlen = len(class_indices)

                valid_class_indices = valid_classlen_list[i]
                valid_classlen = len(valid_class_indices)

                test_class_indices = test_classlen_list[i]
                test_classlen = len(test_class_indices)

                for j in range(len(prob)):
                    # split dataset
                    left = int(np.sum(prob[:j, index + i]) / np.sum(prob[:, index + i]) * classlen)
                    right = int(np.sum(prob[:j + 1, index + i]) / np.sum(prob[:, index + i]) * classlen)

                    valid_left = int(np.sum(prob[:j, index + i]) / np.sum(prob[:, index + i]) * valid_classlen)
                    valid_right = int(np.sum(prob[:j + 1, index + i]) / np.sum(prob[:, index + i]) * valid_classlen)

                    test_left = int(np.sum(prob[:j, index + i]) / np.sum(prob[:, index + i]) * test_classlen)
                    test_right = int(np.sum(prob[:j + 1, index + i]) / np.sum(prob[:, index + i]) * test_classlen)
                    # print(i, datalen, left, right)
                    part_ind = class_indices[left:right]
                    if right - left > 0:
                        task_list['train'][j][0] += right - left
                        if str(k) not in task_list['train'][j][1].keys():
                            task_list['train'][j][1][str(k)] = part_ind
                        else:
                            task_list['train'][j][1][str(k)].extend(part_ind)

                    part_ind = valid_class_indices[valid_left:valid_right]
                    if valid_right - valid_left > 0:
                        task_list['valid'][j][0] += valid_right - valid_left
                        if str(k) not in task_list['valid'][j][1].keys():
                            task_list['valid'][j][1][str(k)] = part_ind
                        else:
                            task_list['valid'][j][1][str(k)].extend(part_ind)

                    part_ind = test_class_indices[test_left:test_right]
                    if test_right - test_left > 0:
                        task_list['test'][j][0] += test_right - test_left
                        if str(k) not in task_list['test'][j][1].keys():
                            task_list['test'][j][1][str(k)] = part_ind
                        else:
                            task_list['test'][j][1][str(k)].extend(part_ind)

        index += len(classlen_list)
    for i in range(len(task_list["train"])):
        task_list["train"][i][0] = math.ceil(task_list["train"][i][0] / args.per_gpu_train_batch_size)
        for j in task_list["train"][i][1].keys():
            part_ind = task_list["train"][i][1][j]
            random.shuffle(part_ind)
            task_list["train"][i][1][j] = sampler.SubsetRandomSampler(part_ind)

        task_list["valid"][i][0] = math.ceil(task_list["valid"][i][0] / args.per_gpu_eval_batch_size)
        for j in task_list["valid"][i][1].keys():
            task_list["valid"][i][1][j] = SubsetSequentialSampler(task_list["valid"][i][1][j])

        task_list["test"][i][0] = math.ceil(task_list["test"][i][0] / args.per_gpu_eval_batch_size)
        for j in task_list["test"][i][1].keys():
            task_list["test"][i][1][j] = SubsetSequentialSampler(task_list["test"][i][1][j])
    return task_list
