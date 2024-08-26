import numpy as np
import torch.utils.data.sampler as sampler
import torch.utils.data.distributed as Distributed
import math
from tasks.sample import *


class TaskGenerator(object):
    def __init__(self, args, train_datasets, valid_datasets, test_datasets) -> None:
        self.train_datasets = train_datasets
        self.valid_datasets = valid_datasets
        self.test_datasets = test_datasets
        self.args = args
        self.curr_task_idx = 0
        self.current_task_index = -1
        self.curr_stage_id = -1
        self.task_list = None
        self.task_num = -1

    def assign_tasks(self):
        dataset_len, t_total = 0, 0
        dataset_list, valid_dataset_list, test_dataset_list = [], [], []
        for id, task in enumerate(self.args.task_order):
            dataset = self.train_datasets[self.args.train_tasks[int(task[1:])]]
            if self.args.task_gen_type == 1 or self.args.task_gen_type == 0:
                dataset_list.append(len(dataset))
                valid_dataset_list.append(len(self.valid_datasets[self.args.train_tasks[int(task[1:])]]))
                test_dataset_list.append(len(self.test_datasets[self.args.train_tasks[int(task[1:])]]))
            elif self.args.task_gen_type == 2:
                dataset_list.append(dataset.dataset.class_list)
                valid_dataset_list.append(
                self.valid_datasets[self.args.train_tasks[int(task[1:])]].dataset.class_list)
                test_dataset_list.append(
                self.test_datasets[self.args.train_tasks[int(task[1:])]].dataset.class_list)

            t_total += math.ceil(len(dataset) / (self.args.per_gpu_train_batch_size))
        if self.args.task_gen_type == 0:
            self.task_list = sample_task_in_domain_level(self.args, dataset_list, valid_dataset_list, test_dataset_list)
        elif self.args.task_gen_type == 1:
            self.task_list = sample_task_in_dataset_level(self.args, dataset_list, valid_dataset_list,
                                                          test_dataset_list)
        elif self.args.task_gen_type == 2:
            self.task_list = sample_task_in_class_level(self.args, dataset_list, valid_dataset_list, test_dataset_list)
        self.task_num = len(self.task_list["train"])
        return t_total, self.task_list

    def load_stage_dataset(self, stage_id):
        # load task data in this stage
        self.curr_stage_id = stage_id

        self.data_infos = []
        self.data_loaders = []
        for id in self.task_list["train"][stage_id][1].keys():
            task = self.args.task_order[int(id)]
            data = self.train_datasets[self.args.train_tasks[int(task[1:])]]
            if self.args.distributed:
                self.data_loaders.append(data.load_data(Distributed.DistributedSampler(data.dataset, shuffle=True)))
            else:
                self.data_loaders.append(data.load_data(self.task_list["train"][stage_id][1][id]))
            self.data_infos.append(data.task_info)
            self.data_infos[-1]["datalen"] = len(self.data_loaders[-1])

        self.valid_data_infos = []
        self.valid_data_loaders = []
        for id in self.task_list["valid"][stage_id][1].keys():
            task = self.args.task_order[int(id)]
            data = self.valid_datasets[self.args.train_tasks[int(task[1:])]]
            if self.args.distributed:
                self.valid_data_loaders.append(
                    data.load_data(Distributed.DistributedSampler(data.dataset, shuffle=False)))
            else:
                self.valid_data_loaders.append(data.load_data(self.task_list["valid"][stage_id][1][id]))
            self.valid_data_infos.append(data.task_info)
            self.valid_data_infos[-1]["datalen"] = len(self.valid_data_loaders[-1])

        self.test_data_infos = []
        self.test_data_loaders = []
        for id in self.task_list["test"][stage_id][1].keys():
            task = self.args.task_order[int(id)]
            data = self.test_datasets[self.args.train_tasks[int(task[1:])]]
            if self.args.distributed:
                self.test_data_loaders.append(
                    data.load_data(Distributed.DistributedSampler(data.dataset, shuffle=False)))
            else:
                self.test_data_loaders.append(data.load_data(self.task_list["test"][stage_id][1][id]))
            self.test_data_infos.append(data.task_info)
            self.test_data_infos[-1]["datalen"] = len(self.test_data_loaders[-1])
        return self.data_infos

    def get_dataset_iter(self, fn="train"):
        if fn == "train":
            data_infos = self.data_infos
            data_loaders = self.data_loaders
        elif fn == "valid":
            data_infos = self.valid_data_infos
            data_loaders = self.valid_data_loaders
        elif fn == "test":
            data_infos = self.test_data_infos
            data_loaders = self.test_data_loaders
        self.curr_task_idx = 0
        self.epoch_data_infos = [i for i in data_infos]
        self.epoch_data_iterator = [iter(data_loader) for data_loader in data_loaders]
        self.epoch_data_lenlist = np.array([len(data_loader) for data_loader in data_loaders])

    def get_stage_length(self):
        if self.curr_stage_id == -1:
            return 0
        else:
            return self.task_list["train"][self.curr_stage_id][0]

    def get_batch_data(self, step):
        try:
            if self.args.sample_method == "total_random":
                self.curr_task_idx = list(
                    np.random.multinomial(1, self.epoch_data_lenlist / sum(self.epoch_data_lenlist))).index(1)
                # self.epoch_data_lenlist[self.curr_task_idx] -= 1
            else:
                if self.args.task_chang_freq != -1 and step % self.args.task_chang_freq == 0:
                    self.curr_task_idx = np.random.randint(0, len(self.epoch_data_infos))
            task_info = self.epoch_data_infos[self.curr_task_idx]
            batch = next(self.epoch_data_iterator[self.curr_task_idx])
            # print("<<"*10)
        except:
            print("The train data %s is used out!" % task_info["task_name"])
            flag = True
            while flag:
                del self.epoch_data_infos[self.curr_task_idx]
                del self.epoch_data_iterator[self.curr_task_idx]

                if len(self.epoch_data_infos) == 0:
                    return None, None

                self.curr_task_idx = 0
                task_info = self.epoch_data_infos[self.curr_task_idx]
                try:
                    batch = next(self.epoch_data_iterator[self.curr_task_idx])
                    flag = False
                except:
                    print("The train data %s is used out!" % task_info["task_name"])
        self.epoch_data_lenlist[self.curr_task_idx] -= 1

        return task_info, batch

