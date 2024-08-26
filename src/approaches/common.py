import torch
import logging
import os
import time

from copy import deepcopy
from transformers import get_linear_schedule_with_warmup
from models import create_model, create_pri_head, run_pri_batch
from dataset import load_labeled_dataset
from utils import utils
from dataset.config import *
from tasks.generator import TaskGenerator
from torch.cuda.amp import GradScaler

logger = logging.getLogger(__name__)

class ContinualLearning():
    def __init__(self, args, saver):
        # args
        self.args = args
        # models
        self.base_model = None
        self.pri_head = None
        # data
        self.train_datasets = None
        self.test_datasets = None
        self.valid_datasets = None
        # task_generator
        self.task_generator = None
        self.t_total = None
        self.task_list = None
        self.task_num = None

        self.saver = saver
        self.log_recorder = []

        self.model_names = ["base_model", "pri_head"]
        self.best_model_cache = {}

        self.num_labels = None

        self.device = args.device
        self.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

        self.scaler = None

    def setup_base_models(self):
        self.down_num = len(self.args.train_tasks)
        setattr(self.args, 'down_num', self.down_num)
        model_name = self.args.model_name_or_path
        assert model_name, 'The name of the model is not Set. Maybe use roberta-base as the default'

        if os.path.isdir(self.args.model_name_or_path):
            logger.info("The cache path %s of the pretrained model exists" % (self.args.model_name_or_path))
        else:
            logger.info("Download model to the cache path %s" % (self.args.model_name_or_path))
            self.saver.download_pretrain_model(self.args.model_type, self.args.model_name_or_path)
            logger.info("Finish download the model %s" % (self.args.model_type))

        self.base_model, self.tokenizer, self.config = create_model(self.args)
        self.base_model.to(self.device)

    def setup_datasets(self):
        # Setting up the dataset
        for task_name in self.args.train_tasks:
            task_path = os.path.join(self.args.data_cache_dir, DATA_DIR_MAP[task_name])
            if os.path.isdir(task_path):
                logger.info("The cache path %s of the dataset exists" % (task_path))
            else:
                logger.info("Download dataset to the cache path %s" % (task_path))
                self.saver.download_dataset(DATA_DIR_MAP[task_name], task_path)
                logger.info("Finish download the dataset %s" % (task_path))

        self.train_datasets, _, label_maps, label2task_id = load_labeled_dataset(self.args, self.tokenizer)
        self.test_datasets, _, label_maps, label2task_id = load_labeled_dataset(self.args, self.tokenizer, "test",
                                                                                label_maps, label2task_id)
        self.valid_datasets, self.num_labels, self.label_maps, self.label2task_id = load_labeled_dataset(self.args,
                                                                                                         self.tokenizer,
                                                                                                         "valid",
                                                                                                         label_maps,
                                                                                                         label2task_id)

    def setup_tasks(self):
        # the task generator to assign data
        self.task_generator = TaskGenerator(self.args, self.train_datasets, self.valid_datasets,
                                            self.test_datasets)
        self.t_total, self.task_list = self.task_generator.assign_tasks()
        self.task_num = len(self.task_list["train"])
        setattr(self.args, 'task_num', self.task_num)

    def setup_head_models(self):
        self.args.embedding_size = len(self.tokenizer)
        self.pri_head = create_pri_head(self.config, self.args, self.num_labels, self.label2task_id, self.base_model,
                                        self.tokenizer)
        self.base_model.to(self.device)
        if self.pri_head is not None:
            self.pri_head.to(self.device)

        if self.args.block_size <= 0:
            self.args.block_size = self.tokenizer.model_max_length
            # Our input block size will be the max possible for the model
        else:
            self.args.block_size = min(self.args.block_size, self.tokenizer.model_max_length)

    def setup_before_task(self):
        logger.info("do the external initial before task setup each model")
        # TODO:
        logger.info("Nothing here")

    def setup_external(self):
        logger.info("do the external initial for each model.")
        # TODO: 
        logger.info("Nothing here.")

    def do_each_stage(self):
        logger.info("do the preparation before each stage.")
        # TODO: 
        logger.info("Nothing here.")

    def do_after_stage(self):
        logger.info("do the ending-jobs after each stage.")
        # TODO: 
        logger.info("Nothing here.")

    def process_task_batch(self, batch, task_info, return_type=0):
        task_type = task_info["task_type"]
        if task_type in ["seq_cla", "token_cla", "qa"]:
            output = run_pri_batch(self.base_model, self.pri_head, batch, self.args)
        else:
            raise ValueError("unrecognized task type named %s !" % task_type)

        if return_type == 0:
            loss = output[0]
            return loss
        elif return_type == 1:
            label = output[-1]
            return label
        else:
            return output

    def add_log(self, txt):
        self.log_recorder.append("Time %s:   %s" % (time.strftime('%Y-%m-%d %H:%M:%S'), txt))

    def save_log(self):
        if self.args.is_save_log_oss:
            self.saver.put_object(self.args.oss_log_path, "\n".join(self.log_recorder))
        if self.args.is_save_log_disc:
            with open(self.args.disc_log_path, "w") as f:
                f.write("\n".join(self.log_recorder))

    def evaluate(self, flag, name):
        print(self.scheduler.get_lr())
        flag = flag.lower()
        if "stage" in name:
            self.task_generator.get_dataset_iter(flag)
            datalen = self.task_list[flag][int(name[6:])][0]
            if datalen == 0:
                logger.info('>>> {:5s} on task {:15s}: has no data in evaluation set <<<'.format(flag, name))
                if self.args.is_save_log:
                    self.add_log('>>> {:5s} on task {:15s}: has no data in evaluation set <<<'.format(flag, name))
                return 0, 0, 0
            test_loss, test_acc, f1 = self.get_metrics(datalen=datalen, flag=flag, name=name, dataset_type="generator")
        else:
            if flag == "train":
                dataset = self.train_datasets[name]
            elif flag == "test":
                dataset = self.test_datasets[name]
            elif flag == "valid":
                dataset = self.valid_datasets[name]
            else:
                raise ValueError(
                    "No such dataset \"%s\" to evaluate !!!" % flag)
            if name in SQUAD_TASK_NAME_LIST:
                pass
                em, nf1, nem = self.get_metrics(dataset=dataset, flag=flag, name=name, dataset_type="dataset")

            else:
                test_loss, test_acc, f1 = self.get_metrics(dataset=dataset, flag=flag, name=name,
                                                           dataset_type="dataset")
        return test_loss, test_acc, f1

    def get_metrics(self, dataset=None, datalen=0, flag=None, name=None, dataset_type="dataset"):
        self.pri_head.eval()
        self.base_model.eval()

        total_loss = 0
        total_acc = 0
        total_num = 0
        target_list = []
        pred_list = []
        i = 0
        if dataset_type == "dataset":
            data_iter = iter(dataset.load_data())
            datalen = len(dataset.load_data())
        with torch.no_grad():
            for step in range(datalen // self.args.n_gpu):
                if dataset_type == "dataset":
                    batch = next(data_iter)
                else:
                    _, batch = self.task_generator.get_batch_data(step)
                inputs, segment_ids, input_mask, labels, dataset_id, task_id = batch
                real_b = inputs.size(0)
                inputs = inputs.to(self.args.device)
                segment_ids = segment_ids.to(self.args.device)
                input_mask = input_mask.to(self.args.device)
                labels = labels.to(self.args.device)
                dataset_id = dataset_id.to(self.args.device)
                task_id = task_id.to(self.args.device)
                output = self.pri_head(self.base_model, inputs,
                                       token_type_ids=segment_ids,
                                       attention_mask=input_mask,
                                       labels=labels,
                                       dataset_ids=dataset_id,
                                       task_ids=task_id)
                loss = output[0]
                _, pred = output[1].max(1)
                labels = output[-1]

                hits = sum(list(pred == labels))
                target_list.append(labels)
                pred_list.append(pred)
                # Log
                total_loss += loss.data.cpu().numpy().item() * real_b
                total_acc += hits
                total_num += real_b
                i += 1
            test_loss, test_acc = total_loss / total_num, total_acc / total_num
            if name in ["chemprot", "rct"]:
                f1_type = 'micro'
            else:
                f1_type = 'macro'
            f1 = utils.f1_compute_fn(y_pred=torch.cat(pred_list, 0), y_true=torch.cat(target_list, 0), average=f1_type)

        logger.info(
            '>>> {:5s} on task {:15s}: loss={:.3f}, acc={:5.1f}%, f1-{:5s}={:2.2f} <<<'.format(flag, name, test_loss,
                                                                                               100 * test_acc, f1_type,
                                                                                               f1 * 100))
        if self.args.is_save_log:
            self.add_log('>>> {:5s} on task {:15s}: loss={:.3f}, acc={:5.1f}%, f1-{:5s}={:2.2f} <<<'.format(flag, name,
                                                                                                            test_loss,
                                                                                                            100 * test_acc,
                                                                                                            f1_type,
                                                                                                            f1 * 100))

        self.pri_head.train()
        self.base_model.train()
        # Get the metrics from the classifier
        torch.cuda.empty_cache()
        return test_loss, test_acc, f1

    def save_models(self, id, path=None):
        # for name in self.model_names:
        #     if isinstance(name, str):
        #         net = getattr(self, name)
        #         if self.args.save_model_online:
        #             self.saver.save(net, id, name)
        #             continue
        #         save_filename = '%s_%s.pth' % (str(id), name)
        #         save_path = os.path.join(self.save_dir, save_filename)
        #         if len(self.gpu_ids) > 0 and torch.cuda.is_available():
        #             torch.save(net.state_dict(), save_path)
        #         else:
        #             torch.save(net.cpu().state_dict(), save_path)
        for name in ["pri_head"]:
            if isinstance(name, str):
                net = getattr(self, name)
                if self.args.distributed:
                    net = net.module
                if self.args.algo == "shlpt":
                    ckpt = {
                        "promptnumber": net.prompt_number,
                        "promptembedding": net.previous_prompts[-1],
                    }
                else:
                    ckpt = {
                        "promptnumber": net.prompt_number,
                        "promptembedding": net.prompt_embedding
                    }
                if self.args.use_hat_mask:
                    ckpt["taskembedding"] = net.task_embedding
                if self.args.prefix_mlp:
                    ckpt["mlp"] = net.mlp
                if path is not None:
                    torch.save(ckpt, path)
                else:
                    torch.save(ckpt, self.args.save_path + 'stage' + str(id))

    def load_model(self, id):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                if self.args.save_model_online:
                    self.saver.load(net, id, name, self.device)
                    continue
                load_filename = '%s_%s.pth' % (str(id), name)
                if self.args.load_name:
                    load_path = os.path.join(self.save_dir, self.args.load_name, load_filename)
                else:
                    load_path = os.path.join(self.save_dir, load_filename)
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location=str(self.device))
                net.load_state_dict(state_dict)

    def save_best_models_to_cache(self):
        # for name in self.model_names:
        for name in ["pri_head"]:
            if isinstance(name, str):
                net = getattr(self, name)
                if self.args.distributed:
                    net = net.module
                self.best_model_cache[name] = deepcopy(net.state_dict())

    def load_best_models_from_cache(self):
        # for name in self.model_names:
        for name in ["pri_head"]:
            if isinstance(name, str):
                net = getattr(self, name)
                if self.args.distributed:
                    net = net.module
                net.load_state_dict(deepcopy(self.best_model_cache[name]))

    def setup_optimizer(self, stage_id=None):
        raise NotImplementedError(
            "Please Implement the `setup_optimizer` method in your class.")

    def setup_schedule(self, steps):
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=int(self.args.warmup_frac * steps), num_training_steps=steps)
        self.scheduler_pri = get_linear_schedule_with_warmup(
            self.optimizer_pri, num_warmup_steps=int(self.args.warmup_frac * steps), num_training_steps=steps)

        logger.info("schedule Setup ...... Done!")

    def setup_scaler(self):
        if self.args.use_scaler:
            self.scaler = GradScaler()

    def continual_train(self):
        raise NotImplementedError(
            "Please Implement the `continual_train_one_batch` method in your class.")
