import torch
import os
import torch.nn as nn
from dataset.config import *
from approaches.common import ContinualLearning
from dataset import load_labeled_dataset
from transformers import Adafactor
from torch.optim import AdamW
from fairscale.optim.oss import OSS
from sklearn.metrics import f1_score
from utils.utils import normalize_text, nf1_score
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SHLPTCL(ContinualLearning):
    def __init__(self, args, saver):
        super().__init__(args, saver)
        self.name = "BKT"
        self.args = args

    def setup_datasets(self):
        for task_name in self.args.train_tasks:
            task_path = os.path.join(self.args.data_cache_dir, DATA_DIR_MAP[task_name])
            if os.path.isdir(task_path):
                logger.info("The cache path %s of the dataset exists" % (task_path))
            else:
                logger.info("Download dataset to the cache path %s" % (task_path))
                self.saver.download_dataset(DATA_DIR_MAP[task_name], task_path)
                logger.info("Finish download the dataset %s" % (task_path))

        # Dividing a portion of samples from the test set as validation set.
        if self.args.split_test_file:
            self.train_datasets, _, label_maps, label2task_id = load_labeled_dataset(self.args, self.tokenizer)
            self.valid_datasets, self.test_datasets, self.label_maps, self.label2task_id = load_labeled_dataset(
                self.args, self.tokenizer, "test",
                label_maps, label2task_id, split_test_set=True)
        # Dividing a portion of samples from the training set as validation set.
        elif self.args.split_train_file_to_valid:
            logger.info("Split train file into train set and valid set")
            self.train_datasets, self.valid_datasets, label_maps, label2task_id = load_labeled_dataset(
                self.args, self.tokenizer, split_train_set_to_valid=True)
            self.test_datasets, _, self.label_maps, self.label2task_id = load_labeled_dataset(self.args, self.tokenizer,
                                                                                              "test",
                                                                                              label_maps, label2task_id)
        # Dividing a portion of samples from the training set as validation set and test set.
        elif self.args.split_train_file_to_valid_and_test:
            logger.info("Split train file into train set and valid, test set")
            self.train_datasets, self.valid_datasets, self.test_datasets, self.label_maps, self.label2task_id = load_labeled_dataset(
                self.args, self.tokenizer, split_train_set_to_valid_and_test=True)
        # Load exist validation set
        else:
            self.train_datasets, _, label_maps, label2task_id = load_labeled_dataset(self.args, self.tokenizer)
            self.test_datasets, _, label_maps, label2task_id = load_labeled_dataset(self.args, self.tokenizer, "test",
                                                                                    label_maps, label2task_id)
            self.valid_datasets, self.num_labels, self.label_maps, self.label2task_id = load_labeled_dataset(self.args,
                                                                                                             self.tokenizer,
                                                                                                             "valid",
                                                                                                             label_maps,
                                                                                                             label2task_id)

    def run_pri_batch(self, args, batch, pri_head, base_model):
        ex_index = None
        if self.args.add_hsc_loss or self.args.add_asc_loss_unfiltered or self.args.add_asc_loss_filtered:
            c_ids, c_mask, t_ids, t_mask, dataset_id, ex_index = batch
            # ex_index = ex_index.to(self.args.device)
        else:
            c_ids, c_mask, t_ids, t_mask, dataset_id = batch
        c_ids = c_ids.to(args.device)
        c_mask = c_mask.to(args.device)
        t_ids = t_ids.to(args.device)
        t_mask = t_mask.to(args.device)
        dataset_id = dataset_id.to(args.device)
        loss = pri_head(c_ids, c_mask, t_ids, t_mask, base_model,
                        self.tokenizer.pad_token_id, dataset_id, ex_index)
        return loss

    def continual_train(self, batch, task_info, step, total_steps, cls_losses, hsc_losses, asc_losses):
        loss, loss2, loss3 = self.run_pri_batch(self.args, batch, self.pri_head, self.base_model)
        cls_losses.append(loss.item())
        if loss2 is not None:
            hsc_losses.append(loss2.item())
            loss = loss + loss2
        if loss3 is not None:
            asc_losses.append(loss3.item())
            loss = loss + loss3
        loss = loss / self.args.gradient_accumulation_steps
        loss.backward()
        if step % self.args.gradient_accumulation_steps == 0 or step == total_steps - 1:
            torch.nn.utils.clip_grad_norm_(self.pri_head.parameters(), self.args.max_grad_norm)
            self.optimizer_pri.step()
            self.optimizer_pri.zero_grad()

        return loss * self.args.gradient_accumulation_steps

    def setup_optimizer(self, stage_id=None):

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.pri_head.named_parameters() if "mlp" in n],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate,
            },
            {
                "params": [p for n, p in self.pri_head.named_parameters() if "mlp" not in n],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate,
            }
        ]
        self.optimizer_pri = AdamW(optimizer_grouped_parameters, eps=1e-8)
        # self.optimizer_pri = Adafactor(optimizer_grouped_parameters, scale_parameter=False, relative_step=False,
        #                                warmup_init=False, eps=(1e-30, 1e-3))

        if self.args.distributed:
            base_optimizer_arguments = {"lr": self.args.learning_rate, "clip_threshold": self.args.max_grad_norm,
                                        "decay_rate": -0.8,
                                        "weight_decay": self.args.weight_decay,
                                        "scale_parameter": False, "relative_step": False, "warmup_init": False}
            self.optimizer_pri = OSS(params=filter(lambda p: p.requires_grad, self.pri_head.parameters()),
                                     optim=Adafactor, **base_optimizer_arguments)

    def get_metrics(self, dataset=None, datalen=0, flag=None, name=None, dataset_type="dataset"):
        self.pri_head.eval()
        self.base_model.eval()

        total_num = 0
        total_loss = 0
        hits = 0
        if dataset_type == "dataset":
            data_iter = iter(dataset.load_data())
            datalen = len(dataset.load_data())
            target_len = dataset.task_info["target_len"]
            task_name = dataset.task_info["task_name"]
        with torch.no_grad():
            pred_list = []
            target_list = []

            for step in range(datalen // self.args.n_gpu):
                if dataset_type == "dataset":
                    batch = next(data_iter)
                else:
                    task_info, batch = self.task_generator.get_batch_data(step)
                    target_len = task_info["target_len"]
                    task_name = task_info["task_name"]
                if self.args.add_hsc_loss or self.args.add_asc_loss_unfiltered or self.args.add_asc_loss_filtered:
                    c_ids, c_mask, t_ids, t_mask, dataset_id, ex_index = batch
                else:
                    c_ids, c_mask, t_ids, t_mask, dataset_id = batch
                    ex_index = None
                real_b = c_ids.size(0)
                total_num += real_b
                c_ids = c_ids.to(self.args.device)
                c_mask = c_mask.to(self.args.device)
                t_ids = t_ids.to(self.args.device)
                t_mask = t_mask.to(self.args.device)
                dataset_id = dataset_id.to(self.args.device)

                c_embed = self.base_model.encoder.embed_tokens(c_ids)
                if dataset_type == "dataset" and self.pri_head.previous_prompts is not None:
                    prompt_embed = self.pri_head.previous_prompts[dataset_id + self.args.pre_prompt_number]
                else:
                    if self.args.prefix_mlp:
                        prompt_embed = self.pri_head.prefix_mlp(self.pri_head.prompt_embedding)
                        prompt_embed = prompt_embed.repeat(c_embed.size(0), 1, 1)
                    else:
                        prompt_embed = self.pri_head.prompt_embedding.repeat(c_embed.size(0), 1, 1)
                    # Transfer Knowledge from previous prompts
                    if self.pri_head.previous_prompts is not None:
                        out = self.pri_head.estimator(c_embed, self.pri_head.previous_prompts, prompt_embed)
                        prompt_embed = out["prompt_embed"]
                        if step < 2:
                            # if step == 0 and dataset_type == "dataset":
                            out = self.pri_head.estimator(c_embed, self.pri_head.previous_prompts, prompt_embed)
                            similarity = out["similarity"]
                            # print(similarity)
                inputs_embeds = torch.cat([prompt_embed, c_embed], 1)
                prompt_mask = torch.full((c_mask.size(0), prompt_embed.size(1)), 1).to(self.args.device)
                attention_mask = torch.cat([prompt_mask, c_mask], 1)
                decoder_input_ids = (torch.ones((c_ids.size(0), 1),
                                                dtype=torch.long) * self.base_model.config.decoder_start_token_id).to(
                    self.args.device)
                outputs = self.base_model.generate(
                    inputs_embeds=inputs_embeds,
                    decoder_input_ids=decoder_input_ids,
                    attention_mask=attention_mask,
                    max_length=target_len,
                    use_cache=True,
                    num_beams=4,
                    repetition_penalty=2.5,
                    length_penalty=1.0,
                    early_stopping=True
                )
                predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True,
                                                          clean_up_tokenization_sapces=False)
                targets = self.tokenizer.batch_decode(t_ids, skip_special_tokens=True,
                                                      clean_up_tokenization_sapces=False)
                contents = self.tokenizer.batch_decode(c_ids, skip_special_tokens=True,
                                                       clean_up_tokenization_sapces=False)

                predictions = list(map(normalize_text, map(str.strip, predictions)))
                targets = list(map(normalize_text, map(str.strip, targets)))
                contents = list(map(str.strip, contents))

                pred_results, target_results = predictions, targets
                for i in range(len(pred_results)):
                    pred_result = pred_results[i]
                    target_result = target_results[i]
                    if pred_result.lower() == target_result.lower():
                        hits += 1
                    else:
                        pass

                pred_list += pred_results
                target_list += target_results

            test_acc = hits / total_num
            # TODO: correct f1 calculation
            test_primary_metric_score = test_acc
            # test_f1 = f1_score(target_list, pred_list)

            test_loss = total_loss / total_num
            if task_name in ['multirc']:
                test_f1 = f1_score(target_list, pred_list, average='micro')
                test_primary_metric_score = test_f1
            elif task_name in ['squad', 'nqopen', 'srl']:
                test_f1 = 0
                for i in range(len(pred_list)):
                    test_f1 += nf1_score(pred_list[i], target_list[i])
                test_f1 = test_f1 / total_num
                test_primary_metric_score = test_f1

        if test_primary_metric_score == test_acc:
            logger.info(
                '>>> {:5s} on task {:15s}: loss={:.3f}, accuracy={:5.1f}% <<<'.format(flag, name, test_loss,
                                                                                                 test_acc * 100))
        else:
            logger.info(
                '>>> {:5s} on task {:15s}: loss={:.3f}, accuracy={:5.1f}%, f1={:2.2f}<<<'.format(flag, name, test_loss,
                                                                                            test_acc * 100,
                                                                                            test_f1 * 100))
        if self.args.is_save_log:
            if test_primary_metric_score == test_acc:
                self.add_log(
                    '>>> {:5s} on task {:15s}: loss={:.3f}, accuracy={:5.1f}% <<<'.format(flag, name, test_loss,
                                                                                                    test_acc * 100))
            else:
                self.add_log(
                '>>> {:5s} on task {:15s}: loss={:.3f}, accuracy={:5.1f}%, f1={:2.2f}<<<'.format(flag, name, test_loss,
                                                                                                test_acc * 100,
                                                                                                test_f1 * 100))

        self.pri_head.train()
        self.base_model.train()
        # Get the metrics from the classifier
        torch.cuda.empty_cache()
        return test_loss, test_acc, test_primary_metric_score

    def evaluate(self, flag, name):
        # logger.info("eval begin")
        flag = flag.lower()
        if "stage" in name:
            self.task_generator.get_dataset_iter(flag)
            datalen = self.task_list[flag][int(name[6:])][0]
            if datalen == 0:
                logger.info('>>> {:5s} on task {:15s}: has no data in evaluation set <<<'.format(flag, name))
                if self.args.is_save_log:
                    self.add_log('>>> {:5s} on task {:15s}: has no data in evaluation set <<<'.format(flag, name))
                return 0, 0, 0
            test_loss, test_acc, test_f1 = self.get_metrics(datalen=datalen, flag=flag, name=name,
                                                            dataset_type="generator")
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
            test_loss, test_acc, test_f1 = self.get_metrics(dataset=dataset, flag=flag, name=name,
                                                            dataset_type="dataset")

        return test_loss, test_acc, test_f1

    def do_each_stage(self, stage_id):
        self.pri_head.isTraining = True

        if self.pri_head.previous_prompts is not None:
            # random initialize new prompt parameter for next task
            with torch.no_grad():
                prompt_embedding = self.pri_head.init_new_prompt(self.base_model).to(self.args.device)
                self.pri_head.prompt_embedding.data = prompt_embedding
                # if self.args.prefix_mlp:
                    # init mlp in pri head
                    # self.pri_head.init_mlp()
                    # logger.info(">>> re init mlp in pri head <<<")

        # save previous prompt's activation states
        if self.args.add_asc_loss_unfiltered or self.args.add_asc_loss_filtered:
            self.pri_head.neurons = [[] for _ in range(len(self.args.layers))]

            def hook_fn(idx):
                def fn(_, __, output):
                    self.pri_head.neurons[idx] = [output]

                return fn

            for idx, n in enumerate(self.args.layers):
                self.base_model.decoder.block[n].layer[2].DenseReluDense.wi.register_forward_hook(hook_fn(idx))

        if self.pri_head.previous_prompts is not None and self.args.add_asc_loss_unfiltered:
            del self.pri_head.previous_tasks_activations
            logger.info(">> calculate previous tasks' activations")
            with torch.no_grad():
                self.task_generator.load_stage_dataset(stage_id)
                self.task_generator.get_dataset_iter("train")
                total = self.task_list["train"][stage_id][0] // self.args.n_gpu
                activations = {}
                for step in range(self.task_list["train"][stage_id][0] // self.args.n_gpu):
                    task_info, batch = self.task_generator.get_batch_data(step)
                    if batch is None:
                        break
                    c_ids, c_mask, t_ids, t_mask, dataset_id, ex_index = batch
                    c_ids = c_ids.to(self.args.device)
                    c_mask = c_mask.to(self.args.device)
                    t_ids = t_ids.to(self.args.device)
                    t_mask = t_mask.to(self.args.device)
                    t_ids[t_ids[:, :] == self.tokenizer.pad_token_id] = -100
                    c_embed = self.base_model.encoder.embed_tokens(c_ids)
                    all_activations = None
                    for idx, pre_prompt in enumerate(self.pri_head.previous_prompts):
                        pre_prompt_embed = pre_prompt.repeat(c_embed.size(0), 1, 1)
                        pre_inputs_embeds = torch.cat([pre_prompt_embed, c_embed], 1)
                        prompt_mask = torch.full((c_mask.size(0), pre_prompt_embed.size(1)), 1).to(self.args.device)
                        attention_mask = torch.cat([prompt_mask, c_mask], 1)

                        pre_outputs = self.base_model(
                            inputs_embeds=pre_inputs_embeds,
                            attention_mask=attention_mask,
                            labels=t_ids,
                            decoder_attention_mask=t_mask,
                            decoder_input_ids=None,
                            output_attentions=True,
                            output_hidden_states=True,
                        )
                        neurons = [[] for _ in range(len(self.args.layers))]
                        for k in range(len(self.args.layers)):
                            neurons[k] = torch.cat(self.pri_head.neurons[k])
                        neurons = torch.stack(neurons)
                        neurons = neurons[:, :, :1, :]  # only use the first token
                        neurons = neurons.view(c_ids.size(0), 1, len(self.args.layers), -1)
                        neuron_after_relu = torch.relu(neurons)
                        # sign_neuron = torch.sign(neuron_after_relu)
                        # sign_index = torch.nonzero(sign_neuron[0][0][0]).squeeze()
                        # print(sign_index, sign_index.size(), idx)
                        if all_activations is None:
                            all_activations = neuron_after_relu
                        else:
                            all_activations = torch.cat([all_activations, neuron_after_relu], dim=1)
                    for i in range(all_activations.size(0)):
                        activations[ex_index[i].item()] = all_activations[i].cpu().detach().numpy()
                sorted_activations = [v for k, v in sorted(activations.items(), key=lambda item: item[0])]
                sorted_activations = torch.tensor(np.array(sorted_activations))
            self.pri_head.previous_tasks_activations = sorted_activations
            logger.info(">> calculate activations finish")

        if self.pri_head.previous_prompts is not None and self.args.add_asc_loss_filtered:
            del self.pri_head.previous_tasks_activations
            logger.info(">> calculate previous tasks' activations")
            with torch.no_grad():
                self.task_generator.load_stage_dataset(stage_id)
                self.task_generator.get_dataset_iter("train")
                total = self.task_list["train"][stage_id][0] // self.args.n_gpu
                activations = {}
                activations_wo_prompt = {}
                for step in range(self.task_list["train"][stage_id][0] // self.args.n_gpu):
                    task_info, batch = self.task_generator.get_batch_data(step)
                    if batch is None:
                        break
                    c_ids, c_mask, t_ids, t_mask, dataset_id, ex_index = batch
                    c_ids = c_ids.to(self.args.device)
                    c_mask = c_mask.to(self.args.device)
                    t_ids = t_ids.to(self.args.device)
                    t_mask = t_mask.to(self.args.device)
                    t_ids[t_ids[:, :] == self.tokenizer.pad_token_id] = -100
                    c_embed = self.base_model.encoder.embed_tokens(c_ids)
                    all_activations = None
                    pre_outputs = self.base_model(
                        inputs_embeds=c_embed,
                        attention_mask=c_mask,
                        labels=t_ids,
                        decoder_attention_mask=t_mask,
                        decoder_input_ids=None,
                        output_attentions=True,
                        output_hidden_states=True,
                    )
                    neurons = [[] for _ in range(len(self.args.layers))]
                    for k in range(len(self.args.layers)):
                        neurons[k] = torch.cat(self.pri_head.neurons[k])
                    neurons = torch.stack(neurons)
                    neurons = neurons[:, :, :1, :]  # only use the first token
                    neurons = neurons.view(c_ids.size(0), 1, len(self.args.layers), -1)
                    neuron_after_relu = torch.relu(neurons)
                    sign_neuron_wo_prompt = torch.sign(neuron_after_relu)
                    for i in range(sign_neuron_wo_prompt.size(0)):
                        activations_wo_prompt[ex_index[i].item()] = sign_neuron_wo_prompt[i].cpu().detach().numpy()

                    for idx, pre_prompt in enumerate(self.pri_head.previous_prompts):
                        pre_prompt_embed = pre_prompt.repeat(c_embed.size(0), 1, 1)
                        pre_inputs_embeds = torch.cat([pre_prompt_embed, c_embed], 1)
                        prompt_mask = torch.full((c_mask.size(0), pre_prompt_embed.size(1)), 1).to(self.args.device)
                        attention_mask = torch.cat([prompt_mask, c_mask], 1)

                        pre_outputs = self.base_model(
                            inputs_embeds=pre_inputs_embeds,
                            attention_mask=attention_mask,
                            labels=t_ids,
                            decoder_attention_mask=t_mask,
                            decoder_input_ids=None,
                            output_attentions=True,
                            output_hidden_states=True,
                        )
                        neurons = [[] for _ in range(len(self.args.layers))]
                        for k in range(len(self.args.layers)):
                            neurons[k] = torch.cat(self.pri_head.neurons[k])
                        neurons = torch.stack(neurons)
                        neurons = neurons[:, :, :1, :]  # only use the first token
                        neurons = neurons.view(c_ids.size(0), 1, len(self.args.layers), -1)
                        neuron_after_relu = torch.relu(neurons)
                        # sign_neuron = torch.sign(neuron_after_relu)
                        # sign_index = torch.nonzero(sign_neuron[0][0][0]).squeeze()
                        # print(sign_index, sign_index.size(), idx)
                        if all_activations is None:
                            all_activations = neuron_after_relu
                        else:
                            all_activations = torch.cat([all_activations, neuron_after_relu], dim=1)
                    for i in range(all_activations.size(0)):
                        activations[ex_index[i].item()] = all_activations[i].cpu().detach().numpy()
                sorted_activations = [v for k, v in sorted(activations.items(), key=lambda item: item[0])]
                sorted_activations = torch.tensor(np.array(sorted_activations))
                sorted_activations_wo_prompt = [v for k, v in
                                                sorted(activations_wo_prompt.items(), key=lambda item: item[0])]
                sorted_activations_wo_prompt = torch.tensor(np.array(sorted_activations_wo_prompt))
            self.pri_head.previous_tasks_activations = sorted_activations
            self.pri_head.previous_tasks_activations_wo_prompt = sorted_activations_wo_prompt
            logger.info(">> calculate activations finish")

        # save previous task's hidden states
        if self.pri_head.previous_prompts is not None and self.args.add_hsc_loss:
            del self.pri_head.previous_tasks_hidden_states
            logger.info(">> calculate previous tasks' last hidden states")
            with torch.no_grad():
                self.task_generator.load_stage_dataset(stage_id)
                self.task_generator.get_dataset_iter("train")

                total = self.task_list["train"][stage_id][0] // self.args.n_gpu
                # print(total)
                # TODO: optimize the storage
                hidden_states = {}
                for step in range(self.task_list["train"][stage_id][0] // self.args.n_gpu):
                    task_info, batch = self.task_generator.get_batch_data(step)
                    if batch is None:
                        break
                    c_ids, c_mask, t_ids, t_mask, dataset_id, ex_index = batch
                    c_ids = c_ids.to(self.args.device)
                    c_mask = c_mask.to(self.args.device)
                    t_ids = t_ids.to(self.args.device)
                    t_mask = t_mask.to(self.args.device)
                    t_ids[t_ids[:, :] == self.tokenizer.pad_token_id] = -100
                    c_embed = self.base_model.encoder.embed_tokens(c_ids)
                    all_hidden_states = None
                    for pre_prompt in self.pri_head.previous_prompts:
                        pre_prompt_embed = pre_prompt.repeat(c_embed.size(0), 1, 1)
                        pre_inputs_embeds = torch.cat([pre_prompt_embed, c_embed], 1)
                        prompt_mask = torch.full((c_mask.size(0), pre_prompt_embed.size(1)), 1).to(self.args.device)
                        attention_mask = torch.cat([prompt_mask, c_mask], 1)

                        pre_outputs = self.base_model(
                            inputs_embeds=pre_inputs_embeds,
                            attention_mask=attention_mask,
                            labels=t_ids,
                            decoder_attention_mask=t_mask,
                            decoder_input_ids=None,
                            output_attentions=True,
                            output_hidden_states=True,
                        )
                        pre_decoder_hidden_states = pre_outputs[3]
                        pre_last_hidden_states = pre_decoder_hidden_states[-1]
                        pre_last_hidden_states = pre_last_hidden_states.unsqueeze(1)
                        if all_hidden_states is None:
                            all_hidden_states = pre_last_hidden_states
                        else:
                            all_hidden_states = torch.cat([all_hidden_states, pre_last_hidden_states], dim=1)
                    for i in range(all_hidden_states.size(0)):
                        hidden_states[ex_index[i].item()] = all_hidden_states[i].cpu().detach().numpy()
                sorted_states = [v for k, v in sorted(hidden_states.items(), key=lambda item: item[0])]
                sorted_states = torch.tensor(np.array(sorted_states))
            self.pri_head.previous_tasks_hidden_states = sorted_states
            logger.info(">> calculate last hidden states finish")

        # visualization
        if self.pri_head.previous_prompts is not None and self.args.add_asc_loss_unfiltered and self.args.visualize:
            del self.pri_head.previous_tasks_activations
            logger.info(">> calculate previous tasks' activations")
            with torch.no_grad():
                self.task_generator.load_stage_dataset(stage_id)
                self.task_generator.get_dataset_iter("train")
                total = self.task_list["train"][stage_id][0] // self.args.n_gpu
                activations = {}
                for step in range(self.task_list["train"][stage_id][0] // self.args.n_gpu):
                    task_info, batch = self.task_generator.get_batch_data(step)
                    if batch is None:
                        break
                    c_ids, c_mask, t_ids, t_mask, dataset_id, ex_index = batch
                    c_ids = c_ids.to(self.args.device)
                    c_mask = c_mask.to(self.args.device)
                    t_ids = t_ids.to(self.args.device)
                    t_mask = t_mask.to(self.args.device)
                    t_ids[t_ids[:, :] == self.tokenizer.pad_token_id] = -100
                    c_embed = self.base_model.encoder.embed_tokens(c_ids)
                    all_activations = None
                    for idx, pre_prompt in enumerate(self.pri_head.previous_prompts):
                        pre_prompt_embed = pre_prompt.repeat(c_embed.size(0), 1, 1)
                        pre_inputs_embeds = torch.cat([pre_prompt_embed, c_embed], 1)
                        prompt_mask = torch.full((c_mask.size(0), pre_prompt_embed.size(1)), 1).to(self.args.device)
                        attention_mask = torch.cat([prompt_mask, c_mask], 1)
                        # without prompt
                        # pre_inputs_embeds = c_embed
                        # attention_mask = c_mask

                        pre_outputs = self.base_model(
                            inputs_embeds=pre_inputs_embeds,
                            attention_mask=attention_mask,
                            labels=t_ids,
                            decoder_attention_mask=t_mask,
                            decoder_input_ids=None,
                            output_attentions=True,
                            output_hidden_states=True,
                        )
                        neurons = [[] for _ in range(len(self.args.layers))]
                        for k in range(len(self.args.layers)):
                            neurons[k] = torch.cat(self.pri_head.neurons[k])
                        neurons = torch.stack(neurons)
                        neurons = neurons[:, :, :1, :]  # only use the first token
                        neurons = neurons.view(c_ids.size(0), 1, len(self.args.layers), -1)
                        neuron_after_relu = torch.relu(neurons)
                        # sign_neuron = torch.sign(neuron_after_relu)
                        # sign_index = torch.nonzero(sign_neuron[0][0][0]).squeeze()
                        # print(sign_index, sign_index.size(), idx)
                        if all_activations is None:
                            all_activations = neuron_after_relu
                        else:
                            all_activations = torch.cat([all_activations, neuron_after_relu], dim=1)
                    for i in range(all_activations.size(0)):
                        activations[ex_index[i].item()] = all_activations[i].cpu().detach().numpy()
                sorted_activations = [v for k, v in sorted(activations.items(), key=lambda item: item[0])]
                act_path = "../../ckpt/activation/" + self.args.act_path
                sorted_activations = torch.tensor(np.array(sorted_activations))
                torch.save(sorted_activations, act_path + "activations.pt")
            self.pri_head.previous_tasks_activations = sorted_activations
            logger.info(">> calculate activations finish")
            exit()


    def do_after_stage(self, stage_id):
        logger.info(">> save current task prompt")
        with torch.no_grad():
            self.pri_head.isTraining = False
            if stage_id == 0 and self.pri_head.previous_prompts is None:
                if self.args.prefix_mlp:
                    prompt = self.pri_head.prefix_mlp(self.pri_head.prompt_embedding)
                    self.pri_head.previous_prompts = prompt.clone().detach().requires_grad_(False).to(
                        self.args.device).unsqueeze(0)
                else:
                    self.pri_head.previous_prompts = self.pri_head.prompt_embedding.clone().detach().requires_grad_(
                        False).to(self.args.device).unsqueeze(0)

            elif stage_id == 1:
                if self.args.prefix_mlp:
                    prompt = self.pri_head.prefix_mlp(self.pri_head.prompt_embedding)
                    summed_prompt_embedding = self.pri_head.previous_prompts + prompt
                else:
                    if self.args.all_dissimilar:
                        summed_prompt_embedding = self.pri_head.prompt_embedding.unsqueeze(0)
                    else:
                        summed_prompt_embedding = self.pri_head.previous_prompts + self.pri_head.prompt_embedding
                summed_prompt_embedding = summed_prompt_embedding.clone().detach().requires_grad_(False).to(
                    self.args.device)
                self.pri_head.previous_prompts = torch.cat(
                    [self.pri_head.previous_prompts, summed_prompt_embedding],
                    0)
            else:
                total_sim = None
                self.task_generator.get_dataset_iter("train")
                total_steps = self.task_list["train"][stage_id][0] // self.args.n_gpu
                for step in range(total_steps):
                    task_info, batch = self.task_generator.get_batch_data(0)
                    c_ids = batch[0]
                    c_ids = c_ids.to(self.args.device)
                    c_embed = self.base_model.encoder.embed_tokens(c_ids)
                    if self.args.prefix_mlp:
                        prompt_embed = self.pri_head.prefix_mlp(self.pri_head.prompt_embedding)
                        prompt_embed = prompt_embed.repeat(c_embed.size(0), 1, 1)
                    else:
                        prompt_embed = self.pri_head.prompt_embedding.repeat(c_embed.size(0), 1, 1)
                    out = self.pri_head.estimator(c_embed, self.pri_head.previous_prompts, prompt_embed)
                    similarity = out['similarity']
                    mean_sim = torch.mean(similarity, dim=0)
                    if total_sim is None:
                        total_sim = mean_sim
                    else:
                        total_sim += mean_sim
                total_sim /= (total_steps)
                composed_prompt_embedding = torch.einsum('p, pld -> ld', total_sim, self.pri_head.previous_prompts)
                if self.args.prefix_mlp:
                    prompt = self.pri_head.prefix_mlp(self.pri_head.prompt_embedding)
                    summed_prompt_embedding = composed_prompt_embedding + prompt
                else:
                    summed_prompt_embedding = composed_prompt_embedding + self.pri_head.prompt_embedding
                summed_prompt_embedding = summed_prompt_embedding.unsqueeze(0)
                summed_prompt_embedding = summed_prompt_embedding.clone().detach().requires_grad_(False).to(
                    self.args.device)
                self.pri_head.previous_prompts = torch.cat(
                    [self.pri_head.previous_prompts, summed_prompt_embedding],
                    0)
