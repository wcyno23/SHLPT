import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import random
import logging
import numpy as np
from torch.nn import LogSoftmax
from torch.nn import Softmax
from torch.nn.functional import kl_div
from utils.utils import init_weights
from copy import deepcopy

logger = logging.getLogger(__name__)


class ResMLP(torch.nn.Module):
    def __init__(self,
                 bottleneck_size,
                 module_type='MLP1',
                 emb_dimension=1024,
                 residual=True,
                 layer_norm=True,
                 ):
        """MLP class for soft prompt re-parameterization. MLP can have a Residual connection.
        Args:
            bottleneck_size (int): Dimension of the MLP bottlenack.
            module_type (str, optional): Type of MLP to be used.
                Currently supports 1-layer and 2-layer MLPs, and simple transformer layer ('MLP1'/'MLP2'/'transformer').
                Defaults to 'MLP1'.
            emb_dimension (int, optional): Dimension of T5 model embeddings. Defaults to 1024 (T5-large embedding dimension).
            residual (bool, optional): Whether to use residual connection in MLP. Defaults to True.
        """
        super().__init__()
        if module_type == 'MLP1':
            if layer_norm:
                self.module = nn.Sequential(
                    nn.Linear(emb_dimension, bottleneck_size),
                    nn.ReLU(),
                    nn.Linear(bottleneck_size, emb_dimension),
                    nn.LayerNorm(emb_dimension),
                )
            else:
                self.module = nn.Sequential(
                    nn.Linear(emb_dimension, bottleneck_size),
                    nn.Tanh(),
                    nn.Linear(bottleneck_size, emb_dimension),
                )

        elif module_type == 'MLP2':
            self.module = nn.Sequential(
                nn.Linear(emb_dimension, bottleneck_size),
                nn.ReLU(),
                nn.Linear(bottleneck_size, bottleneck_size // 2),
                nn.Tanh(),
                nn.Linear(bottleneck_size // 2, emb_dimension),
            )
        elif module_type == 'MLP3':
            self.module = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(emb_dimension, bottleneck_size),
                nn.Tanh(),
                nn.Linear(bottleneck_size, emb_dimension),
                nn.Dropout(0.1),
            )
        elif module_type == 'transformer':
            device = 'cuda'
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dimension, nhead=2, dropout=0.05).to(device)
            self.module = nn.TransformerEncoder(self.encoder_layer, num_layers=2).to(device)

        self.residual = residual
        if self.residual:
            print('Using skip connection in MLP')

    def forward(self, inputs):
        if self.residual:
            return self.module(inputs) + inputs
        else:
            return self.module(inputs)


class SimilarityEstimator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.add_prompt_lambda:
            prompt_lambda = torch.tensor([1.0])
            prompt_lambda.requires_grad = True
            self.prompt_lambda = nn.parameter.Parameter(prompt_lambda)


class FixedSimilarity(SimilarityEstimator):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        values = []
        for value in self.args.fixed_value.split(","):
            values.append(float(value))
        self.similarity = torch.FloatTensor(values)
        self.similarity.requires_grad = False
        self.similarity = self.similarity.to(self.args.device)

    def forward(self, c_embed, previous_prompts, prompt_embed, is_train=False):
        out = dict()
        pre_prompts = previous_prompts.repeat(c_embed.shape[0], 1, 1, 1)
        summed_prompts = torch.einsum('p, bpld -> bld', self.similarity, pre_prompts)
        out['prompt_embed'] = summed_prompts + prompt_embed
        out['similarity'] = self.similarity.repeat(c_embed.shape[0], 1)
        return out


class VoidEstimator(SimilarityEstimator):
    def __init__(self, args, emb_dimension=1024):
        super().__init__(args)
        self.args = args
        self.attn_W_down = nn.Linear(emb_dimension, 100, bias=False)
        self.attn_W_up = nn.Linear(100, emb_dimension, bias=False)
        self.attn_non_linear = nn.SiLU()
        self.layer_norm = nn.LayerNorm(emb_dimension)
        self.threshold = nn.Threshold(self.args.threshold, 0)

    def forward(self, c_embed, previous_prompts, prompt_embed, is_train=False):
        out = dict()
        avg_c_embed, _ = torch.max(c_embed, 1)
        pre_prompts = previous_prompts.repeat(c_embed.shape[0], 1, 1, 1)
        # involve current prompt when calculate similarity
        if self.args.involve_current_prompt:
            prompt_embed = prompt_embed.unsqueeze(1)
            pre_prompts = torch.cat([pre_prompts, prompt_embed], 1)
        avg_pre_prompts, _ = torch.max(pre_prompts, 2)
        # calculate similarity
        x = self.attn_W_down(avg_c_embed)
        x = self.attn_non_linear(x)
        x = self.attn_W_up(x)
        x = self.layer_norm(x)
        x = x.unsqueeze(-1)
        attn_scores = avg_pre_prompts.bmm(x).squeeze(-1) / self.args.softmax_temperature
        normalized_attn_scores = F.softmax(attn_scores, -1)
        filtered_attn_scores = self.threshold(normalized_attn_scores)
        batch_mean_scores = torch.mean(normalized_attn_scores, 0)
        dissimilar_index = torch.nonzero(batch_mean_scores < self.args.threshold).squeeze()
        out['dissimilar_index'] = dissimilar_index.repeat(c_embed.shape[0], 1)
        out['prompt_embed'] = prompt_embed
        out['similarity'] = filtered_attn_scores  #
        print(out['similarity'])
        return out


class SimplestEstimator(SimilarityEstimator):
    def __init__(self, args):
        super().__init__(args)
        self.args = args


class ConvolutionEstimator(SimilarityEstimator):
    def __init__(self, args):
        super().__init__(args)
        self.args = args


class AverageEstimator(SimilarityEstimator):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

    def forward(self, c_embed, previous_prompts, prompt_embed, is_train=False):
        out = dict()
        avg_previous_prompts = torch.mean(previous_prompts, dim=0)
        avg_previous_prompts = avg_previous_prompts.unsqueeze(0).repeat(c_embed.shape[0], 1, 1)
        pre_num = previous_prompts.size(0)
        out['dissimilar_index'] = None  # not add contrastive loss

        out['prompt_embed'] = avg_previous_prompts + prompt_embed
        similarity = torch.ones(pre_num).to(self.args.device) / pre_num
        out['similarity'] = similarity.repeat(c_embed.shape[0], 1)
        # print(out['similarity'])
        return out


class AttentionEstimator(SimilarityEstimator):
    def __init__(self, args, emb_dimension=1024):
        super().__init__(args)
        self.args = args
        self.attn_W_down = nn.Linear(emb_dimension, 100, bias=False)
        self.attn_W_up = nn.Linear(100, emb_dimension, bias=False)
        self.attn_non_linear = nn.SiLU()
        self.layer_norm = nn.LayerNorm(emb_dimension)
        self.threshold = nn.Threshold(self.args.threshold, 0)

    def forward(self, c_embed, previous_prompts, prompt_embed, is_train=False):
        out = dict()
        avg_c_embed, _ = torch.max(c_embed, 1)
        pre_prompts = previous_prompts.repeat(c_embed.shape[0], 1, 1, 1)
        # involve current prompt when calculate similarity
        if self.args.involve_current_prompt:
            prompt_embed = prompt_embed.unsqueeze(1)
            pre_prompts = torch.cat([pre_prompts, prompt_embed], 1)
        avg_pre_prompts, _ = torch.max(pre_prompts, 2)
        # calculate attention scores
        x = self.attn_W_down(avg_c_embed)
        x = self.attn_non_linear(x)
        x = self.attn_W_up(x)
        x = self.layer_norm(x)
        x = x.unsqueeze(-1)
        attn_scores = avg_pre_prompts.bmm(x).squeeze(-1) / self.args.softmax_temperature
        normalized_attn_scores = F.softmax(attn_scores, -1)
        filtered_attn_scores = self.threshold(normalized_attn_scores)

        if not self.args.all_dissimilar:
            filtered_attn_scores = filtered_attn_scores / torch.sum(filtered_attn_scores, dim=-1, keepdim=True)
        summed_prompts = torch.einsum('bp, bpld -> bld', filtered_attn_scores, pre_prompts)

        batch_mean_scores = torch.mean(normalized_attn_scores, 0)
        dissimilar_index = torch.nonzero(batch_mean_scores < self.args.threshold).squeeze()
        out['dissimilar_index'] = dissimilar_index.repeat(c_embed.shape[0], 1)

        if self.args.involve_current_prompt:
            return summed_prompts
        # if self.args.add_prompt_lambda:
        #     return summed_prompts * self.prompt_lambda + prompt_embed
        out['prompt_embed'] = summed_prompts + prompt_embed
        out['similarity'] = filtered_attn_scores  #
        return out


class SHLPTHead(nn.Module):
    def __init__(self, config, args, base_model, tokenizer):
        super(SHLPTHead, self).__init__()
        self.args = args
        prompt_number = args.prompt_number
        # initialize prompt embedding
        if self.args.use_pre_prompt:
            logger.info("use previous prompt to initialize")
            prompt_ckpt = torch.load(args.prompt_path, map_location={'cuda:0': 'cuda:3'})
            prompt_number_ckpt = prompt_ckpt['promptnumber']
            assert prompt_number == prompt_number_ckpt
            prompt_embedding = prompt_ckpt['promptembedding']
        else:
            prompt_embedding = self.init_new_prompt(base_model)
        self.prompt_number = prompt_number
        self.prompt_embedding = nn.parameter.Parameter(prompt_embedding)
        # Trained Task specific prompts
        self.prompts = []
        # Previous Prompts
        self.previous_prompts = None
        if self.args.load_pre_prompts:
            self.load_pre_prompt(self.args.pre_prompts_path, prompt_embedding, base_model)
        if self.args.estimator == "attention":
            self.estimator = AttentionEstimator(self.args)
        elif self.args.estimator == "fixed":
            self.estimator = FixedSimilarity(self.args)
        elif self.args.estimator == "void":
            self.estimator = VoidEstimator(self.args)
        elif self.args.estimator == "avg":
            self.estimator = AverageEstimator(self.args)
        else:
            raise NotImplementedError('Estimator [%s] is not implemented' % self.args.estimator)
        if self.args.add_kl_dis:
            self.softmax = Softmax(dim=2)
            self.log_softmax = LogSoftmax(dim=2)
        self.criterion = nn.CrossEntropyLoss()
        self.previous_tasks_hidden_states = {}
        self.previous_tasks_activations = {}
        self.train_batch_size = args.per_gpu_train_batch_size
        self.grads = {}
        if self.args.prefix_mlp:
            self.get_MLP("MLP1")

    # Create MLP for prompt tuning
    def get_MLP(self, prefix_MLP):
        if prefix_MLP == 'None':
            self.prefix_mlp = None
        else:
            print('Using MLP reparametrization with bottleneck = 1024')
        self.prefix_mlp = ResMLP(bottleneck_size=self.args.bottleneck_size,
                                 module_type=prefix_MLP)
        self.prefix_mlp.to(self.args.device)

    def init_mlp(self):
        init_weights(self.prefix_mlp)

    # Initialize new task prompt from random vocab. tokens
    def init_new_prompt(self, base_model, requires_grad=True):
        N = base_model.encoder.embed_tokens.weight.shape[0]
        prompt_weights = []
        for i in range(self.args.prompt_number):
            with torch.no_grad():
                j = np.random.randint(N)  # random token
                w = deepcopy(base_model.encoder.embed_tokens.weight[j].detach().cpu().numpy())
                prompt_weights.append(w)

        prompt_weights = np.array(prompt_weights)
        if requires_grad:
            prompt_embedding = torch.tensor(prompt_weights, requires_grad=True)
        else:
            prompt_embedding = torch.tensor(prompt_weights, requires_grad=False)
        return prompt_embedding

    # Load saved previous prompts for convenient testing
    def load_pre_prompt(self, paths, prompt_embed, base_model):
        logger.info("load previous prompt to test knowledge transfer")
        for path in paths.split(","):
            prompt_ckpt = torch.load(path, map_location={'cuda:0': 'cuda:3'})
            prompt_embedding = prompt_ckpt['promptembedding']
            prompt_embedding = torch.unsqueeze(prompt_embedding, 0)
            prompt_embedding = prompt_embedding.to(self.args.device)
            if self.previous_prompts is not None:
                self.previous_prompts = torch.cat([self.previous_prompts, prompt_embedding], 0)
            else:
                self.previous_prompts = prompt_embedding
        if self.args.involve_random_prompt:
            prompt_embed = self.init_new_prompt(base_model, requires_grad=False).to(self.args.device)
            prompt_embed = prompt_embed.unsqueeze(0)
            self.previous_prompts = torch.cat([self.previous_prompts, prompt_embed], 0)

    def update_task_embedding(self):
        if self.task_embedding is not None:
            self.task_embeddings.append(self.task_embedding(self.idx).clone().detach())
        del self.task_embedding
        self.task_embedding = torch.nn.Embedding(1, self.previous_prompts.size(0)).to(self.args.device)

    def forward(self, c_ids, c_mask, t_ids, t_mask, base_model, pad_token_id,
                task_id, ex_index=None, is_train=True):
        loss2 = None
        loss3 = None
        out = None

        encoder = base_model.encoder
        # calculate task loss
        t_ids[t_ids[:, :] == pad_token_id] = -100
        c_embed = encoder.embed_tokens(c_ids)
        if self.args.prefix_mlp:
            prompt_embed = self.prefix_mlp(self.prompt_embedding)
            prompt_embed = prompt_embed.repeat(c_embed.size(0), 1, 1)
        else:
            prompt_embed = self.prompt_embedding.repeat(c_embed.size(0), 1, 1)

        if self.previous_prompts is not None:
            out = self.estimator(c_embed, self.previous_prompts, prompt_embed, is_train)
            prompt_embed = out['prompt_embed']
        prompt_mask = torch.full((c_mask.size(0), prompt_embed.size(1)), 1).to(self.args.device)
        inputs_embeds = torch.cat([prompt_embed, c_embed], 1)
        attention_mask = torch.cat([prompt_mask, c_mask], 1)

        outputs = base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=t_ids,
            decoder_input_ids=None,
            decoder_attention_mask=t_mask,
            output_attentions=True,
            output_hidden_states=True
        )

        # calculate hsc loss & asc loss
        if self.previous_prompts is not None:
            # hsc loss
            if self.args.add_hsc_loss and is_train:
                dissimilar_index = out['dissimilar_index']
                if dissimilar_index.size(0) > 0:
                    decoder_hidden_states = outputs[3]
                    last_hidden_states = decoder_hidden_states[-1]
                    cos = nn.CosineSimilarity(dim=-1)
                    pos = cos(last_hidden_states, last_hidden_states)
                    pos = torch.mean(pos, dim=1)
                    logits = pos.reshape(-1, 1)
                    all_hidden_states = self.previous_tasks_hidden_states[ex_index]
                    all_hidden_states = all_hidden_states.to(self.args.device)

                    all_hidden_states = all_hidden_states[torch.arange(c_ids.size(0)).unsqueeze(1), dissimilar_index, :,
                                        :]
                    negs = None
                    for i in range(all_hidden_states.size(1)):
                        pre_last_hidden_states = all_hidden_states[:, i, :, :]
                        neg = cos(last_hidden_states, pre_last_hidden_states)
                        neg = torch.mean(neg, dim=1)
                        if negs is None:
                            negs = neg.reshape(-1, 1)
                        else:
                            negs = torch.cat([negs, neg.reshape(-1, 1)], dim=1)
                    if negs is not None:
                        if self.args.mean:
                            neg_mean = torch.mean(negs, dim=1).unsqueeze(-1)
                            logits = torch.cat([logits, neg_mean], dim=1)
                        else:
                            logits = torch.cat([logits, negs], dim=1)

                    logits /= self.args.hsc_temperature
                    labels = torch.zeros(c_ids.size(0)).cuda().long()
                    # TODO: not sure use mean on the output similarity, maybe need more views
                    loss2 = self.criterion(logits, labels) * self.args.hsc_lamb
            # asc loss
            if self.args.add_asc_loss_unfiltered and is_train:
                dissimilar_index = out['dissimilar_index']
                if dissimilar_index.size(0) > 0:
                    neurons = [[] for _ in range(len(self.args.layers))]
                    for k in range(len(self.args.layers)):
                        neurons[k] = torch.cat(self.neurons[k])

                    neurons = torch.stack(neurons)
                    neurons = neurons[:, :, :1, :]  # only use the first token
                    neurons = neurons.view(c_ids.size(0), 1, len(self.args.layers), -1)
                    # neuron after relu
                    activations = torch.relu(neurons)
                    # use straight through estimator
                    if self.args.use_ste:
                        sign_activations = (torch.sign(activations) - activations).detach() + activations

                    activations.register_hook(self.save_grad('activations'))
                    cos = nn.CosineSimilarity(dim=-1)
                    pos = cos(activations, activations)
                    logits = pos.reshape(-1, 1)
                    all_activations = self.previous_tasks_activations[ex_index]
                    all_activations = all_activations.to(self.args.device)
                    all_activations = all_activations[torch.arange(c_ids.size(0)).unsqueeze(1), dissimilar_index, :, :]
                    negs = None
                    for i in range(all_activations.size(1)):
                        pre_activations = all_activations[:, i, :, :]
                        pre_activations = pre_activations.unsqueeze(2)
                        if self.args.use_ste:
                            pre_sign_activations = (torch.sign(
                                pre_activations) - pre_activations).detach() + pre_activations
                            if self.args.add_not:
                                neg = cos((1 - sign_activations), (1 - pre_sign_activations))
                            else:
                                neg = cos(sign_activations, pre_sign_activations)
                        else:
                            # add not function to activations
                            if self.args.add_not:
                                neg = cos((1 - activations), (1 - pre_activations))
                            else:
                                neg = cos(activations, pre_activations)
                        if negs is None:
                            negs = neg.reshape(-1, 1)
                        else:
                            negs = torch.cat([negs, neg.reshape(-1, 1)], dim=1)

                    if negs is not None:
                        if self.args.mean:
                            neg_mean = torch.mean(negs, dim=1).unsqueeze(-1)
                            logits = torch.cat([logits, neg_mean], dim=1)
                        else:
                            logits = torch.cat([logits, negs], dim=1)
                    logits /= self.args.asc_temperature
                    labels = torch.zeros(c_ids.size(0)).cuda().long()
                    loss3 = self.criterion(logits, labels) * self.args.asc_lamb
            # asc loss filtered by neurons activated by instance X
            if self.args.add_asc_loss_filtered and is_train:
                dissimilar_index = out['dissimilar_index']
                if dissimilar_index.size(0) > 0:
                    neurons = [[] for _ in range(len(self.args.layers))]
                    for k in range(len(self.args.layers)):
                        neurons[k] = torch.cat(self.neurons[k])

                    neurons = torch.stack(neurons)
                    neurons = neurons[:, :, :1, :]  # only use the first token
                    neurons = neurons.view(c_ids.size(0), 1, len(self.args.layers), -1)
                    # neuron after relu
                    activations = torch.relu(neurons)
                    # use straight through estimator
                    if self.args.use_ste:
                        sign_activations = (torch.sign(activations) - activations).detach() + activations
                    activations.register_hook(self.save_grad('activations'))
                    cos = nn.CosineSimilarity(dim=-1)
                    pos = cos(activations, activations)
                    logits = pos.reshape(-1, 1)
                    all_activations = self.previous_tasks_activations[ex_index]
                    all_activations = all_activations.to(self.args.device)
                    all_activations = all_activations[torch.arange(c_ids.size(0)).unsqueeze(1), dissimilar_index, :, :]
                    activation_wo_prompt = self.previous_tasks_activations_wo_prompt[ex_index]
                    activation_wo_prompt = activation_wo_prompt.to(self.args.device)
                    activation_wo_prompt = activation_wo_prompt[torch.arange(c_ids.size(0)).unsqueeze(1), 0, :, :]
                    sign_activations_wo_prompt = torch.sign(activation_wo_prompt)
                    negs = None
                    for i in range(all_activations.size(1)):
                        pre_activations = all_activations[:, i, :, :]
                        pre_activations = pre_activations.unsqueeze(2)
                        # use straight through estimator
                        if self.args.use_ste:
                            pre_sign_activations = (torch.sign(
                                pre_activations) - pre_activations).detach() + pre_activations
                            # add not function to activations
                            if self.args.add_not:
                                neg = cos((1 - sign_activations), (1 - pre_sign_activations))
                            else:
                                filtered_sign_activations = sign_activations * (1 - sign_activations_wo_prompt)
                                filtered_pre_sign_activations = pre_sign_activations * (1 - sign_activations_wo_prompt)
                                neg = cos(filtered_sign_activations, filtered_pre_sign_activations)
                        else:
                            # add not function to activations
                            if self.args.add_not:
                                neg = cos((1 - activations), (1 - pre_activations))
                            else:
                                filtered_sign_activations = activations * (1 - sign_activations_wo_prompt)
                                filtered_pre_sign_activations = pre_activations * (1 - sign_activations_wo_prompt)
                                neg = cos(filtered_sign_activations, filtered_pre_sign_activations)
                        if negs is None:
                            negs = neg.reshape(-1, 1)
                        else:
                            negs = torch.cat([negs, neg.reshape(-1, 1)], dim=1)
                    if negs is not None:
                        if self.args.mean:
                            neg_mean = torch.mean(negs, dim=1).unsqueeze(-1)
                            logits = torch.cat([logits, neg_mean], dim=1)
                        else:
                            logits = torch.cat([logits, negs], dim=1)
                    logits /= self.args.asc_temperature
                    labels = torch.zeros(c_ids.size(0)).cuda().long()
                    loss3 = self.criterion(logits, labels) * self.args.asc_lamb

        loss = outputs[0]
        return loss, loss2, loss3

    def save_grad(self, name):
        def hook(grad):
            self.grads[name] = grad

        return hook
