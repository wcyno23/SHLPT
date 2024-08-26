import os
import logging
from transformers import (
	AutoConfig,
	AutoModel,
	AutoTokenizer,
)
import re
import torch
from models.shlpt_head import SHLPTHead

logger = logging.getLogger(__name__)


def create_model(args):

    if args.model_type == 't5-large':
        config = AutoConfig.from_pretrained(args.model_name_or_path, output_hidden_states=True)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, output_hidden_states=True)
    elif args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, output_hidden_states=True)
    else:
        # When we release a pip version exposing CONFIG_MAPPING,
        # we can do `config = CONFIG_MAPPING[args.model_type]()`.
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --config_name"
        )

    # Setting up tokenizer and pre-trained model
    if args.model_type == "t5-large":
        # tokenizer = AutoTokenizer.from_pretrained(args.model_type)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, model_max_length=512)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    elif args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )
    if args.algo in ["progressive_prompts", "shlpt", "t5basic", "t5ewc", "l2p", "coda", 't5er', 't5derpp']:
        if "t5-v1_1" in args.model_type:
            from transformers import T5ForConditionalGeneration
            model = T5ForConditionalGeneration.from_pretrained(
                args.model_name_or_path
            )
        elif "t5" in args.model_type:
            from transformers import T5ForConditionalGeneration
            model = T5ForConditionalGeneration.from_pretrained(
                args.model_type
            )
        elif "gpt2" in args.model_type:
            from transformers import GPT2LMHeadModel
            model = GPT2LMHeadModel.from_pretrained(
                args.model_name_or_path
            )
        # for name, param in model.named_parameters():
        #     param.requires_grad = False
    elif args.algo in ["adapter", "cla"]:
        from models.base_arch.bert_adapter import MyBertModel
        model = MyBertModel.from_pretrained(
                args.model_name_or_path,
                config=config,
                args=args)
    else:
        if args.model_name_or_path:
            model = AutoModel.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config
            )
        else:
            model = AutoModel.from_config(config)
    
    # if not hasattr(tokenizer, "pad_token"):
    if not tokenizer.pad_token:
        logger.info("model %s has no attribute pad_token"%(args.model_name_or_path))
        tokenizer.pad_token = tokenizer.unk_token
        config.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        logger.info("set %s as the pad token, and its index is %d"%(tokenizer.unk_token, config.pad_token_id))


    if args.algo in ["progressive_prompts", "shlpt", "l2p"]:
        for param in model.parameters():
            param.requires_grad = False

    return model, tokenizer, config





def create_pri_head(config, args, num_labels, label2task_id, base_model, tokenizer):
    pri_model = None
    if args.pri_task == "shlpt":
        pri_model = SHLPTHead(config, args, base_model, tokenizer)
    return pri_model


def run_pri_batch(base_model, model, batch, args):
    if len(batch) == 6:
        inputs, segment_ids, input_mask, labels, dataset_id, task_id = batch
        extra_info = None
    else:
        inputs, segment_ids, input_mask, labels, dataset_id, task_id, extra_info = batch
        extra_info = extra_info.to(args.device)
    inputs = inputs.to(args.device)
    segment_ids = segment_ids.to(args.device)
    input_mask = input_mask.to(args.device)
    labels = labels.to(args.device)
    dataset_id = dataset_id.to(args.device)
    task_id = task_id.to(args.device)
    outputs = model(base_model, inputs, 
                    token_type_ids=segment_ids, 
                    attention_mask=input_mask, 
                    labels=labels,
                    dataset_ids=dataset_id,
                    task_ids=task_id,
                    extra_info=extra_info)
    return outputs
