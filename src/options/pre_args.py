dataset_args = {
    "dbpedia_14": {
        'train_tasks': "dbpedia_14",
        "task_order": "d0",
        "max_datalen": -1,
        "test_max_datalen": -1,
    },
    'amazon_new': {
        'train_tasks': "amazon_new",
        "task_order": "d0",
        "max_datalen": -1,
        "test_max_datalen": -1,
    },
    'yahoo_answers_topics': {
        'train_tasks': "yahoo_answers_topics",
        "task_order": "d0",
        "max_datalen": -1,
        "test_max_datalen": -1,
    },
    'ag_news': {
        'train_tasks': "ag_news",
        "task_order": "d0",
        "max_datalen": -1,
        "test_max_datalen": -1,
    },
    'yelp': {
        'train_tasks': "yelp",
        "task_order": "d0",
        "max_datalen": -1,
        "test_max_datalen": -1,
    },
    "mnli": {
        'train_tasks': "mnli",
        'task_order': "d0",
        'max_datalen': -1,
        'test_max_datalen': -1,
    },
    "cb": {
        'train_tasks': "cb",
        'task_order': "d0",
        'max_datalen': -1,
        'test_max_datalen': -1,
    },
    "rte": {
        'train_tasks': "rte",
        'task_order': "d0",
        'max_datalen': -1,
        'test_max_datalen': -1,
    },
    "qqp": {
        'train_tasks': "qqp",
        'task_order': "d0",
        'max_datalen': -1,
        'test_max_datalen': -1,
    },
    "squad": {
        'train_tasks': "squad",
        'task_order': "d0",
        'max_datalen': -1,
        'test_max_datalen': -1,
    },
    'multirc': {
        'train_tasks': "multirc",
        'task_order': "d0",
        'max_datalen': -1,
        'test_max_datalen': -1,
    },
    'wic': {
        'train_tasks': "wic",
        'task_order': "d0",
        'max_datalen': -1,
        'test_max_datalen': -1,
    },
    'copa': {
        'train_tasks': "copa",
        'task_order': "d0",
        'max_datalen': -1,
        'test_max_datalen': -1,
    },
    'boolq': {
        'train_tasks': "boolq",
        'task_order': "d0",
        'max_datalen': -1,
        'test_max_datalen': -1,
    },
    'imdb': {
        'train_tasks': "imdb",
        'task_order': "d0",
        'max_datalen': -1,
        'test_max_datalen': -1,
    },
    'sst2': {
        'train_tasks': "sst2",
        'task_order': "d0",
        'max_datalen': -1,
        'test_max_datalen': -1,
    },
    'srl': {
        'train_tasks': "srl",
        'task_order': "d0",
        'max_datalen': -1,
        'test_max_datalen': -1,
    },
    'woz': {
        'train_tasks': "woz",
        'task_order': "d0",
        'max_datalen': -1,
        'test_max_datalen': -1,
    },
    "scl_seq1": {
        'train_tasks': "dbpedia_14:amazon_new:yahoo_answers_topics:ag_news",
        'task_order': "d0:d1:d2:d3",
        'max_datalen': -1,
        'test_max_datalen': -1,
    },
    "scl_seq2": {
        'train_tasks': "dbpedia_14:amazon_new:ag_news:yahoo_answers_topics",
        'task_order': "d0:d1:d2:d3",
        'max_datalen': -1,
        'test_max_datalen': -1,
    },
    "scl_seq3": {
        'train_tasks': "yahoo_answers_topics:amazon_new:ag_news:dbpedia_14",
        'task_order': "d0:d1:d2:d3",
        'max_datalen': -1,
        'test_max_datalen': -1,
    },
    "long_seq1": {
        'train_tasks': "mnli:cb:wic:copa:qqp:boolq:rte:imdb:yelp:amazon_new:sst2:dbpedia_14:ag_news:multirc:yahoo_answers_topics",
        'task_order': "d0:d1:d2:d3:d4:d5:d6:d7:d8:d9:d10:d11:d12:d13:d14",
        'max_datalen': -1,
        'text_max_datalen': -1,
    },
    "long_seq2": {
        'train_tasks': "multirc:boolq:wic:mnli:cb:copa:qqp:rte:imdb:sst2:dbpedia_14:ag_news:yelp:amazon_new:yahoo_answers_topics",
        'task_order': "d0:d1:d2:d3:d4:d5:d6:d7:d8:d9:d10:d11:d12:d13:d14",
        'max_datalen': -1,
        'text_max_datalen': -1,
    },
    "long_seq3": {
        'train_tasks': "yelp:amazon_new:mnli:cb:copa:qqp:rte:imdb:sst2:dbpedia_14:ag_news:yahoo_answers_topics:multirc:boolq:wic",
        'task_order': "d0:d1:d2:d3:d4:d5:d6:d7:d8:d9:d10:d11:d12:d13:d14",
        'max_datalen': -1,
        'text_max_datalen': -1,
    },
    "dis_seq1": {
        'train_tasks': 'yahoo_answers_topics:rte:qqp:cb:mnli',
        'task_order': 'd0:d1:d2:d3:d4',
        'max_datalen': -1,
        'test_max_datalen': -1,
    },
    "dis_seq2": {
        'train_tasks': 'qqp:rte:squad:mnli:cb',
        'task_order': 'd0:d1:d2:d3:d4',
        'max_datalen': -1,
        'test_max_datalen': -1,
    },
    "dis_seq3": {
        'train_tasks': 'multirc:rte:squad:wic:mnli',
        'task_order': 'd0:d1:d2:d3:d4',
        'max_datalen': -1,
        'test_max_datalen': -1,
    },
    "decanlp1": {
        'train_tasks': "srl:mnli:sst2:squad",
        'task_order': "d0:d1:d2:d3",
        'max_datalen': -1,
        'test_max_datalen': -1,
    },
    "decanlp2": {
        'train_tasks': "mnli:sst2:squad:srl",
        'task_order': "d0:d1:d2:d3",
        'max_datalen': -1,
        'test_max_datalen': -1,
    },
    "decanlp3": {
        'train_tasks': "mnli:squad:sst2:srl",
        'task_order': "d0:d1:d2:d3",
        'max_datalen': -1,
        'test_max_datalen': -1,
    },
}

task_args = {
    'total_mix': {
        'task_gen_type': 0,
        'sample_method': "total_random"
    },
    'dataset_stream': {
        'task_gen_type': 1,
        'task_boundary': 0,
        'major_num': 1,
        'corpus_involved': 0,
        'sample_method': "total_random"
    },
    'dataset_mix': {
        'task_gen_type': 1,
        'task_boundary': 1,
        'major_num': 1,
        'major_prob': 0.8,
        'major_step': -1,
        'corpus_involved': 1,
        'cover_width': 1,
        'sample_method': "total_random"
    },
}

train_args = {
    'train_offline_1_bsz': {
        'per_gpu_train_batch_size': 1,
        'per_gpu_eval_batch_size': 1,
        'save_model_online': False,
        'is_save_log_oss': False,
        'eval_every': 1,
        'cache_dir': '',
        'data_cache_dir': '../dataset'
    },
    'train_offline_2_bsz': {
        'per_gpu_train_batch_size': 2,
        'per_gpu_eval_batch_size': 4,
        'save_model_online': False,
        'is_save_log_oss': False,
        'eval_every': 1,
        'cache_dir': '',
        'data_cache_dir': '../dataset'
    },
    'train_offline_4_bsz': {
        'per_gpu_train_batch_size': 4,
        'per_gpu_eval_batch_size': 8,
        'save_model_online': False,
        'is_save_log_oss': False,
        'eval_every': 1,
        'cache_dir': '',
        'data_cache_dir': '../dataset'
    },
    'train_offline_8_bsz': {
        'per_gpu_train_batch_size': 8,
        'per_gpu_eval_batch_size': 16,
        'save_model_online': False,
        'is_save_log_oss': False,
        'eval_every': 1,
        'cache_dir': '',
        'data_cache_dir': '../dataset'
    },
    'train_offline_16_bsz': {
        'per_gpu_train_batch_size': 16,
        'per_gpu_eval_batch_size': 16,
        'save_model_online': False,
        'is_save_log_oss': False,
        'eval_every': 1,
        'cache_dir': '',
        'data_cache_dir': '../dataset'
    },
    'train_offline_16G': {
        'per_gpu_train_batch_size': 16,
        'per_gpu_eval_batch_size': 64,
        'save_model_online': False,
        'is_save_log_oss': False,
        'eval_every': 100,
        'cache_dir': '../../cache',
        'data_cache_dir': '../dataset'
    }
}

model_args = {
    't5-large-lr0.3': {
        'model_type': 't5-large',
        # 'max_seq_len': 256,
        'max_seq_len': 512,
        'learning_rate': 3e-1,
        'weight_decay': 1e-5,
    },
    't5-large-lr0.3-256': {
        'model_type': 't5-large',
        'max_seq_len': 256,
        'learning_rate': 3e-1,
        'weight_decay': 1e-5,
    },
    't5-large-lr0.3-256-coda': {
        'model_type': 't5-large',
        'max_seq_len': 256,
        'learning_rate': 3e-1,
        'weight_decay': 0,
    },
    't5-large-lr1e-4': {
        'model_type': 't5-large',
        'max_seq_len': 256,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
    },
    't5-large-lr4e-4': {
        'model_type': 't5-large',
        'max_seq_len': 256,
        'learning_rate': 4e-4,
        'weight_decay': 0.01,
    },
    't5-large-lr1e-5': {
        'model_type': 't5-large',
        'max_seq_len': 256,
        'learning_rate': 1e-5,
        'weight_decay': 0.01,
    },
    't5-large-lr1e-3': {
        'model_type': 't5-large',
        'max_seq_len': 256,
        'learning_rate': 1e-3,
        'weight_decay': 0.01,
    },
}

algo_args = {
    'shlpt': {

    },
    't5basic': {

    },
    't5ewc': {
        "ewc_gamma": 0.9,
        'reg_on_base': True,
        "reg_on_head": True,
    },
}
