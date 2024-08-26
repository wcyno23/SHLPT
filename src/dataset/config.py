TASK_ID_NAME_MAP = {}

T5_CSV_TASK_NAME_LIST = [
    'dbpedia_14',
    'amazon_new',
    'yahoo_answers_topics',
    'ag_news',
    'yelp'
]

T5_TSV_TASK_NAME_LIST = [
    'sst2',
    'rte',
    'mnli',
]

T5_JSONL_TASK_NAME_LIST = [
    'wic',
    'cb',
    'copa',
    'boolq',
    'multirc',
    'nqopen',
    'qqp',
]

T5_JSON_TASK_NAME_LIST = [
    'squad',
    'imdb',
    'srl',
    'woz'
]

VALID_FIlE_ONLY_TASK_LIST = [
    'dbpedia_14',
    'amazon_new',
    'yahoo_answers_topics',
    'ag_news',
    'yelp',
    'sst2',
    'rte',
    'wic',
    'cb',
    'copa',
    'boolq',
    'multirc',
    'squad',
    'nqopen',
    'qqp',
    'mnli',
    'imdb',
    'srl',
    'woz',
]

QA_TASK_NAME_LIST = [
    'squad',
    'nqopen',
    'srl',
    'woz',
]

DATA_DIR_MAP = {
    "sst": "sst",
    "srl": "srl",
    'dbpedia_14': "dbpedia_14",
    'amazon_new': "amazon_new",
    'yahoo_answers_topics': "yahoo_answers_topics",
    'ag_news': "ag_news",
    'yelp': "yelp",
    'sst2': "SST-2",
    'rte': "RTE",
    'wic': "WiC",
    'cb': "CB",
    'copa': "COPA",
    'boolq': "BoolQ",
    'multirc': "MultiRC",
    'squad': "SQUAD",
    'nqopen': "NQopen",
    'qqp': "QQP",
    'mnli': "MNLI",
    'imdb': 'IMDB',
}


# Column keys used in the dataset
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    # "qnli": ("question", "sentence"),
    "qnli": ("text1", "text2"),
    "qqp": ("text1", "text2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),

    "boolq": ("passage", "question"),
    "copa": ('choice1', 'choice2', 'premise', 'question'),
    "wic": ("start1", "end1", "sentence1", "start2", "end2", "sentence2", "word"),
    "wsc": ("span1_text", "span1_index", "span2_text", "span2_index", "text"),
    "wsc_bool": ("span1_text", "span1_index", "span2_text", "span2_index", "text"),
    "cb": ("premise", "hypothesis"),
    "record": ("passage", "query", "entities"),
    "multirc": ("question", "answer", "paragraph"),
    "rte_superglue": ("premise", "hypothesis"),

    "scicite": ("sectionName", "string"),
    "imdb": ("sentence", None),

    "ag_news": ("title", "content"),
    # "ag_news": ("label", "title", "content"),
    "yelp_review_full": ("text", None),
    "yahoo_answers_topics": ("question_title", "question_content", "best_answer"),
    # "yahoo_answers_topics": ("label", "title", "content"),
    "dbpedia_14": ("title", "content"),

    "ag": ("content", None),
    "yelp": ("content",None),
    "yahoo": ("content", None),
    "dbpedia": ("content", None),
    "amazon": ("content", None),
    "amazon_new": ("title", "content"),
    "nqopen": ("question", None),
}

# Label text for T5 tasks
# (T5 has text-to-text format for text and labels)
task_to_labels = {
    "cola": ("not_acceptable", "acceptable"),
    "mnli": ("entailment", "neutral", "contradiction"),
    "mnli-mm": (),
    "mrpc": ("not_equivalent", "equivalent"),
    "qnli": ("entailment", "not_entailment"),
    "qqp": ("not_duplicate", "duplicate"),
    "rte": ("entailment", "not_entailment"),
    "sst2": ("negative", "positive"),
    "stsb": (),
    "wnli": (),

    "boolq": ("false", "true"),
    "copa": ("false", "true"),
    "wic": ("false", "true"),
    "wsc_bool": ("false", "true"),
    "cb": ("entailment", "contradiction", "neutral"),
    "multirc": ("false", "true"),
    "rte_superglue": ("entailment", "not_entailment"),

    "scicite": (),
    "imdb": ("negative", "positive"),

    "ag_news": ("world", "sports", "business", "science"),
    "yelp_review_full": ("terrible", "bad", "middle", "good", "wonderful"),
    "yahoo_answers_topics": ("society and culture", "science", "health", "education and reference",
                             "computers and internet", "sports", "business", "entertainment and music",
                             "family and relationships", "politics and government"),
    "dbpedia_14": ("company", "educationalinstitution", "artist", "athlete", "officeholder",
                   "meanoftransportation", "building", "naturalplace", "village", "animal",
                   "plant", "album", "film", "writtenwork"),

    "ag": ("world", "sports", "business", "science"),
    "yelp": ("terrible", "bad", "middle", "good", "wonderful"),
    "yahoo": ("society and culture", "science", "health", "education and reference",
              "computers and internet", "sports", "business", "entertainment and music",
              "family and relationships", "politics and government"),
    "dbpedia": ("company", "educationalinstitution", "artist", "athlete", "officeholder",
                "meanoftransportation", "building", "naturalplace", "village", "animal",
                "plant", "album", "film", "writtenwork"),
    "amazon": ("terrible", "bad", "middle", "good", "wonderful"),
    "amazon_new": ("terrible", "bad", "middle", "good", "wonderful"),
}
