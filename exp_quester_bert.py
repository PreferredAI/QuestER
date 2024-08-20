import os
import cornac
from cornac.data import Reader
from eval_method import QuestERStratifiedSplit
from text_modality import ReviewAndItemQAModality
from quester_bert import QuestER
from cornac.data.text import BaseTokenizer
import numpy as np

import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass


map_name_to_handle = {
    'small_bert/bert_en_uncased_L-2_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2',
    'small_bert/bert_en_uncased_L-4_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/2',
    'small_bert/bert_en_uncased_L-6_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/2',
    'small_bert/bert_en_uncased_L-8_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/2',
    'small_bert/bert_en_uncased_L-10_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/2',
    'small_bert/bert_en_uncased_L-12_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/2',
    'albert_en_base':
        'https://tfhub.dev/tensorflow/albert_en_base/2',
    'electra_small':
        'https://tfhub.dev/google/electra_small/2',
    'electra_base':
        'https://tfhub.dev/google/electra_base/2',
    'experts_pubmed':
        'https://tfhub.dev/google/experts/bert/pubmed/2',
    'experts_wiki_books':
        'https://tfhub.dev/google/experts/bert/wiki_books/2',
    'talking-heads_base':
        'https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1',
}

map_model_to_preprocess = {
    'small_bert/bert_en_uncased_L-2_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'albert_en_base':
        'https://tfhub.dev/tensorflow/albert_en_preprocess/3',
    'electra_small':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'electra_base':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'experts_pubmed':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'experts_wiki_books':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'talking-heads_base':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
}


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="input directory")
    parser.add_argument("-ct", "--cluster_threshold", type=float, default=0.8)
    parser.add_argument("-mu", "--min_user_freq", type=int, default=5)
    parser.add_argument("-mi", "--min_item_freq", type=int, default=5)
    parser.add_argument("-na", "--max_num_answer", type=int, default=1)
    parser.add_argument("-k", "--n_factors", type=int, default=8)
    parser.add_argument("-d", "--mlp_out_factors", type=int, default=128)
    parser.add_argument("-e", "--epoch", type=int, default=20)
    parser.add_argument(
        "-b", "--bert_model",
        type=str,
        default="small_bert/bert_en_uncased_L-2_H-128_A-2",
        choices=[
            'small_bert/bert_en_uncased_L-2_H-128_A-2',
            'small_bert/bert_en_uncased_L-4_H-128_A-2',
            'small_bert/bert_en_uncased_L-6_H-128_A-2',
            'small_bert/bert_en_uncased_L-8_H-128_A-2',
            'small_bert/bert_en_uncased_L-10_H-128_A-2',
            'small_bert/bert_en_uncased_L-12_H-128_A-2',
            'albert_en_base',
            'electra_small',
            'electra_base',
            'experts_pubmed',
            'experts_wiki_books',
            'talking-heads_base'
        ]
    )
    parser.add_argument("-bs", "--batch_size", type=int, default=64)
    parser.add_argument(
        "-s", "--model_selection", type=str, choices=["best", "last"], default="best"
    )
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    return parser.parse_args()


args = parse_arguments()
feedback = Reader(min_user_freq=args.min_user_freq, min_item_freq=args.min_item_freq).read(os.path.join(args.input, "rating.txt"), fmt="UIRT", sep="\t")
reviews = Reader().read(
    os.path.join(args.input, "review.txt"), fmt="UIReview", sep="\t"
)

data_dir = args.input
MAX_VOCAB = 4000
EMB_SIZE = 100
ID_EMB_SIZE = args.n_factors
N_FACTORS = args.n_factors
ATTENTION_SIZE = 8
BATCH_SIZE = args.batch_size
MAX_NUM_REVIEW = 32
MAX_NUM_QUESTION = 32
MAX_NUM_ANSWER = args.max_num_answer
DROPOUT_RATE = 0.5
TEST_SIZE = 0.1
VAL_SIZE = 0.1
CLUSTER_THRESHOLD = 0.8
centroid_questions_file = open(os.path.join(data_dir, "centroid_questions.txt"), "r")
centroid_questions = centroid_questions_file.readlines()
cluster_label_in_order = []
cluster_count = []
with open(os.path.join(data_dir, "cluster.count"), "r") as f:
    for line in f:
        tokens = line.split(",")
        cluster_label_in_order.append(int(tokens[0]))
        cluster_count.append(int(tokens[1]))

pct = np.array(cluster_count) / sum(cluster_count)
max_keep_idx = 0
for i in range(len(pct)):
    if pct[: i + 1].sum() >= args.cluster_threshold:
        max_keep_idx = i + 1
        break

print("Max keep idx (coverage:{}): {}".format(args.cluster_threshold, max_keep_idx))

item_question_clusters = {}
with open(os.path.join(data_dir, "item_question_clusters.txt"), "r") as f:
    for line in f:
        tokens = line.split(",")
        item_question_clusters[tokens[0]] = [int(cluster) for cluster in tokens[1:]]
qas = []
with open(os.path.join(data_dir, "qa.txt"), "r") as f:
    for line in f:
        tokens = line.split("\t\t")
        asin = tokens[0]
        qas.append(
            (
                asin,
                [
                    tuple(
                        [
                            qtoken
                            for q_inc, qtoken in enumerate(question.split("\t"))
                            if q_inc % 2 == 0
                        ]
                    )
                    for question, cluster_label in zip(
                        tokens[1:], item_question_clusters.get(asin, [])
                    )
                    if cluster_label in cluster_label_in_order[:max_keep_idx]
                ],
            )
        )

mean_question = " ".join(centroid_questions[max_keep_idx:]).replace("\n", " ")

item_with_qas = [x[0] for x in qas]
item_without_qas = list(set([x[1] for x in feedback if x[1] not in item_with_qas]))
[x[1].append((mean_question,)) for x in qas]
qas = qas + [(x, [(mean_question,)]) for x in item_without_qas]

review_and_item_qa_modality = ReviewAndItemQAModality(
    data=reviews,
    qa_data=qas,
    tokenizer=BaseTokenizer(stop_words="english"),
    max_vocab=MAX_VOCAB,
)

eval_method = QuestERStratifiedSplit(
    data=feedback,
    group_by="item",
    test_size=TEST_SIZE,
    val_size=VAL_SIZE,
    exclude_unknowns=True,
    review_and_item_qa_text=review_and_item_qa_modality,
    verbose=True,
    seed=123,
)


models = [
    QuestER(
        name=f"{os.path.basename(data_dir)}_QuestERBERT_{'_'.join(args.bert_model.split('/'))}_F_{args.n_factors}_A_{ATTENTION_SIZE}_NReview_{MAX_NUM_REVIEW}_NQuestion_{MAX_NUM_QUESTION}_NAnswer_{MAX_NUM_ANSWER}_E_{args.epoch}_BS_{BATCH_SIZE}",
        embedding_size=EMB_SIZE,
        id_embedding_size=ID_EMB_SIZE,
        n_factors=args.n_factors,
        attention_size=ATTENTION_SIZE,
        mlp_out_factors=args.mlp_out_factors,
        dropout_rate=DROPOUT_RATE,
        max_num_review=MAX_NUM_REVIEW,
        max_num_question=MAX_NUM_QUESTION,
        max_num_answer=MAX_NUM_ANSWER,
        batch_size=BATCH_SIZE,
        max_iter=args.epoch,
        model_selection=args.model_selection,
        preprocessor_url=map_model_to_preprocess[args.bert_model],
        encoder_url=map_name_to_handle[args.bert_model],
        optimizer="adam",
        learning_rate=args.learning_rate,
        verbose=True,
        seed=123,
    )
]
exp = cornac.Experiment(
    eval_method=eval_method,
    models=models,
    metrics=[
        cornac.metrics.MSE(),
    ],
)

exp.run()
print(data_dir)
selected_model = models[0]
epoch = selected_model.best_epoch if args.model_selection == 'best' else args.epochs.split(',')[0]
model_name = '{}_e_{}'.format(selected_model.name, epoch)
export_dir = os.path.join(args.input, model_name)
os.makedirs(export_dir, exist_ok=True)
import util
from importlib import reload
if args.model_selection == 'best':
    util.export_ranked_questions(selected_model, os.path.join(export_dir, 'ranked_questions.txt'))
    util.export_useful_review_ranking(selected_model, os.path.join(export_dir, 'useful_review_ranking.txt'))
    util.export_most_useful_review(selected_model, os.path.join(export_dir, 'most_useful_review.txt'))
    util.export_important_question_ranking(selected_model, os.path.join(export_dir, 'important_question_ranking.txt'))
    util.export_quester_explanations(selected_model, export_dir)
# import pdb; pdb.set_trace()