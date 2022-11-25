import os
import cornac
from cornac.data import Reader
from eval_method import QuestERStratifiedSplit
from text_modality import ReviewAndItemQAModality
from quester import QuestER
from cornac.data.text import BaseTokenizer
import numpy as np

import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="input directory")
    parser.add_argument("-ct", "--cluster_threshold", type=float, default=0.8)
    parser.add_argument("-k", "--n_factors", type=int, default=8)
    parser.add_argument("-e", "--epoch", type=int, default=20)
    parser.add_argument("-bs", "--batch_size", type=int, default=64)
    parser.add_argument(
        "-s", "--model_selection", type=str, choices=["best", "last"], default="best"
    )
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    return parser.parse_args()


args = parse_arguments()
feedback = Reader().read(os.path.join(args.input, "rating.txt"), fmt="UIRT", sep="\t")
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
MAX_TEXT_LENGTH = 128
DROPOUT_RATE = 0.5
TEST_SIZE = 0.1
VAL_SIZE = 0.1
KERNEL_SIZES = [3]
N_FILTERS = 64
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
item_without_qas = list(set([x[1] for x in reviews if x[1] not in item_with_qas]))
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

pretrained_word_embeddings = {}
with open(f"download/glove/glove.6B.{EMB_SIZE}d.txt", encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        pretrained_word_embeddings[word] = coefs

exp = cornac.Experiment(
    eval_method=eval_method,
    models=[
        QuestER(
            name=f"QuestER",
            embedding_size=EMB_SIZE,
            id_embedding_size=ID_EMB_SIZE,
            n_factors=args.n_factors,
            attention_size=ATTENTION_SIZE,
            kernel_sizes=KERNEL_SIZES,
            n_filters=N_FILTERS,
            dropout_rate=DROPOUT_RATE,
            max_text_length=MAX_TEXT_LENGTH,
            max_num_review=32,
            max_num_qa=32,
            batch_size=BATCH_SIZE,
            max_iter=args.epoch,
            model_selection=args.model_selection,
            optimizer="adam",
            learning_rate=args.learning_rate,
            init_params={"pretrained_word_embeddings": pretrained_word_embeddings},
            verbose=True,
            seed=123,
        )
    ],
    metrics=[
        cornac.metrics.MSE(),
    ],
)

exp.run()
