import os
import numpy as np
from tqdm.auto import trange



from cornac.metrics import MSE

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers, Input
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from cornac.utils import get_rng
from cornac.models import Recommender
from cornac.exception import ScoreException
from cornac.utils.init_utils import uniform
from cornac.eval_methods.base_method import rating_eval
from cornac.models.narre.narre import TextProcessor, AddGlobalBias


def get_item_qa(
    batch_iids, train_set, max_text_length, max_num_question=32, max_num_answer=16
):
    batch_item_questions, batch_item_num_questions = [], []
    batch_item_question_answers, batch_item_question_num_answers = [], []
    for idx in batch_iids:
        item_questions = []
        item_answers = []
        item_question_num_answers = []
        if idx in train_set.review_and_item_qa_text.item_qas:
            for inc, item_question_answers_ids in enumerate(
                train_set.review_and_item_qa_text.item_qas[idx]
            ):
                if max_num_question is not None and inc == max_num_question:
                    break
                item_question_answers_batch_seq = train_set.review_and_item_qa_text.batch_seq(
                    item_question_answers_ids[: 1 + max_num_answer],
                    max_length=max_text_length
                )  # keep the question and maximum number of answers acompanying with the given question

                item_questions.append(
                    item_question_answers_batch_seq[0]
                )
                if len(item_question_answers_ids) == 1: # question w/o answer, use question content as answer
                    item_answers.append(item_question_answers_batch_seq[:1])
                    item_question_num_answers.append(1)
                else:
                    item_answers.append(item_question_answers_batch_seq[1:])
                    item_question_num_answers.append(len(item_question_answers_batch_seq[1:]))
            item_answers = np.array([
                np.concatenate([item_answers[i], np.zeros((max_num_answer - item_answers[i].shape[0], item_answers[i].shape[1]))])
                for i in range(len(item_answers))
            ], dtype="int32")
            item_answers = np.concatenate([item_answers, np.zeros((max_num_question - item_answers.shape[0], item_answers.shape[1], item_answers.shape[2]))]).astype(np.int32)
        item_questions = pad_sequences(
            item_questions, padding="post", maxlen=max_text_length
        )
        batch_item_num_questions.append(len(item_questions))
        item_questions = np.concatenate([item_questions, np.zeros((max_num_question - item_questions.shape[0], item_questions.shape[1]))]).astype(np.int32)
        batch_item_questions.append(item_questions)
        batch_item_question_answers.append(item_answers)
        batch_item_question_num_answers.append(item_question_num_answers)
    batch_item_questions = pad_sequences(batch_item_questions, padding="post")
    if len(batch_item_questions.shape) == 2:
        batch_item_questions = batch_item_questions.reshape(
            batch_item_questions.shape[0], 0, max_text_length
        )
    batch_item_num_questions = np.array(batch_item_num_questions)
    batch_item_question_num_answers = pad_sequences(batch_item_question_num_answers, padding="post", maxlen=max_num_question)
    batch_item_question_answers = np.array(batch_item_question_answers, dtype=np.int32)
    return batch_item_questions, batch_item_num_questions, batch_item_question_answers, batch_item_question_num_answers


def get_review_data(
    batch_ids, train_set, max_text_length, by="user", max_num_review=32
):
    batch_reviews, batch_num_reviews = [], []
    review_group = (
        train_set.review_and_item_qa_text.user_review
        if by == "user"
        else train_set.review_and_item_qa_text.item_review
    )
    for idx in batch_ids:
        ids, review_ids = [], []
        for inc, (jdx, review_idx) in enumerate(review_group.get(idx, {}).items()):
            if max_num_review is not None and inc == max_num_review:
                break
            ids.append(jdx)
            review_ids.append(review_idx)
        reviews = train_set.review_and_item_qa_text.batch_seq(
            review_ids, max_length=max_text_length
        )
        batch_num_reviews.append(len(reviews))
        reviews = pad_sequences(reviews, padding="post", maxlen=max_text_length)
        reviews = np.concatenate([reviews, np.zeros((max_num_review - reviews.shape[0], reviews.shape[1]))]).astype(np.int32)
        batch_reviews.append(reviews)
    batch_reviews = pad_sequences(batch_reviews, padding="post")
    batch_ratings = (
        np.zeros((len(batch_ids), train_set.num_items), dtype=np.float32)
        if by == "user"
        else np.zeros((len(batch_ids), train_set.num_users), dtype=np.float32)
    )
    rating_group = train_set.user_data if by == "user" else train_set.item_data
    for batch_inc, idx in enumerate(batch_ids):
        jds, ratings = rating_group[idx]
        for jdx, rating in zip(jds, ratings):
            batch_ratings[batch_inc, jdx] = rating
    batch_num_reviews = np.array(batch_num_reviews, dtype=np.int32)
    return batch_reviews, batch_num_reviews, batch_ratings


class Model:
    def __init__(
        self,
        n_users,
        n_items,
        vocab,
        global_mean,
        n_factors=32,
        embedding_size=100,
        id_embedding_size=32,
        attention_size=16,
        kernel_sizes=[3],
        n_filters=64,
        n_user_mlp_factors=128,
        n_item_mlp_factors=128,
        dropout_rate=0.5,
        max_text_length=50,
        max_num_review=None,
        max_num_question=None,
        max_num_answer=None,
        pretrained_word_embeddings=None,
        temperature_parameter=1.0,
        verbose=False,
        seed=None,
    ):
        self.n_users = n_users
        self.n_items = n_items
        self.n_vocab = vocab.size
        self.global_mean = global_mean
        self.n_factors = n_factors
        self.embedding_size = embedding_size
        self.id_embedding_size = id_embedding_size
        self.attention_size = attention_size
        self.kernel_sizes = kernel_sizes
        self.n_filters = n_filters
        self.n_user_mlp_factors = n_user_mlp_factors
        self.n_item_mlp_factors = n_item_mlp_factors
        self.dropout_rate = dropout_rate
        self.max_text_length = max_text_length
        self.max_num_review = max_num_review
        self.max_num_question = max_num_question
        self.max_num_answer = max_num_answer
        self.verbose = verbose
        if seed is not None:
            self.rng = get_rng(seed)
            tf.random.set_seed(seed)

        embedding_matrix = uniform(
            shape=(self.n_vocab, self.embedding_size),
            low=-0.5,
            high=0.5,
            random_state=self.rng,
        )
        embedding_matrix[:4, :] = np.zeros((4, self.embedding_size))
        if pretrained_word_embeddings is not None:
            oov_count = 0
            for word, idx in vocab.tok2idx.items():
                embedding_vector = pretrained_word_embeddings.get(word)
                if embedding_vector is not None:
                    embedding_matrix[idx] = embedding_vector
                else:
                    oov_count += 1
            if self.verbose:
                print("Number of OOV words: %d" % oov_count)

        embedding_matrix = initializers.Constant(embedding_matrix)
        i_user_id = Input(shape=(1,), dtype="int32", name="input_user_id")
        i_item_id = Input(shape=(1,), dtype="int32", name="input_item_id")
        i_user_rating = Input(
            shape=(self.n_items), dtype="float32", name="input_user_rating"
        )
        i_item_rating = Input(
            shape=(self.n_users), dtype="float32", name="input_item_rating"
        )
        i_user_review = Input(
            shape=(None, self.max_text_length), dtype="int32", name="input_user_review"
        )
        i_item_review = Input(
            shape=(None, self.max_text_length), dtype="int32", name="input_item_review"
        )
        i_item_question = Input(
            shape=(None, self.max_text_length), dtype="int32", name="input_item_question"
        )
        i_item_question_answer = Input(
            shape=(None, None, self.max_text_length), dtype="int32", name="input_item_question_answer"
        )
        i_user_num_reviews = Input(
            shape=(1,), dtype="int32", name="input_user_number_of_review"
        )
        i_item_num_reviews = Input(
            shape=(1,), dtype="int32", name="input_item_number_of_review"
        )
        i_item_num_questions = Input(
            shape=(1,), dtype="int32", name="input_item_number_of_question"
        )
        i_item_question_num_answers = Input(
            shape=(None, 1,), dtype="int32", name="input_item_question_number_of_answer"
        )

        l_text_embedding = layers.Embedding(
            self.n_vocab,
            self.embedding_size,
            embeddings_initializer=embedding_matrix,
            mask_zero=True,
            name="layer_text_embedding",
        )
        l_user_embedding = layers.Embedding(
            self.n_users,
            self.id_embedding_size,
            embeddings_initializer="uniform",
            name="user_embedding",
        )
        l_item_embedding = layers.Embedding(
            self.n_items,
            self.id_embedding_size,
            embeddings_initializer="uniform",
            name="item_embedding",
        )

        user_bias = layers.Embedding(
            self.n_users,
            1,
            embeddings_initializer=tf.initializers.Constant(0.1),
            name="user_bias",
        )
        item_bias = layers.Embedding(
            self.n_items,
            1,
            embeddings_initializer=tf.initializers.Constant(0.1),
            name="item_bias",
        )

        user_text_processor = TextProcessor(
            self.max_text_length,
            filters=self.n_filters,
            kernel_sizes=self.kernel_sizes,
            dropout_rate=self.dropout_rate,
            name="user_text_processor",
        )
        item_text_processor = TextProcessor(
            self.max_text_length,
            filters=self.n_filters,
            kernel_sizes=self.kernel_sizes,
            dropout_rate=self.dropout_rate,
            name="item_text_processor",
        )
        item_question_text_processor = TextProcessor(
            self.max_text_length,
            filters=self.n_filters,
            kernel_sizes=self.kernel_sizes,
            dropout_rate=self.dropout_rate,
            name="item_question_text_processor",
        )
        item_question_answer_text_processor = TextProcessor(
            self.max_text_length,
            filters=self.n_filters,
            kernel_sizes=self.kernel_sizes,
            dropout_rate=self.dropout_rate,
            name="item_question_answer_text_processor",
        )
        user_review_h = user_text_processor(
            l_text_embedding(i_user_review), training=True
        )
        item_review_h = item_text_processor(
            l_text_embedding(i_item_review), training=True
        )
        item_question_h = item_question_text_processor(l_text_embedding(i_item_question), training=True)

        item_question_answer_h = item_question_answer_text_processor(l_text_embedding(tf.reshape(i_item_question_answer, shape=(-1, max_num_answer, max_text_length))), training=True)

        l_user_mlp = keras.models.Sequential(
            [
                layers.Dense(
                    self.n_user_mlp_factors, input_dim=self.n_items, activation="tanh"
                ),
                layers.Dense(self.n_user_mlp_factors // 2, activation="tanh"),
                layers.Dense(self.n_filters, activation="tanh"),
                layers.BatchNormalization(),
            ]
        )
        l_item_mlp = keras.models.Sequential(
            [
                layers.Dense(
                    self.n_item_mlp_factors, input_dim=self.n_users, activation="tanh"
                ),
                layers.Dense(self.n_item_mlp_factors // 2, activation="tanh"),
                layers.Dense(self.n_filters, activation="tanh"),
                layers.BatchNormalization(),
            ]
        )
        user_rating_h = l_user_mlp(i_user_rating)
        item_rating_h = l_item_mlp(i_item_rating)
        # mlp
        a_user = layers.Dense(1, activation=None, use_bias=True)(
            layers.Dense(self.attention_size, activation="tanh", use_bias=True)(
                tf.multiply(user_review_h, tf.expand_dims(user_rating_h, 1))
            )
        )
        a_user_masking = tf.expand_dims(
            tf.sequence_mask(
                tf.reshape(i_user_num_reviews, [-1]), maxlen=max_num_review
            ),
            -1,
        )
        user_attention = layers.Softmax(axis=1, name="user_attention")(
            a_user, a_user_masking
        )

        a_item_question_dense = layers.Dense(
            self.attention_size, activation="tanh", use_bias=True
        )(item_question_h)
        a_item_question_answer_dense = layers.Dense(
            self.attention_size, activation="tanh", use_bias=True
        )(item_question_answer_h)

        phi_jk = tf.expand_dims(a_item_question_dense, axis=2)
        psi_jkl = tf.reshape(a_item_question_answer_dense, shape=(-1, max_num_question, max_num_answer, a_item_question_answer_dense.shape[-1]))
        upsilon_jkl = layers.Dense(1, activation=None, use_bias=False, name="upsilon_jkl")(tf.multiply(psi_jkl, phi_jk) + psi_jkl)
        upsilon_jkl_mask = tf.cast(
            tf.expand_dims(
                tf.cast(
                    tf.reshape(
                        tf.sequence_mask(tf.reshape(i_item_question_num_answers, [-1]), maxlen=max_num_answer),
                        shape=(-1, max_num_question, max_num_answer)
                    ),
                    dtype=tf.int32,
                ) * tf.cast(
                    tf.expand_dims(
                        tf.sequence_mask(tf.reshape(i_item_num_questions, [-1]), maxlen=max_num_question),
                        axis=-1,
                    ),
                    dtype=tf.int32,
                ),
                axis=-1
            ),
            dtype=tf.bool
        )
        delta_jkl = layers.Softmax(axis=2, name="delta_jkl")(upsilon_jkl / temperature_parameter, upsilon_jkl_mask)
        omega_jk = tf.reshape(
            tf.reduce_sum(
                layers.Multiply()([tf.reshape(delta_jkl, shape=(-1, max_num_answer, 1)), item_question_answer_h]), axis=1
            ),
            shape=(-1, max_num_question, item_question_answer_h.shape[-1])
        )
        chi_jk = layers.Dense(self.attention_size, activation="tanh", use_bias=True, name="chi_jk")(omega_jk)

        a_item_review_dense = layers.Dense(
            self.attention_size, activation="tanh", use_bias=True
        )(tf.multiply(item_review_h, tf.expand_dims(item_rating_h, 1)))
        rho_ij = tf.expand_dims(a_item_review_dense, axis=2)
        eta_jkl = layers.Dense(1, activation=None, use_bias=False, name="eta_jkl")(
            tf.multiply(tf.expand_dims(chi_jk, axis=1), rho_ij) + rho_ij
        )
        eta_jkl_mask = tf.cast(
            tf.expand_dims(
                tf.cast(
                    tf.expand_dims(
                        tf.sequence_mask(
                            tf.reshape(i_item_num_reviews, [-1]), maxlen=max_num_review
                        ),
                        axis=-1,
                    ),
                    dtype=tf.int32,
                )
                * tf.cast(
                    tf.expand_dims(
                        tf.sequence_mask(
                            tf.reshape(i_item_num_questions, [-1]), maxlen=max_num_question
                        ),
                        axis=1,
                    ),
                    dtype=tf.int32,
                ),
                axis=-1,
            ),
            dtype=tf.bool,
        )

        beta_jkl = layers.Softmax(axis=2, name="beta_jkl")(eta_jkl / temperature_parameter, eta_jkl_mask)

        d_jk = tf.reduce_sum(
            layers.Multiply()([beta_jkl, tf.expand_dims(chi_jk, axis=1)]), axis=2
        )

        kappa_jk = layers.Dense(1, activation=None, use_bias=False, name="kappa_jk")(
            layers.Dense(self.attention_size, activation="tanh", use_bias=True)(d_jk)
        )

        gamma_jk = layers.Softmax(axis=1, name="gamma_jk")(
            kappa_jk / temperature_parameter,
            tf.expand_dims(
                tf.sequence_mask(
                    tf.reshape(i_item_num_reviews, [-1]), maxlen=max_num_review
                ),
                -1,
            ),
        )
        ou = layers.Dense(self.n_factors, use_bias=True, name="ou")(
            layers.Dropout(rate=self.dropout_rate, name="user_Oi")(
                tf.reduce_sum(layers.Multiply()([user_attention, user_review_h]), 1)
            )
        )
        oi = layers.Dense(self.n_factors, use_bias=True, name="oi")(
            layers.Dropout(rate=self.dropout_rate, name="item_Oi")(
                tf.reduce_sum(layers.Multiply()([gamma_jk, d_jk]), axis=1)
            )
        )

        pu = layers.Concatenate(axis=-1, name="pu")(
            [
                tf.expand_dims(user_rating_h, axis=1),
                tf.expand_dims(ou, axis=1),
                l_user_embedding(i_user_id),
            ]
        )

        qi = layers.Concatenate(axis=-1, name="qi")(
            [
                tf.expand_dims(item_rating_h, axis=1),
                tf.expand_dims(oi, axis=1),
                l_item_embedding(i_item_id),
            ]
        )

        W1 = layers.Dense(1, activation=None, use_bias=False, name="W1")
        add_global_bias = AddGlobalBias(init_value=self.global_mean, name="global_bias")
        r = layers.Add(name="prediction")(
            [W1(tf.multiply(pu, qi)), user_bias(i_user_id), item_bias(i_item_id)]
        )
        r = add_global_bias(r)
        self.graph = keras.Model(
            inputs=[
                i_user_id,
                i_item_id,
                i_user_rating,
                i_user_review,
                i_user_num_reviews,
                i_item_rating,
                i_item_review,
                i_item_num_reviews,
                i_item_question,
                i_item_num_questions,
                i_item_question_answer,
                i_item_question_num_answers
            ],
            outputs=r,
        )
        if self.verbose:
            self.graph.summary()


    def get_weights(self, train_set, batch_size=64):
        user_attention_review_pooling = keras.Model(
            inputs=[
                self.graph.get_layer("input_user_id").input,
                self.graph.get_layer("input_user_rating").input,
                self.graph.get_layer("input_user_review").input,
                self.graph.get_layer("input_user_number_of_review").input,
            ],
            outputs=self.graph.get_layer("pu").output,
        )
        item_attention_pooling = keras.Model(
            inputs=[
                self.graph.get_layer("input_item_id").input,
                self.graph.get_layer("input_item_rating").input,
                self.graph.get_layer("input_item_review").input,
                self.graph.get_layer("input_item_number_of_review").input,
                self.graph.get_layer("input_item_question").input,
                self.graph.get_layer("input_item_number_of_question").input,
                self.graph.get_layer("input_item_question_answer").input,
                self.graph.get_layer("input_item_question_number_of_answer").input
            ],
            outputs=[
                self.graph.get_layer("qi").output,
                self.graph.get_layer("delta_jkl").output,
                self.graph.get_layer("eta_jkl").output,
                self.graph.get_layer("beta_jkl").output,
                self.graph.get_layer("kappa_jk").output,
                self.graph.get_layer("gamma_jk").output,
            ],
        )
        P = np.zeros(
            (self.n_users, self.n_filters + self.n_factors + self.id_embedding_size),
            dtype=np.float32,
        )
        Q = np.zeros(
            (self.n_items, self.n_filters + self.n_factors + self.id_embedding_size),
            dtype=np.float32,
        )
        Xi = np.zeros((self.n_items, self.max_num_question, self.max_num_answer), dtype=np.float32)
        Eta = np.zeros(
            (self.n_items, self.max_num_review, self.max_num_question), dtype=np.float32
        )
        Beta = np.zeros(
            (self.n_items, self.max_num_review, self.max_num_question), dtype=np.float32
        )
        Kappa = np.zeros((self.n_items, self.max_num_review), dtype=np.float32)
        Gamma = np.zeros((self.n_items, self.max_num_review), dtype=np.float32)
        for batch_users in train_set.user_iter(batch_size):
            user_reviews, user_num_reviews, user_ratings = get_review_data(
                batch_users,
                train_set,
                self.max_text_length,
                by="user",
                max_num_review=self.max_num_review,
            )
            pu = user_attention_review_pooling(
                [batch_users, user_ratings, user_reviews, user_num_reviews],
                training=False,
            )
            P[batch_users] = pu.numpy().reshape(
                len(batch_users),
                self.n_filters + self.n_factors + self.id_embedding_size,
            )
        for batch_items in train_set.item_iter(batch_size):
            item_reviews, item_num_reviews, item_ratings = get_review_data(
                batch_items,
                train_set,
                self.max_text_length,
                by="item",
                max_num_review=self.max_num_review,
            )
            item_questions, item_num_questions, item_question_answers, item_question_num_answers  = get_item_qa(
                batch_items, train_set, self.max_text_length, max_num_question=self.max_num_question, max_num_answer=self.max_num_answer,
            )
            qi, xi_jk, eta_jl, beta_jl, kappa_j, gamma_j = item_attention_pooling(
                [
                    batch_items,
                    item_ratings,
                    item_reviews,
                    item_num_reviews,
                    item_questions,
                    item_num_questions,
                    item_question_answers,
                    item_question_num_answers
                ],
                training=False,
            )
            Xi[
                batch_items, : xi_jk.shape[1], : xi_jk.shape[2]
            ] = xi_jk.numpy().reshape(xi_jk.shape[:3])
            Eta[
                batch_items, : eta_jl.shape[1], : eta_jl.shape[2]
            ] = eta_jl.numpy().reshape(eta_jl.shape[:3])
            Beta[
                batch_items, : beta_jl.shape[1], : beta_jl.shape[2]
            ] = beta_jl.numpy().reshape(beta_jl.shape[:3])
            Kappa[batch_items, : kappa_j.shape[1]] = kappa_j.numpy().reshape(
                kappa_j.shape[:2]
            )
            Gamma[batch_items, : gamma_j.shape[1]] = gamma_j.numpy().reshape(
                gamma_j.shape[:2]
            )
            Q[batch_items] = qi.numpy().reshape(
                len(batch_items),
                self.n_filters + self.n_factors + self.id_embedding_size,
            )
        W1 = self.graph.get_layer("W1").get_weights()[0]
        bu = self.graph.get_layer("user_bias").get_weights()[0]
        bi = self.graph.get_layer("item_bias").get_weights()[0]
        mu = self.graph.get_layer("global_bias").get_weights()[0][0]
        return P, Q, W1, bu, bi, mu, Xi, Eta, Beta, Kappa, Gamma


class QuestER(Recommender):
    """

    Parameters
    ----------
    name: string, default: 'Question-Attentive Review-Level Recommendation Explanation'
        The name of the recommender model.

    embedding_size: int, default: 100
        Word embedding size

    n_factors: int, default: 8
        The dimension of the user/item's latent factors.

    attention_size: int, default: 8
        Attention size

    kernel_sizes: list, default: [3]
        List of kernel sizes of conv2d

    n_filters: int, default: 64
        Number of filters

    dropout_rate: float, default: 0.5
        Dropout rate of neural network dense layers

    max_text_length: int, default: 128
        Maximum number of tokens in a review instance

    max_num_review: int, default: None
        Maximum number of reviews that you want to feed into training. By default, the model will be trained with all reviews.

    batch_size: int, default: 64
        Batch size

    max_iter: int, default: 10
        Max number of training epochs

    optimizer: string, optional, default: 'adam'
        Optimizer for training is either 'adam' or 'rmsprop'.

    learning_rate: float, optional, default: 0.001
        Initial value of learning rate for the optimizer.

    trainable: boolean, optional, default: True
        When False, the model will not be re-trained, and input of pre-trained parameters are required.

    verbose: boolean, optional, default: True
        When True, running logs are displayed.

    init_params: dictionary, optional, default: None
        Initial parameters, pretrained_word_embeddings could be initialized here, e.g., init_params={'pretrained_word_embeddings': pretrained_word_embeddings}

    seed: int, optional, default: None
        Random seed for weight initialization.
        If specified, training will take longer because of single-thread (no parallelization).

    References
    ----------

    """

    def __init__(
        self,
        name="QuestER",
        embedding_size=100,
        id_embedding_size=8,
        n_factors=8,
        attention_size=8,
        kernel_sizes=[3],
        n_filters=64,
        dropout_rate=0.5,
        max_text_length=128,
        max_num_review=32,
        max_num_question=32,
        max_num_answer=16,
        batch_size=64,
        max_iter=10,
        optimizer="adam",
        learning_rate=0.001,
        temperature_parameter=1e-2,
        model_selection="last",  # last or best
        user_based=True,
        trainable=True,
        verbose=True,
        init_params=None,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.seed = seed
        self.embedding_size = embedding_size
        self.id_embedding_size = id_embedding_size
        self.n_factors = n_factors
        self.attention_size = attention_size
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes
        self.dropout_rate = dropout_rate
        self.max_text_length = max_text_length
        self.max_num_review = max_num_review
        self.max_num_question = max_num_question
        self.max_num_answer = max_num_answer
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.temperature_parameter = temperature_parameter
        self.model_selection = model_selection
        self.user_based = user_based
        # Init params if provided
        self.init_params = {} if init_params is None else init_params
        self.losses = {"train_losses": [], "val_losses": []}

    def fit(self, train_set, val_set=None):
        """Fit the model to observations.

        Parameters
        ----------
        train_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as additional modalities.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            User-Item preference data for model selection purposes (e.g., early stopping).

        Returns
        -------
        self : object
        """
        Recommender.fit(self, train_set, val_set)

        if self.trainable:
            if not hasattr(self, "model"):
                self.model = Model(
                    self.train_set.num_users,
                    self.train_set.num_items,
                    self.train_set.review_and_item_qa_text.vocab,
                    self.train_set.global_mean,
                    n_factors=self.n_factors,
                    embedding_size=self.embedding_size,
                    id_embedding_size=self.id_embedding_size,
                    attention_size=self.attention_size,
                    kernel_sizes=self.kernel_sizes,
                    n_filters=self.n_filters,
                    dropout_rate=self.dropout_rate,
                    max_text_length=self.max_text_length,
                    max_num_review=self.max_num_review,
                    max_num_question=self.max_num_question,
                    max_num_answer=self.max_num_answer,
                    pretrained_word_embeddings=self.init_params.get(
                        "pretrained_word_embeddings"
                    ),
                    temperature_parameter=self.temperature_parameter,
                    verbose=self.verbose,
                    seed=self.seed,
                )
            self._fit()

        return self

    def _fit(self):
        loss = keras.losses.MeanSquaredError()
        if self.optimizer == "rmsprop":
            optimizer_ = keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        else:
            optimizer_ = keras.optimizers.Adam(learning_rate=self.learning_rate)
        train_loss = keras.metrics.Mean(name="loss")
        val_loss = 1e9
        best_val_loss = 1e9
        self.best_epoch = None
        loop = trange(
            self.max_iter,
            disable=not self.verbose,
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        )
        for i_epoch, _ in enumerate(loop):
            train_loss.reset_states()
            for i, (batch_users, batch_items, batch_ratings) in enumerate(
                self.train_set.uir_iter(self.batch_size, shuffle=True)
            ):
                user_reviews, user_num_reviews, user_ratings = get_review_data(
                    batch_users,
                    self.train_set,
                    self.max_text_length,
                    by="user",
                    max_num_review=self.max_num_review,
                )
                item_reviews, item_num_reviews, item_ratings = get_review_data(
                    batch_items,
                    self.train_set,
                    self.max_text_length,
                    by="item",
                    max_num_review=self.max_num_review,
                )
                item_questions, item_num_questions, item_question_answers, item_question_num_answers = get_item_qa(
                    batch_items,
                    self.train_set,
                    self.max_text_length,
                    max_num_question=self.max_num_question,
                    max_num_answer=self.max_num_answer,
                )
                with tf.GradientTape() as tape:
                    predictions = self.model.graph(
                        [
                            batch_users,
                            batch_items,
                            user_ratings,
                            user_reviews,
                            user_num_reviews,
                            item_ratings,
                            item_reviews,
                            item_num_reviews,
                            item_questions,
                            item_num_questions,
                            item_question_answers,
                            item_question_num_answers,
                        ],
                        training=True,
                    )
                    _loss = loss(batch_ratings, predictions)
                gradients = tape.gradient(_loss, self.model.graph.trainable_variables)
                optimizer_.apply_gradients(
                    zip(gradients, self.model.graph.trainable_variables)
                )
                train_loss(_loss)
                if i % 10 == 0:
                    loop.set_postfix(
                        loss=train_loss.result().numpy(),
                        val_loss=val_loss,
                        best_val_loss=best_val_loss,
                        best_epoch=self.best_epoch,
                    )
            current_weights = self.model.get_weights(self.train_set, self.batch_size)
            if self.val_set is not None:
                (
                    self.P,
                    self.Q,
                    self.W1,
                    self.bu,
                    self.bi,
                    self.mu,
                    self.Xi,
                    self.Eta,
                    self.Beta,
                    self.Kappa,
                    self.Gamma,
                ) = current_weights
                [current_val_mse], _ = rating_eval(
                    model=self,
                    metrics=[MSE()],
                    test_set=self.val_set,
                    user_based=self.user_based,
                )
                val_loss = current_val_mse
                if best_val_loss > val_loss:
                    best_val_loss = val_loss
                    self.best_epoch = i_epoch + 1
                    best_weights = current_weights
                loop.set_postfix(
                    loss=train_loss.result().numpy(),
                    val_loss=val_loss,
                    best_val_loss=best_val_loss,
                    best_epoch=self.best_epoch,
                )
            self.losses["train_losses"].append(train_loss.result().numpy())
            self.losses["val_losses"].append(val_loss)
        loop.close()

        # save weights for predictions
        (
            self.P,
            self.Q,
            self.W1,
            self.bu,
            self.bi,
            self.mu,
            self.Xi,
            self.Eta,
            self.Beta,
            self.Kappa,
            self.Gamma,
        ) = (
            best_weights
            if self.val_set is not None and self.model_selection == "best"
            else current_weights
        )
        if self.verbose:
            print("Learning completed!")

    def save(self, save_dir=None):
        """Save a recommender model to the filesystem.

        Parameters
        ----------
        save_dir: str, default: None
            Path to a directory for the model to be stored.

        """
        if save_dir is None:
            return
        model = self.model
        del self.model

        model_file = Recommender.save(self, save_dir)

        self.model = model
        self.model.graph.save(model_file.replace(".pkl", ".cpt"))

        return model_file

    @staticmethod
    def load(model_path, trainable=False):
        """Load a recommender model from the filesystem.

        Parameters
        ----------
        model_path: str, required
            Path to a file or directory where the model is stored. If a directory is
            provided, the latest model will be loaded.

        trainable: boolean, optional, default: False
            Set it to True if you would like to finetune the model. By default,
            the model parameters are assumed to be fixed after being loaded.

        Returns
        -------
        self : object
        """
        model = Recommender.load(model_path, trainable)
        model.model.graph = keras.models.load_model(
            model.load_from.replace(".pkl", ".cpt")
        )

        return model

    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform score prediction.

        item_idx: int, optional, default: None
            The index of the item for that to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items
        """
        if item_idx is None:
            if self.is_unknown_user(user_idx):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d)" % user_idx
                )
            h0 = self.P[user_idx] * self.Q
            known_item_scores = h0.dot(self.W1) + self.bu[user_idx] + self.bi + self.mu
            return known_item_scores.ravel()
        else:
            if self.is_unknown_user(user_idx) or self.is_unknown_item(
                item_idx
            ):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )
            known_item_score = (
                (self.P[user_idx] * self.Q[item_idx]).dot(self.W1)
                + self.bu[user_idx]
                + self.bi[item_idx]
                + self.mu
            )
            return known_item_score
