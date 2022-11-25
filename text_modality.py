from cornac.data import TextModality
from cornac.data.text import Tokenizer, Vocabulary
from typing import List, Dict, Callable, Union
from collections import OrderedDict


class ReviewAndItemQAModality(TextModality):
    def __init__(
        self,
        data: List[tuple] = None,
        qa_data: List[tuple] = None,
        tokenizer: Tokenizer = None,
        vocab: Vocabulary = None,
        max_vocab: int = None,
        max_doc_freq: Union[float, int] = 1.0,
        min_doc_freq: int = 1,
        tfidf_params: Dict = None,
        **kwargs
    ):
        super().__init__(
            tokenizer=tokenizer,
            vocab=vocab,
            max_vocab=max_vocab,
            max_doc_freq=max_doc_freq,
            min_doc_freq=min_doc_freq,
            tfidf_params=tfidf_params,
            **kwargs
        )
        self.raw_data = data
        self.raw_qa_data = qa_data

    def _build_corpus(self, uid_map, iid_map, dok_matrix):
        id_map = None
        corpus = []
        self.user_review = OrderedDict()
        self.item_review = OrderedDict()
        reviews = OrderedDict()
        for raw_uid, raw_iid, review in self.raw_data:
            user_idx = uid_map.get(raw_uid, None)
            item_idx = iid_map.get(raw_iid, None)
            if (
                user_idx is None
                or item_idx is None
                or dok_matrix[user_idx, item_idx] == 0
            ):
                continue
            idx = len(reviews)
            reviews.setdefault(idx, review)
            user_dict = self.user_review.setdefault(user_idx, OrderedDict())
            user_dict[item_idx] = idx
            item_dict = self.item_review.setdefault(item_idx, OrderedDict())
            item_dict[user_idx] = idx
            corpus.append(review)
        self.reviews = reviews

        self.item_qas = OrderedDict()
        qas = OrderedDict()
        qa_idx_offset = len(reviews)
        for raw_iid, questions in self.raw_qa_data:
            idx = iid_map.get(raw_iid, None)
            if idx is None:
                continue
            item_qas_list = self.item_qas.setdefault(idx, [])
            for question_answers in questions:
                t_qas = []
                q_idx = qa_idx_offset + len(qas)
                question = question_answers[0]
                corpus.append(question)
                qas.setdefault(q_idx, question)
                t_qas.append(q_idx)
                answers = question_answers[1:]
                for answer in answers:
                    a_idx = qa_idx_offset + len(qas)
                    corpus.append(answer)
                    qas.setdefault(a_idx, answer)
                    t_qas.append(a_idx)
                item_qas_list.append(t_qas)
        self.qas = qas

        return corpus, id_map

    def build(self, uid_map=None, iid_map=None, dok_matrix=None, **kwargs):
        """Build the model based on provided list of ordered ids"""
        if uid_map is None or iid_map is None or dok_matrix is None:
            raise ValueError("uid_map, iid_map, and dok_matrix are required")
        self.corpus, id_map = self._build_corpus(uid_map, iid_map, dok_matrix)
        super().build(id_map=id_map)

        return self


class ReviewSentenceAndItemQAModality(TextModality):
    def __init__(
        self,
        data: List[tuple] = None,
        qa_data: List[tuple] = None,
        tokenizer: Tokenizer = None,
        vocab: Vocabulary = None,
        max_vocab: int = None,
        max_doc_freq: Union[float, int] = 1.0,
        min_doc_freq: int = 1,
        tfidf_params: Dict = None,
        **kwargs
    ):
        super().__init__(
            tokenizer=tokenizer,
            vocab=vocab,
            max_vocab=max_vocab,
            max_doc_freq=max_doc_freq,
            min_doc_freq=min_doc_freq,
            tfidf_params=tfidf_params,
            **kwargs
        )
        self.raw_data = data
        self.raw_qa_data = qa_data

    def _build_corpus(self, uid_map, iid_map, dok_matrix):
        from nltk.tokenize import sent_tokenize

        id_map = None
        corpus = []
        self.user_review_sentence = OrderedDict()
        self.item_review_sentence = OrderedDict()
        sentences = OrderedDict()
        for raw_uid, raw_iid, review in self.raw_data:
            user_idx = uid_map.get(raw_uid, None)
            item_idx = iid_map.get(raw_iid, None)
            if (
                user_idx is None
                or item_idx is None
                or dok_matrix[user_idx, item_idx] == 0
            ):
                continue
            user_item_dict = self.user_review_sentence.setdefault(
                user_idx, OrderedDict()
            )
            user_item_sentence_ids = user_item_dict.setdefault(item_idx, [])
            item_user_dict = self.item_review_sentence.setdefault(
                item_idx, OrderedDict()
            )
            item_user_sentence_ids = item_user_dict.setdefault(user_idx, [])
            for sentence in sent_tokenize(review):
                idx = len(sentences)
                sentences.setdefault(idx, sentence)
                user_item_sentence_ids.append(idx)
                item_user_sentence_ids.append(idx)
                corpus.append(sentence)
        self.sentences = sentences

        self.item_qas = OrderedDict()
        qas = OrderedDict()
        qa_idx_offset = len(sentences)
        for raw_iid, questions in self.raw_qa_data:
            idx = iid_map.get(raw_iid, None)
            if idx is None:
                continue
            item_qas_list = self.item_qas.setdefault(idx, [])
            for question_answers in questions:
                t_qas = []
                q_idx = qa_idx_offset + len(qas)
                question = question_answers[0]
                corpus.append(question)
                qas.setdefault(q_idx, question)
                t_qas.append(q_idx)
                answers = question_answers[1:]
                for answer in answers:
                    a_idx = len(qas)
                    corpus.append(answer)
                    qas.setdefault(a_idx, answer)
                    t_qas.append(a_idx)
                item_qas_list.append(t_qas)
        self.qas = qas

        return corpus, id_map

    def build(self, uid_map=None, iid_map=None, dok_matrix=None, **kwargs):
        """Build the model based on provided list of ordered ids"""
        if uid_map is None or iid_map is None or dok_matrix is None:
            raise ValueError("uid_map, iid_map, and dok_matrix are required")
        self.corpus, id_map = self._build_corpus(uid_map, iid_map, dok_matrix)
        super().build(id_map=id_map)

        return self
