from cornac.eval_methods import StratifiedSplit
from dataset import QuestERDataset


class QuestERStratifiedSplit(StratifiedSplit):
    def __init__(
        self,
        data=None,
        fmt="UIR",
        rating_threshold=1.0,
        seed=None,
        exclude_unknowns=True,
        verbose=False,
        **kwargs
    ):
        self.review_and_item_qa_text = kwargs.get("review_and_item_qa_text", None)
        super().__init__(
            data=data,
            fmt=fmt,
            rating_threshold=rating_threshold,
            seed=seed,
            exclude_unknowns=exclude_unknowns,
            verbose=verbose,
            **kwargs
        )

    def build(self, train_data, test_data, val_data=None):
        if train_data is None or len(train_data) == 0:
            raise ValueError("train_data is required but None or empty!")
        if test_data is None or len(test_data) == 0:
            raise ValueError("test_data is required but None or empty!")

        self.global_uid_map.clear()
        self.global_iid_map.clear()

        self._build_datasets(train_data, test_data, val_data)
        self._build_modalities()

        return self

    def _build_datasets(self, train_data, test_data, val_data=None):
        self.train_set = QuestERDataset.build(
            data=train_data,
            fmt=self.fmt,
            global_uid_map=self.global_uid_map,
            global_iid_map=self.global_iid_map,
            seed=self.seed,
            exclude_unknowns=False,
        )
        if self.verbose:
            print("---")
            print("Training data:")
            print("Number of users = {}".format(self.train_set.num_users))
            print("Number of items = {}".format(self.train_set.num_items))
            print("Number of ratings = {}".format(self.train_set.num_ratings))
            print("Max rating = {:.1f}".format(self.train_set.max_rating))
            print("Min rating = {:.1f}".format(self.train_set.min_rating))
            print("Global mean = {:.1f}".format(self.train_set.global_mean))

        self.test_set = QuestERDataset.build(
            data=test_data,
            fmt=self.fmt,
            global_uid_map=self.global_uid_map,
            global_iid_map=self.global_iid_map,
            seed=self.seed,
            exclude_unknowns=self.exclude_unknowns,
        )
        if self.verbose:
            print("---")
            print("Test data:")
            print("Number of users = {}".format(len(self.test_set.uid_map)))
            print("Number of items = {}".format(len(self.test_set.iid_map)))
            print("Number of ratings = {}".format(self.test_set.num_ratings))
            print(
                "Number of unknown users = {}".format(
                    self.test_set.num_users - self.train_set.num_users
                )
            )
            print(
                "Number of unknown items = {}".format(
                    self.test_set.num_items - self.train_set.num_items
                )
            )

        if val_data is not None and len(val_data) > 0:
            self.val_set = QuestERDataset.build(
                data=val_data,
                fmt=self.fmt,
                global_uid_map=self.global_uid_map,
                global_iid_map=self.global_iid_map,
                seed=self.seed,
                exclude_unknowns=self.exclude_unknowns,
            )
            if self.verbose:
                print("---")
                print("Validation data:")
                print("Number of users = {}".format(len(self.val_set.uid_map)))
                print("Number of items = {}".format(len(self.val_set.iid_map)))
                print("Number of ratings = {}".format(self.val_set.num_ratings))

        if self.verbose:
            print("---")
            print("Total users = {}".format(self.total_users))
            print("Total items = {}".format(self.total_items))

        self.train_set.total_users = self.total_users
        self.train_set.total_items = self.total_items

    def _build_modalities(self):
        for user_modality in [
            self.user_feature,
            self.user_text,
            self.user_image,
            self.user_graph,
        ]:
            if user_modality is None:
                continue
            user_modality.build(
                id_map=self.global_uid_map,
                uid_map=self.train_set.uid_map,
                iid_map=self.train_set.iid_map,
                dok_matrix=self.train_set.dok_matrix,
            )

        for item_modality in [
            self.item_feature,
            self.item_text,
            self.item_image,
            self.item_graph,
        ]:
            if item_modality is None:
                continue
            item_modality.build(
                id_map=self.global_iid_map,
                uid_map=self.train_set.uid_map,
                iid_map=self.train_set.iid_map,
                dok_matrix=self.train_set.dok_matrix,
            )

        for modality in [
            self.sentiment,
            self.review_text,
            self.review_and_item_qa_text,
        ]:
            if modality is None:
                continue
            modality.build(
                uid_map=self.train_set.uid_map,
                iid_map=self.train_set.iid_map,
                dok_matrix=self.train_set.dok_matrix,
            )

        for data_set in [self.train_set, self.test_set, self.val_set]:
            if data_set is None:
                continue
            data_set.add_modalities(
                user_feature=self.user_feature,
                user_text=self.user_text,
                user_image=self.user_image,
                user_graph=self.user_graph,
                item_feature=self.item_feature,
                item_text=self.item_text,
                item_image=self.item_image,
                item_graph=self.item_graph,
                sentiment=self.sentiment,
                review_text=self.review_text,
                review_and_item_qa_text=self.review_and_item_qa_text,
            )
