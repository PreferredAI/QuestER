from cornac.data import Dataset


class QuestERDataset(Dataset):
    def add_modalities(self, **kwargs):
        Dataset.add_modalities(self, kwargs=kwargs)
        self.review_and_item_qa_text = kwargs.get("review_and_item_qa_text", None)
