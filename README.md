# Question-Attentive Review-Level Explanation for Neural Rating Regression

This repo contain source code of the paper:

**[Question-Attentive Review-Level Explanation for Neural Rating Regression](https://lthoang.com/assets/publications/tist24.pdf)**
<br>
[Trung-Hoang Le](http://lthoang.com/) and [Hady W. Lauw](http://www.hadylauw.com/)
<br>

A conference version of this paper was presented at [BigData 2022](https://bigdataieee.org/BigData2022/).

Data: https://static.preferred.ai/datasets/quester-data.zip

If you use QuestER in a scientific publication, we would appreciate citations to the following papers:

```
@article{le2024question,
author = {Le, Trung-Hoang and Lauw, Hady W.},
title = {Question-Attentive Review-Level Explanation for Neural Rating Regression},
year = {2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
issn = {2157-6904},
url = {https://doi.org/10.1145/3699516},
doi = {10.1145/3699516},
journal = {ACM Trans. Intell. Syst. Technol.},
month = oct,
keywords = {neural rating regression, recommendation explanation, review-level explanation, question-level explanation}
}
```

```
@inproceedings{le2022question,
  title     = {Question-Attentive Review-Level Recommendation Explanation},
  author    = {Le, Trung-Hoang and Lauw, Hady W.},
  booktitle = {2022 IEEE International Conference on Big Data (Big Data)},
  year      = {2022},
  organization={IEEE}
}
```

## How to run

### Pretrained embeddings

We used Glove 6B tokens, 400K vocab, 100d vectors as pretrained word embeddings, which can be found at [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/).


```bash
conda create --name quester python==3.9
conda activate quester
pip install -r requirements.txt
```

### Run QuestER experiment

```bash
CUDA_VISIBLE_DEVICES=0 python exp.py -i data/musical
```

## Contact
Questions and discussion are welcome: [lthoang.com](http://lthoang.com)
