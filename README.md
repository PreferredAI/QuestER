# Question-Attentive Review-Level Recommendation Explanation

This is the code for the paper:

**[Question-Attentive Review-Level Recommendation Explanation](https://lthoang.com/assets/publications/bigdata22.pdf)**
<br>
[Trung-Hoang Le](http://lthoang.com/) and [Hady W. Lauw](http://www.hadylauw.com/)
<br>
Presented at [BigData 2022](https://bigdataieee.org/BigData2022/)


If you find the code and [data](https://drive.google.com/drive/folders/10HVkH-cY8_GEfrMarCse-WSHelnNBi2s?usp=share_link) useful in your research, please cite:

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
pip install -r requirements.txt
```

### Run QuestER experiment


```bash
CUDA_VISIBLE_DEVICES=0 python exp.py -i data/musical
```

## Contact
Questions and discussion are welcome: [lthoang.com](http://lthoang.com)
