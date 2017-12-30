### 0) requirement
tensorflow 1.1.0	theano 0.9.0	Lasagne 0.2.dev1    python 2.7.10
### 1) Train the BiLSTM with attention on 80% of val data.

```bash
python2 train_on_val.py
```

### 2) Train end-only model

```bash 
python2 plain_rnn_end.py 300 50 1
# change model settings by specifying properties in the init function of Hierachi_RNN class
```

### 3) Citation
if you use our code or model please cite our work
```
@inproceedings{DBLP:conf/acl/CaiTG17,
  author    = {Zheng Cai and
               Lifu Tu and
               Kevin Gimpel},
  title     = {Pay Attention to the Ending: Strong Neural Baselines for the {ROC}
               Story Cloze Task},
  booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational
               Linguistics, {ACL} 2017, Vancouver, Canada, July 30 - August 4, Volume
               2: Short Papers},
  pages     = {616--622},
  year      = {2017},
  crossref  = {DBLP:conf/acl/2017-2},
  url       = {https://doi.org/10.18653/v1/P17-2097},
  doi       = {10.18653/v1/P17-2097},
  timestamp = {Fri, 04 Aug 2017 16:38:24 +0200},
  biburl    = {http://dblp.org/rec/bib/conf/acl/CaiTG17},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}
```