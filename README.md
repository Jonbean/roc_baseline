### 1) Train the BiLSTM with attention on 80% of val data.

```bash
python2 train_on_val.py
```

### 2) Train end-only model.

```bash
# test on romeojuliet 
python2 plain_rnn_end.py 300 50 1
# change model settings by specifying properties in the init function of Hierachi_RNN class
```
