# SDP-LSTM model for TACRED classification

#### Requirements

- python 2.7
- Tensorflow 1.8.0

#### Input files:

- ./data_tacred/dependency/train.deppath.conll
- ./data_tacred/dependency/test.deppath.conll
- ./data_tacred/dependency/dev.deppath.conll

#### Preprocess:

`python2 data_utils.py tacred`

#### Run model:

`python2 train.py tacred`