# Sentence-level Models

#### Requirements

- Python 3.6.4
- Pytorch 0.4.0

#### Input files

- ./data/json/train.json
- ./data/json/test.json
- ./data/json/dev.json

#### Conver Format

`python3 tacred2json.py` (Don't need if provided json file)

#### Prepare Vocabulary

- Put `glove.840B.300d.txt` file in `./data/glove` directory

`python3 vocab.py`

#### Running

`python3 train.py --model <model_name> --log <log_name>`

- model_name can be "pa_lstm" or "bgru"

### BGRU

Bidirectional GRU

### Position-Aware LSTM

Pytorch implementation of Position-Aware LSTM for relation extraction

Reference: https://nlp.stanford.edu/pubs/zhang2017tacred.pdf