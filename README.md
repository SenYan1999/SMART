# SMART
A pytorch implementation of [SMART](https://arxiv.org/abs/1911.03437).

## Requirements
- python 3.7
- pytorch
- tqdm
- transformers
- sklearn

## Performence
Model | CoLA(MCC)
---- | :---:
BERT-Base | 57.0
SMART-Base |  58.7

## Usage
Download and prepare data.(To be implemented, download GLUE to glue_data/)

Prepare dataset
```bash
python main.py --do_prepare
```

Train model with **[SMART](https://arxiv.org/abs/1911.03437)**
```bash
python main.py --do_train --num_epoch 12 --batch_size 32 --task cola
```

Train model with **normal bert fine-tuning**
```bash
python main.py --do_train --normal --num_epoch 12 --batch_size 32 --task cola
```

## TODO
- [ ] Add a script to download glue dataset and extract them to glue_data/.
- [ ] Prepare more experiment results.
