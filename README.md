# SMART
A pytorch implementation of [SMART](https://arxiv.org/abs/1911.03437).

## Requirements
- python 3.7
- pytorch
- tqdm
- transformers
- sklearn

## Performence
Model | CoLA(MCC) | QNLI(ACC) | SST-2(ACC) | MNLI(ACC) | QQP(ACC) | MRPC(ACC) | QNLI(ACC) |
---- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
BERT-Base | 57.0 | 91.1 | **93.2** | **80.8** | **87.3** | 85.2 | 91.1 | 
SMART-Base |  **58.7** | **91.4** | 93.0 | 79.0 | 86.4 | **86.1** | **91.3** |

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
