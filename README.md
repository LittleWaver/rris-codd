# Robust Referring Image Segmentation for Construction and Demolition Waste Recognition

## Abstract

Referring Image Segmentation (RIS) offers a promising paradigm for precise, language-guided object identification in complex environments like Construction and Demolition Waste (CDW) sorting sites. However, existing RIS models exhibit a critical vulnerability: an inherent "blind trust" in textual prompts, leading to unreliable performance when faced with the ambiguous or erroneous descriptions common in real-world scenarios. To address this gap, this work introduces a robust RIS framework tailored for the CDW domain. We propose RefSegformer-CDW, a novel architecture that integrates a streamlined vision-language fusion mechanism to enhance both segmentation accuracy and stability under challenging conditions. To facilitate the development and rigorous evaluation of such robust models, we constructed Ref-CODD, a new dataset featuring not only paired image-text annotations but also a comprehensive set of adversarial negative samples. Extensive experiments demonstrate the superiority of our approach. RefSegformer-CDW significantly outperforms state-of-the-art baselines, particularly on difficult-to-segment material categories and under varying illumination, showcasing its enhanced reliability and practical viability for automated CDW recognition.

## Installation

1.   Clone the project.

``` python
git clone https://github.com/LittleWaver/rris-codd.git

cd rris-codd
```

2.   Create the Conda Environment and install the requirements.

``` python
conda env create -f environment.yml
```

3.   Activate the environment.

``` python
conda activate rris-codd
```

## Data Preparation

1.   CODD dataset.

     1.   download the files, link is: [Construction and Demolition Waste Object Detection Dataset (CODD)](https://data.mendeley.com/datasets/wds85kt64j/3).
     2.   put files as follows:

     ``` shell
     rris-codd
     └── raw
         └── image
             └── CODD
                 ├── testing
                 │   ├── testing_1.jpg
                 │   ├── testing_1.xml
                 │   ├── ...
                 │   └── ...
                 ├── training
                 │   ├── training_1.jpg
                 │   ├── trainning_1.xml
                 │   ├── ...
                 │   └── ...
                 └── validation
                     ├── validation_1.jpg
                     ├── validation_1.xml
                     ├── ...
                     └── ...
     ```

2.   Ref-CODD dataset

     the Ref-CODD dataset is already in the Github project, the files as follows:

     ``` shell
     rris-codd
     └── raw
         └── refs
             └── CODD
                 ├── testing
             	│   ├── instances.json
             	│   └── refs.p
                 ├── training
                 │   ├── instances.json
                 │   └── refs.p
                 └── validation
                     ├── instances.json
                     └── refs.p
     ```

3.   pretrained BERT.

     1.   download the files, link is: [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased).
     2.   put files as follows:

     ``` shell
     rris-codd
     └── raw
         └── pretrained
     		└── bert-base-uncased
                 ├── bpe_simple_vocab_16e6.txt.gz
                 ├── config.json
                 ├── pytorch_model.bin
                 └── vocab.txt
     ```

4.   pretrained Swin Transformer.

     1.   download the files, link is: [microsoft/swin-base-patch4-window12-384](https://huggingface.co/microsoft/swin-base-patch4-window12-384)
     2.   put files as follows:

     ``` shell
     rris-codd
     └── raw
         	├── pretrained
     				└── swin-base
     					   └── swin_base_patch4_window12_384_22k.pth
     ```


## Setup

Before you run the code, you should setup the project parameters according to your need in `config.py` as follows:

``` python
# Path
...

# Dataset
...

# Train
...

# Validation
...

# Ablation Study
...

# RRIS
...

# Language Encoder, BERT
...

# Vision Encoder: Swin Transformer
...
```

## Training

``` shell
# train
python -m src.main --mode train
```

## Evaluation

``` shell
# validation
nohup python -m src.main --mode validate

# illumination 
nohup python -m src.main --mode eval_illumination

# class
nohup python -m src.main --mode eval_class
```

## Acknowledgement

Great thanks to your work: [robust-ref-seg](https://github.com/jianzongwu/robust-ref-seg).

## License

This project is licensed under The MIT License.