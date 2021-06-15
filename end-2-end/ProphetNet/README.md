# ProphetNet

This repo provides the code for reproducing the experiments in [*ProphetNet*](https://arxiv.org/pdf/2001.04063). In the paper, the authors propose a new pre-trained language model called ProphetNet for sequence-to-sequence learning with a novel self-supervised objective called future n-gram prediction. 


## Dependency
- pip install torch==1.3.0  
- pip install fairseq==v0.9.0  
- pip install tensorboardX==1.7    

## Pre-trained Models

Recommended Checkpoint:
- **ProphetNet-En** [[link]](https://msraprophetnet.blob.core.windows.net/prophetnet/release_checkpoints/prophetnet_en.pt)

Expired Checkpoints:
- **ProphetNet-En-16GB** [[link]](https://msraprophetnet.blob.core.windows.net/prophetnet/release_checkpoints/prophetnet_en_16g.pt)

ProphetNet-En and ProphetNet-En-16GB have the same model size. Difference is the pretraining corpus. We recommend the 160GB one.   
ProphetNet-En is pre-trained with 160GB English raw texts, including Wikipedia, books, stories, news, and web texts. The vocabulary of ProphetNet-En is the same as BERT sub-words vocabulary. The vocabulary is based on bpe subwords with a max length matching algorithm. Its vocabulary size is 30,522.  
ProphetNet-En-16GB is pretrained with 16GB English raw texts, including Wikipedia and BookCorpus.

## MUHACU
The preprocessed data is provided as `tokenized ` folder in `forecasting` and `plannning` folders for each task.

### Data Preprocess
Please note that we have provided the `processed` binary data in `forecasting` and `plannning` folders.

#### Human Action Forecasting
``` bash
fairseq-preprocess \
--user-dir ./prophetnet \
--task translation_prophetnet \
--source-lang src --target-lang tgt \
--trainpref forecasting/tokenized/train --validpref forecasting/tokenized/valid --testpref forecasting/tokenized/test \
--destdir forecasting/processed --srcdict ./vocab.txt --tgtdict ./vocab.txt \
--workers 20
```

#### Human Action Planning
``` bash
fairseq-preprocess \
--user-dir ./prophetnet \
--task translation_prophetnet \
--source-lang src --target-lang tgt \
--trainpref planning/tokenized/train --validpref planning/tokenized/valid --testpref planning/tokenized/test \
--destdir planning/processed --srcdict ./vocab.txt --tgtdict ./vocab.txt \
--workers 20
```

### Fine-tune
We fine-tuned the model on 2 * NVIDIA V100 (16GB) GPUs.To finetune the model on MUHACU, please run:

#### Human Action Forecasting
```
DATA_DIR=forecasting/processed
USER_DIR=./prophetnet
ARCH=ngram_transformer_prophet_large
CRITERION=ngram_language_loss
SAVE_DIR=forecasting/finetune_forecasting_checkpoints
TENSORBOARD_LOGDIR=forecasting/finetune_forecasting_tensorboard
PRETRAINED_MODEL=pretrained_checkpoints/prophetnet_en.pt

fairseq-train \
--fp16 \
--user-dir $USER_DIR --task translation_prophetnet --arch $ARCH \
--optimizer adam --adam-betas '(0.9, 0.999)' --clip-norm 0.1 \
--lr 0.0001 \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 1000 \
--dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
--criterion $CRITERION --label-smoothing 0.1 \
--update-freq 32  --max-sentences 2 \
--num-workers 4 \
--load-from-pretrained-model $PRETRAINED_MODEL \
--load-sep \
--ddp-backend=no_c10d --max-epoch 50 \
--max-source-positions 512 --max-target-positions 512 \
--skip-invalid-size-inputs-valid-test \
--seed 1 \
--save-dir $SAVE_DIR \
--keep-last-epochs 10 \
--tensorboard-logdir $TENSORBOARD_LOGDIR \
$DATA_DIR
```

#### Human Action Planning
```
DATA_DIR=planning/processed
USER_DIR=./prophetnet
ARCH=ngram_transformer_prophet_large
CRITERION=ngram_language_loss
SAVE_DIR=planning/finetune_planning_checkpoints
TENSORBOARD_LOGDIR=planning/finetune_planning_tensorboard
PRETRAINED_MODEL=pretrained_checkpoints/prophetnet_en.pt

fairseq-train \
--fp16 \
--user-dir $USER_DIR --task translation_prophetnet --arch $ARCH \
--optimizer adam --adam-betas '(0.9, 0.999)' --clip-norm 0.1 \
--lr 0.0001 \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 1000 \
--dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
--criterion $CRITERION --label-smoothing 0.1 \
--update-freq 32  --max-sentences 2 \
--num-workers 4 \
--load-from-pretrained-model $PRETRAINED_MODEL \
--load-sep \
--ddp-backend=no_c10d --max-epoch 50 \
--max-source-positions 512 --max-target-positions 512 \
--skip-invalid-size-inputs-valid-test \
--seed 1 \
--save-dir $SAVE_DIR \
--keep-last-epochs 10 \
--tensorboard-logdir $TENSORBOARD_LOGDIR \
$DATA_DIR
```

### Checkpoints
- ProphetNet-large-160GB (fine-tuned on MUHACU Human Action Forecasting Task) [[link]]()
- ProphetNet-large-160GB (fine-tuned on MUHACU Human Action Planning Task) [[link]]()


## TIPS:
If you met problems to run fairseq-preprocess, fairseq-train and other commands, or if you want to modify the workflow/inference pipeline, 
it's a good choice to download fairseq git repo, checkout v0.9.0, and merge our codes.   
Then, modify their preprocess.py, train.py or generate.py, to run your new pipeline. 

## Repo Reference
This repo is partially referred to Fairseq-v0.9.0 and MASS.



## How to Cite
If you extend or use this work, please cite the [paper](https://arxiv.org/pdf/2001.04063) where it was introduced:
```
@inproceedings{qi2020prophetnet,
  title={Prophetnet: Predicting future n-gram for sequence-to-sequence pre-training},
  author={Qi, Weizhen and Yan, Yu and Gong, Yeyun and Liu, Dayiheng and Duan, Nan and Chen, Jiusheng and Zhang, Ruofei and Zhou, Ming},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: Findings},
  pages={2401--2410},
  year={2020}
}
@article{qi2021prophetnet,
  title={ProphetNet-X: Large-Scale Pre-training Models for English, Chinese, Multi-lingual, Dialog, and Code Generation},
  author={Qi, Weizhen and Gong, Yeyun and Yan, Yu and Xu, Can and Yao, Bolun and Zhou, Bartuer and Cheng, Biao and Jiang, Daxin and Chen, Jiusheng and Zhang, Ruofei and others},
  journal={arXiv preprint arXiv:2104.08006},
  year={2021}
}
```
[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)
