# MUHACU: Multi-modal Human Activity Understanding

## About

MUlti-modal Human ACtivity Understanding (MUHACU) is a dataset consisting of video frames and corresponding high-level intents, providing cloze based QA sets.
The resource is designed to evaluate the ability of understanding the human indoor activity in multi-modal level. [The paper]() defines a benchmark evaluation consisting of the following tasks:
- Human Action Forecasting
- Human Action Planning

Additional details on the resource and benchmark evaluation are available in this [report]().
We proposed a retrieval & scoring based baseline via a multi-modal knowledge base containing meta information. The knowkedge base can be found [here](). 
Another end-to-end baseline method combines a SOTA approach of video captioning, [UniVL](), and a sequence-to-sequence model, [Propehtnet]().

**Multi-modal Knowledge Base Retrieval & Scoring Framework**

![system](/retrieval.pdf)

## IR+LE System and MLM Dataset 
### Requirements and Setup
Python version >= 3.7

PyTorch version >= 1.4.0

``` bash
# clone the repository
git clone https://github.com/GOALCLEOPATRA/MLM.git
cd MLM
pip install -r requirements.txt
```

### Download MLM dataset

Download the dataset hdf5 files from [here](https://zenodo.org/record/3885753) and place them under the [data](data) folder.

### Train tasks
Multitask Learning (IR + LE)
``` bash
python train.py --task mtl
```

Cross-modal retrieval task
``` bash
python train.py --task ir
```

Location estimation task
``` bash
python train.py --task le
```

For setting other arguments (e.g. epochs, batch size, dropout), please check [args.py](args.py).

### Test tasks
Multi-task Learning (IR + LE)
``` bash
python test.py --task mtl
```

Cross-modal retrieval task
``` bash
python test.py --task ir
```

Location estimation task
``` bash
python test.py --task le
```

All logs and checkpoints will be saved under the experiments folder.

## License
The repository is under [MIT License](LICENSE).

## Cite
``` bash
Coming Soon!
```


