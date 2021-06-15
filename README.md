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

![system](/retrieval.png)

## Retrieval System and MUHACU Knowledge Base
### Requirements and Setup
Python version >= 3.7

PyTorch version >= 1.4.0

``` bash
# clone the repository
git clone https://github.com/MUHACU/MUHACU.git
cd MUHACU
pip install -r requirements.txt
```

### Download MUHACU Knowledge Base

Download the Knowledge Base csv files from [here](https://zenodo.org/record/xxxx) and place them under the [xxx](xxx) folder.

**Multi-modal End-to-End Hybrid Vision & Language Framework**

![system](/e2e.png)

## Hybrid VL System and MUHACU Dataset
### UniVL
### Requirements and Setup
Python version >= 3.7

PyTorch version >= 1.4.0

``` bash
# clone the repository
git clone https://github.com/MUHACU/MUHACU.git
cd MUHACU/UniVL
pip install -r requirements.txt
```

###  Download MUHACU Dataset

Download the Dataset files from [here](https://zenodo.org/record/xxxx) and place them under the [xxx](xxx) folder.

### Train tasks
Please check [UniVL](UniVL) folder for details.

<br></br>
### ProphetNet
### Requirements and Setup
pip install torch==1.3.0
pip install fairseq==v0.9.0
pip install tensorboardX==1.7

###  Download MUHACU Dataset

Download the Dataset files from [here](https://zenodo.org/record/xxxx) and place them under the [xxx](xxx) folder.

### Train & Test tasks
Please check [ProphetNet](ProphetNet) folder for details.

All logs and checkpoints will be saved under the experiments folder.

<br></br>

## License
The repository is under [MIT License](LICENSE).

## Cite
``` bash
Coming Soon!
```


