Data loaders for pretrain and downstream tasks (retrieval and caption). 

## Preprocess on HowTo100M

For pretrain, you need to prepare 3 parts,

### 1. s3d features pretrained on HowTo100M

Download raw videos from the [HowTo100M webpage]([https://www.di.ens.fr/willow/research/howto100m/](https://www.di.ens.fr/willow/research/howto100m/)) and extract [s3d (howto100m)](https://github.com/antoine77340/S3D_HowTo100M) features. You can refer to [VideoFeatureExtractor](https://github.com/ArrowLuo/VideoFeatureExtractor).

### 2. HowTo100M.csv
Note: this file is different from HowTo100M_v1.csv as in [README.txt](https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/README.txt)

The csv format contains two columns. The first column is the video id, and the second is the feature file (sub-path of the npy, which will post append to `--features_path` (refer to pretrain part in [README](../README.md)) to find the npy file when reading).

```
video_id,feature_file
Z8xhli297v8,Z8xhli297v8.npy
...
```
video_id: used to match the caption or transcript
feature_file: used to find the feature file after joining with `--features_path`

### 3. caption.pickle
This pickle file is generated from raw_caption.json in raw_caption.zip introduced in [README.txt](https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/README.txt)

The format of this file is:
```
{
    'video_id 1':{
        'start': array([0.08, 7.37, 15.05, ...], dtype=object),
        'end': array([9.96, 16.98, 27.9, ...], dtype=object),
        'text': array(['sentence 1 placehodolder',
                    'sentence 2 placehodolder',
                    'sentence 3 placehodolder', ...], dtype=object)
    },
    ...
}
```
Keep the `start` is a sorted array.


## Preprocess on YoucookII
The s3d feature extraction is the same as HowTo100M introduced above.

## Generate youcookii_data.pickle
This file is generated from `youcookii_annotations_trainval.json`, which can be downloaded from [official webpage](http://youcook2.eecs.umich.edu/download).

The format of this file is (similar to `caption.pickle` introduced above, but one more key `transcript`. The `transcript` needs to generated by extra ASR tool from speech.):
```
{
    'video_id 1':{
        'start': array([0.08, 7.37, 15.05, ...], dtype=object),
        'end': array([9.96, 16.98, 27.9, ...], dtype=object),
        'text': array(['sentence 1 placehodolder',
                    'sentence 2 placehodolder',
                    'sentence 3 placehodolder', ...], dtype=object)
        'transcript': array(['transcript 1 placehodolder',
                    'transcript 2 placehodolder',
                    'transcript 3 placehodolder', ...], dtype=object)
    },
    ...
}
```
If you want to test on retrieval or caption w/o transcript tasks, you can set `transcript` with `array(['NONE', 'NONE', 'NONE', ...], dtype=object)`.

## Format of csv
```
video_id,feature_file
Z8xhli297v8,Z8xhli297v8
...
```
Note: The video_id and feature_file are the same for the consistency and our historical compatibility. We use feature_file to get the feature from feature pickle.

## Preprocess on MSRVTT
The s3d feature extraction is the same as HowTo100M introduced above.
The data can be downloaded in: https://github.com/microsoft/UniVL/releases/download/v0/msrvtt.zip