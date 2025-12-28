# GCN Emotion



Below is the implementation doc of this repo:

> Our code utilizes different public repos on GitHub:
>
> 1. `datasets.py` and `loader.py` reference the public repo *EEGain*: https://github.com/EmotionLab/EEGain
> 2. The model part references the benchmark work *LibEER*: https://github.com/XJTU-EEG/LibEER/tree/main

### `datasets.py`

> EEGain 在用数据集的时候，只考虑session 1，这是不好的，我们需要修正这个，让DataSet能同时读三个session的数据。
> 
> 但是一个更简便的方法似乎是，把三个session的数据全都放到 `1/` 里面 /doge。后续如果我们有时间把三个session这个内容实现一下。


Notes:
It seems that the output of `DataSet()` didn't match the number of the raw file. This is the info how many data loaded for the subject 1:

```
[INSPECT TRANSFORM] Subject 1
------------------------------------------------------------
Trial 0<EOF>/public/home/hugf2022/emotion/seediv/eeg_raw_data/1/1_20160518.mat/0
  Segments       : 14
  Channels       : 62
  Samples/segment: 800
  Label          : 1
------------------------------------------------------------
Trial 1<EOF>/public/home/hugf2022/emotion/seediv/eeg_raw_data/1/1_20160518.mat/1
  Segments       : 7
  Channels       : 62
  Samples/segment: 800
  Label          : 2
------------------------------------------------------------
Trial 2<EOF>/public/home/hugf2022/emotion/seediv/eeg_raw_data/1/1_20160518.mat/2
  Segments       : 17
  Channels       : 62
  Samples/segment: 800
  Label          : 3
------------------------------------------------------------
```
However, in the raw file, it seems that there are 24 trails in one file. Here is the key of a file:

```
dict_keys(['__header__', '__version__', '__globals__', 'cz_eeg1', 'cz_eeg2', 'cz_eeg3', 'cz_eeg4', 'cz_eeg5', 'cz_eeg6', 'cz_eeg7', 'cz_eeg8', 'cz_eeg9', 'cz_eeg10', 'cz_eeg11', 'cz_eeg12', 'cz_eeg13', 'cz_eeg14', 'cz_eeg15', 'cz_eeg16', 'cz_eeg17', 'cz_eeg18', 'cz_eeg19', 'cz_eeg20', 'cz_eeg21', 'cz_eeg22', 'cz_eeg23', 'cz_eeg24'])
```

**We need to solve this problem later.**