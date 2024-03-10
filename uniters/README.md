## UNITERs

In this directory we describe how we run experiments using UNITERs. We use parts of the code from [Visually Grounded Reasoning over Languages and Cultures (MaRVL)](https://arxiv.org/abs/2109.13238). More detailed instructions can be found in their github repo: [https://github.com/marvl-challenge/marvl-code](https://github.com/marvl-challenge/marvl-code).

## Set Up

1. Set up necessary packages

```bash
conda create -n uniter python=3.6 -y
conda activate uniter
pip install -r requirements.txt
cd volta/apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ../tools/refer
make
```

2. Download preprocessed visual features of MaRVL from [ERDA](https://sid.erda.dk/cgi-sid/ls.py?share_id=hmoEs4a3oG). Then, convert these H5 files into LMDB format by:

```bash
cd feature_extraction
bash h5_to_lmdb.sh
```

3. Follow the instructions in `uniters/volta/data/nlvr2`, extract and serialize NLVR2 image features. Store NLVR2's visual features as `nlvr2_feat.lmdb`.

4. Download pretrained mUNITER and xUNITER from [ERDA](https://sid.erda.dk/cgi-sid/ls.py?share_id=HfTaLDBWJi). 

## Finetuning

1. Go to `uniters/volta/config_tasks/xling_trainval_nlvr2.yml`: modify `data_root` to be the directory where all NLVR2 annotations are stored; modify `train_annotations_jsonpath` to be the train json file of NLVR2; modify `val_annotations_jsonpath` to be the dev json file of NLVR2; modify `features_h5path1` to be the path to `nlvr2_feat.lmdb`. For example:

```yml
dataroot: Multilingual_Visual_Reasoning/data/en/annotations
features_h5path1: Multilingual_Visual_Reasoning/data/en/nlvr2_feat.lmdb
train_annotations_jsonpath: Multilingual_Visual_Reasoning/data/en/annotations_json/train.json
val_annotations_jsonpath: Multilingual_Visual_Reasoning/data/en/annotations_json/dev.json
```

2. Take xUNITER as an example. Go to `uniters/experiments/ctrl_xuniter/nlvr2/train.sh`: modify `PRETRAINED` to be the path to the pretrained xUNITER model downloaded from ERDA; modify `OUTPUT_DIR` and `LOGGING_DIR` to be where you want to store outputs and logs. Outputs will be around 20GB.

```shell
PRETRAINED=/uniters/pretrained/${MODEL}/pytorch_model_9.bin
OUTPUT_DIR=/uniters/results/${MODEL}/NLVR2/train
LOGGING_DIR=/logs/uniters/${MODEL_CONFIG}
```

3. Run ```bash train.sh``` to finetune the model. Using one v100 took us around 10 hours for finetuning each model.

## Testing

Similar to what is done for finetuning, modify `uniters/volta/config_tasks/xling_test_marvl.yml` and the shell files in `uniters/experiments/ctrl_xuniter/marvl` and `/uniters/experiments/ctrl_xuniter/marvl_translate-test`. For `PRETRAINED`, use the best checkpoint `pytorch_model_best.bin` obtained from finetuning.

## References

```
@inproceedings{liu-etal-2021-visually,
    title = "Visually Grounded Reasoning across Languages and Cultures",
    author = "Liu, Fangyu  and
      Bugliarello, Emanuele  and
      Ponti, Edoardo Maria  and
      Reddy, Siva  and
      Collier, Nigel  and
      Elliott, Desmond",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.818",
    pages = "10467--10485",
}
```