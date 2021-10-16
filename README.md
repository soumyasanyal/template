# DL Template
Integrates Pytorch lightning + Hydra + Neptune for LM finetuning

## To finetune bert on SST2 dataset:
```
python main.py
```

## To run without neptune (offline mode)
```
python main.py --offline
```

## To debug on small part of the dataset
```
python main.py --debug
```

## To override specific hydra config
```
python main.py --override fixed
```
This will start using the `fixed` lr scheduler instead of the `linear_with_warmup` mentioned in `configs/config.yaml`. Override can be comma separated config names that we want to override for each category (e.g., `--override fixed,adamw,rtx_2080`.
