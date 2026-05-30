# CAM-Net

Official implementation of **CAM-Net** for cross-view geo-localization.

The code is currently being organized and will be continuously updated.

## Dataset

Please prepare the datasets by yourself, including **CVUSA**, **CVACT**, and **VIGOR**.

You may follow the dataset preparation protocol of [TransGeo](https://github.com/Jeff-Zilence/TransGeo2022).

After downloading the datasets, please modify the corresponding dataset paths in the dataloader or evaluation scripts according to your local environment.

## Requirements

Please install the required packages with:

```bash
pip install -r requirements.txt
```

## Evaluation

You can run the evaluation scripts as follows:

```bash
python eval_cvusa_triple.py
```

or

```bash
python eval_cvact_triple.py
```

or

```bash
python eval_vigor_triple.py
```

Please make sure that the dataset paths and checkpoint paths are correctly specified before running the scripts.

## Training

The training code is currently being organized and will be released later.

## Pretrained Models

The pretrained weights are available at [CAM-Net](https://huggingface.co/Zilong0909/CAM-Net).

Please download the corresponding checkpoint and place it under the project directory

## Acknowledgement

This project is built upon and inspired by several excellent open-source projects, including:

* [Sample4Geo](https://github.com/Skyy93/Sample4Geo)
* [TransGeo](https://github.com/Jeff-Zilence/TransGeo2022)
* [ConvNeXt](https://github.com/facebookresearch/ConvNeXt)

We sincerely thank the authors for their great work.

