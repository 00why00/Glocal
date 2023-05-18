# Glocal Energy-based learning

Official implementation of `Glocal Energy-based Learning for Few Shot Open-Set Recognition` (accepted by CVPR 2023).

[[`Paper`](https://arxiv.org/abs/2304.11855)]
[[`BibTex`](#citation)]

## Requirements

This repo is tested with Python 3.10, Pytorch 1.13, CUDA 11.6. More recent versions of Python and Pytorch with compatible CUDA versions should also support the code.

## Getting Start

### 1. Prepare Dataset

Dataset Source can be downloaded here.

- [MiniImageNet](https://drive.google.com/file/d/12V7qi-AjrYi6OoJdYcN_k502BM_jcP8D/view?usp=sharing)
- [TieredImageNet](https://drive.google.com/open?id=1g1aIDy2Ar_MViF2gDXFYDBTR-HYecV07)
- [CIFAR-FS](https://drive.google.com/file/d/1GjGMI0q3bgcpcB_CjI40fX54WgLPuTpS/view?usp=sharing)

Download them and move datasets under `data` folder.

### 2. Prepare Pretrain Weights

Download pretrain weights [here](https://drive.google.com/drive/folders/1C9l-0SAw__k3OVRaxXMQ41SoG7RUeEAu?usp=sharing) and move them under `initialization` folder.

### 3. Running

Just run ` main.py ` for training and testing

For example, to train and test the 5-way 1-shot/5-shot setting on MiniImageNet:

```
python main.py --gpu 0 --max_epoch 100 --attention --energy --energy_loss --open_loss --lr 0.0001 --energy_method sum --pixel_wise --distance pixel_sim --pixel_conv --init_weights [/path/to/pretrained/weights] --dataset MiniImageNet --shot [number of shots] --ahead_combine --top_k [number of k]
```

to train and test the 5-way 1-shot/5-shot setting on TieredImageNet:

```
python main.py --gpu 0 --max_epoch 200 --attention --energy --energy_loss --open_loss --lr 0.0002 --energy_method sum --pixel_wise --distance pixel_sim --pixel_conv --init_weights [/path/to/pretrained/weights] --dataset TieredImageNet --shot [number of shots] --ahead_combine --top_k [number of k]
```

to train and test the 5-way 1-shot/5-shot setting on CIFAR-FS:

```
python main.py --gpu 0 --max_epoch 100 --attention --energy --energy_loss --open_loss --lr 0.0001 --energy_method sum --pixel_wise --distance pixel_sim --pixel_conv --init_weights [/path/to/pretrained/weights] --dataset CIFAR-FS --shot [number of shots] --ahead_combine --top_k [number of k]
```

to train and test the 5-way 1-shot/5-shot setting on CIFAR-FS without pixel branch:

```
python main.py --gpu 0 --max_epoch 100 --attention --energy --energy_loss --open_loss --lr 0.0001 --energy_method sum --init_weights [/path/to/pretrained/weights] --dataset CIFAR-FS --shot [number of shots]
```

## Citation

```
@InProceedings{Wang_2023_CVPR,
    author    = {Wang, Haoyu and Pang, Guansong and Wang, Peng and Zhang, Lei and Wei, Wei and Zhang, Yanning},
    title     = {Glocal Energy-Based Learning for Few-Shot Open-Set Recognition},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {7507-7516}
}
```

## Acknowledgement

Part of the code is modified from [FEAT](https://github.com/Sha-Lab/FEAT), [SnaTCHer](https://github.com/MinkiJ/SnaTCHer), [TANE](https://github.com/shiyuanh/TANE) and [RFDNet](https://github.com/shule-deng/RFDNet).
