# SNGAN.pytorch
An unofficial Pytorch implementation of [Spectral Normalization for Generative Adversarial Networks](https://openreview.net/pdf?id=B1QRgziT-). 
For official Chainer implementation please refer to [https://github.com/pfnet-research/sngan_projection](https://github.com/pfnet-research/sngan_projection)

This repository is a fork of [GongXinyuu/sngan.pytorch](https://github.com/GongXinyuu/sngan.pytorch). 

The main differences are as following:
* celeba64 dataset support added
* data generation code added
* Tensorflow v1 dependency removed

## Set-up

### install libraries:
```bash
pip install -r requirements.txt
```

### train
```bash
./exps/sngan_celeba.sh
```

### generate
```bash
./exps/gen_celeba.sh
```
