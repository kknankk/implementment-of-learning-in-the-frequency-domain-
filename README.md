# Implementment of Learning in the Frequency Domain

This repository contains the implementation of learning techniques in the frequency domain. The code includes methods and models for processing images using frequency domain transformations (Discrete Cosine Transform (DCT)).I used Tiny-imagenet dataset:https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet

## Acknowledgements

Some parts of this code were adapted from the [DCTNet](https://github.com/kaix90/DCTNet/tree/master) repository by [kaix90](https://github.com/kaix90). I would like to acknowledge and thank the original authors for their contributions to the field.

## Train dataset & train dataloader & train function
dataset folder and main folder were modified by myself. 
I used the torch-dct package to perform 8x8 DCT transformations and extract components of the same frequency, replacing the jpeg2dct package.
And I added the training part.

## training model
python main/train_val.py --gpu-id 2,3  --arch ResNetDCT_Upscaled_Static -d /data/ke/tiny-imagenet-200
