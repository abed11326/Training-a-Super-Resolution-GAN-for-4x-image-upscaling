# Training a Super-Resolution GAN for 4x image upscaling

## Introduction
This repository contains a PyTorch implementation of a Generative Adversarial Network (GAN) for super resolution image upscaling. Specifically, the GAN is trained to upscale low-resolution images by a factor of 4, producing high-resolution images that are visually similar to the original high-resolution images.

## Original Paper
The implementation is based mainly on this [paper](https://arxiv.org/abs/1609.04802v5) with some modifications.

## Modifications
The differences between this implementation and the original paper are:
- <b>Data:</b> The original paper uses a random sample of 350 thousand images from the ImageNet database, but, here a collection of DIV2K dataset, Flickr2K dataset, and OutdoorSceneTraining (OST) dataset is used.</b>
- <b>Discriminator:</b> Dropout layers are added to the discriminator to balance the training more.</b>
