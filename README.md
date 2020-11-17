# SparseVQVAE: Sparse Dictionary based Vector Quantized Variational AutoEncoder
Experimental implementation for a sparse-dictionary based version of the VQ-VAE2 paper
(see: [Generating Diverse High-Fidelity Images with VQ-VAE-2](https://arxiv.org/abs/1906.00446))

This repository builds over PyTorch.

## Introduction

VQ-VAE is a promising direction for image synthesis, that is completely separate from the GAN line of works.
The main idea of this codebase is to create a generalized VQ-VAE,
by replacing the hard selection in the heart of the method to a softer selection by sparse coding.
This stems from the observation that hard selection is in essence the “sparsest code”,
i.e. single non-zero element (or- one hot vector).

In this generalized implementation, we allow the VAE to code each patch with a small set sparse dictionary atoms,
rather than a single code as done in the original work.
We therefore build over the VQVAE2 paper:
 1. We perform sparse dictionary learning, to generate a set of atoms best describing the data.
 2. During training, new images are encoded per patch, where each patch is encoded by a small set of atoms.
 3. We then decode the image back from sparse-codes to pixel space using a learned encoder.
 
 During inference time images may get compressed by employing both encoder & decoder.
 Alternatively, new images can be synthesized by randomizing sparse codes and employing only the decoder.  

We summarize the main contributions of this repository as follows:
1. Sparse dictionary over Pytorch:
    - Sparse dictionary is learned via [Task-Driven Dictionary Learning][1], implemented to be compatible with PyTorch's auto-differentiation.
    - Fast parallel implementations of the [FISTA][2] and [OMP][3] sparse-coding algorithms.
2. A complete sparse-dictionary empowered VQ-VAE2 implementation, including training & evaluation code.   

[1]: https://arxiv.org/abs/1009.5358
[2]: https://people.rennes.inria.fr/Cedric.Herzet/Cedric.Herzet/Sparse_Seminar/Entrees/2012/11/12_A_Fast_Iterative_Shrinkage-Thresholding_Algorithmfor_Linear_Inverse_Problems_(A._Beck,_M._Teboulle)_files/Breck_2009.pdf
[3]: http://www.cs.technion.ac.il/~ronrubin/Publications/KSVD-OMP-v2.pdf

## Dictionary Learning

Explain FISTA vs OMP here

#### Task-Driven Dictionary Learning

#### FISTA

#### OMP

## Applications

#### Compression
#### Synthesis

## Limitations

Explain where work on this repo have halted.

## Installation

Code assumes Linux environment (tested on Ubuntu 16).

### Prerequisites

* python >= 3.6
* pytorch >= 1.4
* cuda 10 or higher (recommended)

### Environment Setup

After cloning this repository:
> cd Sparse_VAE

> pip install -r requirements.txt

## Project Structure

Highlight the different folders - emphasis on FISTAFunction and OMPFunction.

## Usage

1. Training the Sparse-VQVAE encoder-decoder:

Currently this codebase supports CIFAR10 and ImageNet.

* Train with original VQVAE:
> python train_vqvae.py —dataset cifar10 —selection_fn vanilla —size 32 —normalize_embeddings

* Train with FISTA sparse-coding:
> python train_vqvae.py —dataset cifar10 —selection_fn vanilla —size 32 —normalize_embeddings

* TBD: OMP

2. Extract codes for stage 2 training

> python extract_code.py --ckpt checkpoint/[VQ-VAE CHECKPOINT] --name [LMDB NAME] [DATASET PATH]

Or's note: this text is from the original VQVAE repo - applies to us?

3. Stage 2 (PixelSNAIL)

> python train_pixelsnail.py [LMDB NAME]

Or's note: this text is from the original VQVAE repo - applies to us?

## Samples

### Stage 1

Note: This is a training sample

![Sample from Stage 1 (VQ-VAE)](stage1_sample.png)
