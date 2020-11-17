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

This sparse coding problem involves integer programming over a non-covex L0 norm, and therefore is NP-hard.
In practice, the solution is approximated using pursuit algorithms, where the atoms "compete" over which get to describe the input signal.
Generally speaking, there are two flavours of pursuit algorithms: greedy and convex-relaxation.
We provide one example from each family


#### OMP

This method approximates the exact L0 norm solution in a greedy manner, selecting the next atom with the smallest (angular) residual w.r.t. the current code. 
The benefit here is that we have a guaranteed number of `K` non-zer0 elements after `K` iterations.
On the other hand, the selection process makes the process itself less suitable for differentiable programming (aka back-prop).

#### FISTA

Here, the L0 is relaxed to its nearest convex counterpart, the L1 norm which is treated as an additive penalty.
The resulting LASSO problem is a convex one, and has several efficient methods to solve efficiently.
The iterative nature of this methos allows unrolling its structure and approximating it using a neural net (see [LISTA](http://yann.lecun.com/exdb/publis/pdf/gregor-icml-10.pdf))
The drawback here is that the resulting code can have arbitrary number of non-zero elements after a fixed number of iterations.

#### Task-Driven Dictionary Learning

Without going into too many details, this paper proposes a way to calculate the derivative of the spase coding problem with respect to the dictionary.
This is opens the way for a bi-level optimisation procedure, where we optimize the result of an optimization process. 
Using this method we can create a dictionary optimized for any task, specifically the one our vq-vae is meant to solve. 

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

Currently this codebase supports CIFAR10, CIFAR100, and ImageNet.

* Train with original VQVAE:
```
train_vqvae.py --experiment_name="experiment_vq" --selection_fn=vanilla 
```

* Train with FISTA sparse-coding:

```
train_vqvae.py --experiment_name="experiment_fista" --selection_fn=fista 
```

* Train with OMP sparse-coding:

```
train_vqvae.py --experiment_name="experiment_omp" --selection_fn=fista --num_strides=2
```


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
