# SparseVQVAE: Sparse Dictionary based Vector Quantized Variational AutoEncoder
Experimental implementation for a sparse-dictionary based version of the VQ-VAE2 paper
(see: [Generating Diverse High-Fidelity Images with VQ-VAE-2](https://arxiv.org/abs/1906.00446))

This repository builds over PyTorch.

## Authors
Yiftach Ginger ([iftachg](https://github.com/iftachg)), Or Perel ([orperel](https://github.com/orperel)), Roee Litman ([rlit](https://github.com/rlit))

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

This reporistory contains research materials of an unpublished work.
Training + Inference code based on FISTA and OMP over PyTorch is fully functional for compression use cases.
PixelSnail synthesis functionality is partially supported.

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

FISTAFunction and OMPFunction are fast GPU implementations of both algorithms over PyTorch.
Practitioners are welcome to incorporate these functions into their repositories under the license terms of this repository.

The rest of the project structure can be briefly described as such:

* checkpoint/
    * MODELS SAVED HERE for vanilla and pixelsnail (both), as well as args used to generate them.
* models/
    * fista_pixelsnail - the implementation of the modifier pixelsnail based on FISTA
    * model_utils - contains functions for genrating VQVAE objects and loading datasets (CIFAR, imagenet..). All files are downloaded relative to the project path.
    * pixelsnail - the original pixelsnail model. fista_pixelsnail overrides this model and adds additional heads. The vanilla model generates this model twice (top and bottom)
    * quantizers - Contains only the stuff that generates quantized codes: FISTA, OMP and Vanilla VQVAE quantization.
    * vqvae - composed of Encoder / Decoder. Of interest here: we can change the stride to achieve different effects (0 - for decompression ; 1- for vqvae vanilla ; 2 - for compression). The stride should change for both Encoder / Decoder
* scripts/
    * calculate_jpg_psnr - a standalone script, accepts a dataset (hardcoded cifar) and runs compression for multiple quality levels. Outputs the psnr..
    * calculate_model_psnr - similar to the above, only this one receives a model as input and prints it’s compression psnr. Note we have our own manual calculation of PSNR here. FISTA converges for multiple images at the same time, so the slowest image in the batch determines the bottleneck speed. If we run with batch size 1 we’re faster and more accurate.
    * extract_dataset_unlearned_encodings - skip that (was used for experiments on alpha).
    * graph_psnr - takes the PSNR tables we’ve created and generates plots.
    * hyperparameter_alpha_search - convergence of alpha related to amount of nonzeros - calculated twice for random data and second time for the script we’ve just skipped. Most probably we shouldn’t be touching this script..
    * visualize_encodings - a visualization script Yiftach have created for himself. Here we take a model and a dataset, run the model over the dataset and save the output image, to test it’s still valid. If all goes well we shouldn’t be using this file..
* utils/
    * pyfista is implemented here, both. dictionary learning and 
    * pyfista_test - generates fake data to train sparse coding.. We don’t do hyperparams search anymore so we have no additiona uses for this file.
    * pyomp - Holds implementation of forwards for OMP for a single sample at a time (TODO: implement batch OMP if we want).
    * util_funcs - lots of helper functions are stored here. Argument parsers are handled here, as well as seeding and experiments setup (general stuff like assigning an experiment name..)
* dataset - all definitions for used datasets. These are definitions for datasets but there is nothing to configure here.
* extract_code - main for extract_code (2nd step in the algorithm training..)
* mt_sample - multi threaded sampling.. Currently broken. 
* sample - receives a PixelSnail and starts generating images..
* scheduler - Scheduling definitions for number of schedulers, when to save a checkpoint file.. etc..
* train_fista_pixelsnail / train_pixelsnail / train_vqvae - all neural net trainers we support..
* scheduler - Scheduling definitions for number of schedulers, when to save a checkpoint file.. Etc

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


For synthesis, additional steps are required:

2. Extract codes for stage 2 training

> python extract_code.py --ckpt checkpoint/[VQ-VAE CHECKPOINT] --name [LMDB NAME] [DATASET PATH]


3. Stage 2 (PixelSNAIL)

> python train_pixelsnail.py [LMDB NAME]

