
## HypER: Hypernetwork Knowledge Graph Embeddings

<p align="center">
  <img src="https://raw.githubusercontent.com/ibalazevic/HypER/master/hyper.png"/ width=800>
</p>

This codebase contains PyTorch implementation of the paper:

> Hypernetwork Knowledge Graph Embeddings.
> Ivana Balažević, Carl Allen, and Timothy M. Hospedales.
> arXiv preprint arXiv:1808.07018, 2018.
> [[Paper]](https://arxiv.org/pdf/1808.07018.pdf)

### Running a model

To run the model, execute the following command:

`CUDA_VISIBLE_DEVICES=0 python hyper.py --algorithm HypER --dataset FB15k-237`


Available algorithms are:

    HypER
    HypE
    DistMult
    ComplEx
    ConvE

Available datasets are:
    
    FB15k-237
    WN18RR
    FB15k
    WN18

### Requirements

The codebase is implemented in Python 3.6.6. Required packages are:

    numpy      1.14.5
    pytorch    0.4.0

### Citation

If you found this codebase useful, please cite:

    @article{balazevic2018hypernetwork,
    title={Hypernetwork Knowledge Graph Embeddings},
    author={Bala\v{z}evi\'c, Ivana and Allen, Carl and Hospedales, Timothy M},
    journal={arXiv preprint arXiv:1808.07018},
    year={2018}
    }
