
## HypER: Hypernetwork Knowledge Graph Embeddings

This repository contains code for the paper [Hypernetwork Knowledge Graph Embeddings](https://arxiv.org/pdf/1808.07018.pdf).

### Running a model

`CUDA_VISIBLE_DEVICES=0 python hyper.py --algorithm HypER dataset FB15k-237`

This will run the HypER model on FB15k-237.

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
    
### Citation

If you found this repository useful, please cite:

    @article{balazevic2018hypernetwork,
    title={Hypernetwork Knowledge Graph Embeddings},
    author={Bala\v{z}evi\'c, Ivana and Allen, Carl and Hospedales, Timothy M},
    journal={arXiv preprint arXiv:1808.07018},
    year={2018}
    }
