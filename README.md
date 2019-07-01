
## HypER: Hypernetwork Knowledge Graph Embeddings

<p align="center">
  <img src="https://raw.githubusercontent.com/ibalazevic/HypER/master/hyper.png"/ width=800>
</p>

This codebase contains PyTorch implementation of the paper:

> Hypernetwork Knowledge Graph Embeddings.
> Ivana Balažević, Carl Allen, and Timothy M. Hospedales.
> International Conference on Artificial Neural Networks, 2019.
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

To reproduce the results from the paper, use the following combinations of hyperparameters with `batch_size=128`, `ent_vec_dim=200` and `rel_vec_dim=200`:

dataset | lr | dr | input_dropout | feature_map_dropout | hidden_dropout | label_smoothing 
:--- | :---: | :---: | :---: | :---: | :---: | :---: | 
FB15k | 0.005 | 0.995 | 0.2 | 0.2 | 0.3 | 0.
WN18 | 0.001 | 1.0 | 0.2 | 0.2 | 0.3 | 0.1
FB15k-237 | 0.0001 | 0.995 | 0.3 | 0.2 | 0.3 | 0.1
WN18RR | 0.005 | 1.0 | 0.2 | 0.2 | 0.3| 0.1

### Requirements

The codebase is implemented in Python 3.6.6. Required packages are:

    numpy      1.14.5
    pytorch    0.4.0

### Citation

If you found this codebase useful, please cite:

    @inproceedings{balazevic2019hypernetwork,
    title={Hypernetwork Knowledge Graph Embeddings},
    author={Bala\v{z}evi\'c, Ivana and Allen, Carl and Hospedales, Timothy M},
    booktitle={International Conference on Artificial Neural Networks},
    year={2019}
    }
