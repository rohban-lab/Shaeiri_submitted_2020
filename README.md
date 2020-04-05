# Towards Deep Learning Models Resistant to Large Perturbations

This repository contains pre-trained models and codes to reproduce the main experiments of our paper:

**Towards Deep Learning Models Resistant to Large Perturbations**  
*Amirreza Shaeiri, Rozhin Nobahari, Mohammad Hossein Rohban*  
https://arxiv.org/abs/2003.13370

While “adversarial training” fails to train a deep neural network given a large, but reasonable, perturbation magnitude, in this paper, we propose a simple yet effective initialization of the network weights that makes learning on higher levels of noise possible. We evaluate this idea rigorously on MNIST (ε up to ≈ 0.40) and CIFAR10 (ε up to ≈ 32/255) datasets assuming the l∞ attack model.

## Benchmarks
1. Below is a list of two models on CIFAR10 (rows) and their computed accuracies under PGD200 attack.

|                 | Natural |  8/255  |  16/255 |  32/255 |
|-----------------|:-------:|:-------:|:-------:|:-------:|
| 8/255 -> 16/255 |   75%   |   56%   |   32%   |   10%   |
| 8/255 -> 32/255 |   33%   |   28%   |   24%   |   16%   |


2. Below is a list of two models on MNIST (rows) and their computed accuracies under PGD200 attack.

|            | Natural |   0.3   |   0.4   |
|------------|:-------:|:-------:|:-------:|
| 0.3 -> 0.4 |   98%   |   94%   |   90%   |
| IAT        |   98%   |   90%   |   91%   |

Note that these accuracies might be slightly different when you run your own model.


## Running the code

Having cloned the repository, you can reproduce our results:

#### 1. Extended adv training on CIFAR10:
```
```

#### 2. IAT on MNIST:
```
```

#### 3. Extended adv training on MNIST:
```
```


## Pretrained models


## Citation
When citing this work, you should use the following bibtex:

    @misc{shaeiri2020deep,
        title={Towards Deep Learning Models Resistant to Large Perturbations},
        author={Amirreza Shaeiri and Rozhin Nobahari and Mohammad Hossein Rohban},
        year={2020},
        eprint={2003.13370},
        archivePrefix={arXiv},
        primaryClass={stat.ML}
    }
