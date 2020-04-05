# Towards Deep Learning Models Resistant to Large Perturbations

This repository contains pre-trained models and codes to reproduce the main experiments of our paper:

**Towards Deep Learning Models Resistant to Large Perturbations**  
*Amirreza Shaeiri, Rozhin Nobahari and Mohammad Hossein Rohban*  
https://arxiv.org/abs/2003.13370

While “adversarial training” fails to train a deep neural network given a large, but reasonable, perturbation magnitude. In this paper, we propose a simple yet effective initialization of the network weights that makes learning on higher levels of noise possible. We evaluate this idea rigorously on MNIST (ε up to ≈ 0.40) and CIFAR10 (ε up to ≈ 32/255) datasets assuming the l-inf attack model. Also, we have interesting theoretical results about the optimal robust classifier. 


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

Note that these accuracies might be **slightly different** when you run your own model. Also, It is worth noting that we did not try to fine-tune any of these models to improve the accuracy.


## Requirements
```
numpy
torch==1.4.0
torchvision
robustness
```


## Running the code

#### 1. Extended adv training on CIFAR10:
```
You can use train.py in CIFAR10 folder to train your own model using extended adversarial training. Note that base model is addapted from Madry Lab (https://github.com/MadryLab/robustness).
```

#### 2. IAT on MNIST:
```
You can use IAT.ipynb in MNIST folder to train your own model using IAT.
```

#### 3. Extended adv training on MNIST:
```
You can use extended.ipynb in MNIST folder to train your own model using extended adversarial training.
```

## Pretrained models
```
You can use test.py in MNIST and CIFAR10 folders to load our pre-trained models.
```


## Citation
When citing this work, please use the following bibtex:

    @misc{shaeiri2020deep,
        title={Towards Deep Learning Models Resistant to Large Perturbations},
        author={Amirreza Shaeiri and Rozhin Nobahari and Mohammad Hossein Rohban},
        year={2020},
        eprint={2003.13370},
        archivePrefix={arXiv},
        primaryClass={stat.ML}
    }
