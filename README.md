# Keep It on a Leash: Controllable Pseudo-label Generation Towards Realistic Long-Tailed Semi-Supervised Learning
Yaxin Hou, Bo Han, Yuheng Jia, Hui Liu, Junhui Hou, Keep It on a Leash: Controllable Pseudo-label Generation Towards Realistic Long-Tailed Semi-Supervised Learning, Advances in Neural Information Processing Systems, 2nd-7th December, San Diego, 2025.

This is an official [PyTorch](http://pytorch.org) implementation for **Keep It on a Leash: Controllable Pseudo-label Generation Towards Realistic Long-Tailed Semi-Supervised Learning**.

## Introduction
This code is based on the public and widely-used codebase [USB](https://github.com/microsoft/Semi-supervised-learning).

What I've done is just adding our CPG algorithm in `semilearn/imb_algorithms/cpg`.

Also, I've made corresponding modifications to `semilearn/nets/` and several `__init__.py`.

## How to run
For example, on CIFAR-10-LT with long-tailed labeled data ($\gamma_l=100$) and arbitrary unlabeled data($\gamma_u=100$)

```
CUDA_VISIBLE_DEVICES=0 python train.py --c "config/config-1/dawn/115-fixmatch_dawn_food101_lb50_10_ulb450_10_random_0.0_1.yaml"
```
```
CUDA_VISIBLE_DEVICES=0 python train.py --c "config/config-1/dawn/114-fixmatch_dawn_food101_lb50_10_ulb450_10_long-tail_0.0_1.yaml"
```
```
CUDA_VISIBLE_DEVICES=0 python train.py --c "config/config-1/dawn/113-fixmatch_dawn_food101_lb50_10_ulb450_10_reverse-long-tail_0.0_1.yaml"
```
```
CUDA_VISIBLE_DEVICES=0 python train.py --c "config/config-1/dawn/213-fixmatch_dawn_food101_lb50_15_ulb450_15_reverse-long-tail_0.0_1.yaml"
```



```
CUDA_VISIBLE_DEVICES=0 python train.py --c "config/config-1/cpg/215-fixmatch_cpg_food101_lb50_15_ulb450_15_random_0.0_1.yaml"
```

```
CUDA_VISIBLE_DEVICES=3 python train.py --c "config/config-1/cpg/116-fixmatch_cpg_stl10_lb500_100_ulb_full_0.0_1.yaml"
```
```
CUDA_VISIBLE_DEVICES=2 python train.py --c "config/config-1/cpg/216-fixmatch_cpg_stl10_lb500_150_ulb_full_0.0_1.yaml"
```
```
CUDA_VISIBLE_DEVICES=3 python train.py --c "config/config-1/fixmatch/fixmatch_stl10_500_100_ulb_full_0.0_1.yaml"
```
```
CUDA_VISIBLE_DEVICES=2 python train.py --c "config/config-1/fixmatch/fixmatch_stl10_500_150_ulb_full_0.0_1.yaml"
```
```
CUDA_VISIBLE_DEVICES=0 python train.py --c "config/config-1/freematch/freematch_stl10_500_150_ulb_full_0.0_1.yaml" --alg freematch
```
```
python stable_diffusion/generate.py --lb_idx_path "./stable_diffusion/food101/lb_labels_50_15_450_15_exp_random_noise_0.0_seed_1_idx.npy" --output_dir ./data/generated/food101/lb_50_15/label
```

(Note: I know that USB supports multi-GPUs, but I still recommend you to run on single GPU, as some weird problems may occur.)

The model will be automatically evaluated every 1024 iterations during training. After training, the last two lines in `/saved_models/cpg/203-fixmatch_cpg_cifar10_lb400_100_ulb4600_100_random_0.0_1/log.txt` will tell you the best accuracy. 

For example,
```
[2025-04-20 13:35:17,784 INFO] model saved: ./saved_models/cpg/203-fixmatch_cpg_cifar10_lb400_100_ulb4600_100_random_0.0_1/latest_model.pth
[2025-04-20 13:35:17,815 INFO] Model result - eval/best_acc : 0.8233
[2025-04-20 13:35:17,816 INFO] Model result - eval/best_it : 244735
```

## Results

The reported accuracies in Table 1, 2, and 3 in our paper are the average over three different runs (random seeds are 0/1/2). 

## Citation

If you find our method useful, please consider citing our paper:

  ```
  @inproceedings{cpgnips2025,
    title={Keep It on a Leash: Controllable Pseudo-label Generation Towards Realistic Long-Tailed Semi-Supervised Learning},
    author={Hou, Yaxin and Jia, Yuheng and Han, Bo and Liu, Hui and Hou, Junhui},
    booktitle={Advances in Neural Information Processing Systems},
    volume={},
    pages={},
    year={2025}
  }
  ```
