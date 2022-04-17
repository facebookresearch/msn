# MSN  **M**asked **S**iamese **N**etworks

This repo provides a PyTorch implementation of MSN (**M**asked **S**iamese **N**etworks), as described in the paper [Masked Siamese Networks for Label-Efficient Learning](https://arxiv.org/abs/2204.07141).

<p align="center">
<img src="https://user-images.githubusercontent.com/7530871/162791683-7b17121d-358f-4780-a810-8a8f59b29203.png" width="75%">
</p>
  
MSN is a self-supervised learning framework that leverages the idea of mask-denoising while avoiding pixel and token-level reconstruction. Given two views of an image, MSN randomly masks patches from one view while leaving the other view unchanged. The objective is to train a neural network encoder, parametrized with a vision transformer (ViT), to output similar embeddings for the two views. In this procedure, MSN does not predict the masked patches at the input level, but rather performs the denoising step implicitly at the representation level by ensuring that the representation of the masked input matches the representation of the unmasked one.

### Low-shot evaluation on ImageNet-1K


<p align="center">
<img src="https://user-images.githubusercontent.com/7530871/163032763-7a467588-c16c-42f2-ac2e-8623d35f4db3.png" width="75%">
</p>
<sub>
Low-shot Evaluation of self-supervised models, pre-trained on ImageNet-1K. (Left) MSN surpasses the previous 800M parameter state-of-the-art. (Right) MSN achieves good classification performance using less labels than current mask-based auto-encoders.
</sub>

### Visualizations
We can use the RCDM framework of [Bordes et al., 2021](https://arxiv.org/abs/2112.09164) to qualitatively demonstrates the effectiveness of the MSN denoising process.

<p align="center">
<img src="https://user-images.githubusercontent.com/7530871/163029571-3ebc244c-e503-45f4-8a52-8d3f2ffed00d.png" width="60%">
</p>
<sub>
  First column: original image. Second column: image with 90% of patches masked used to compute representations of an MSN pre-trained ViT-L/7 encoder. Other columns: RCDM sampling from generative model conditioned on MSN representation of masked image. Qualities that vary across samples represent information that is not contained in the pre-trained representation; e.g., in this case, MSN discards background, pose, and lighting information. Qualities that are common across samples represent information contained in the pre-trained representation. Even with high-masking ratio, MSN retains semantic information about the images.
</sub>

## Pre-trained models
Coming soon!

## Running MSN self-supervised pre-training

### Config files
All experiment parameters are specified in config files (as opposed to command-line-arguments). Config files make it easier to keep track of different experiments, as well as launch batches of jobs at a time. See the [configs/](configs/) directory for example config files.

### Requirements
* Python 3.8 (or newer)
* PyTorch install 1.11.0 (older versions may work too)
* torchvision
* Other dependencies: PyYaml, numpy, opencv, submitit, cyanure

### Single-GPU training
Our implementation starts from the [main.py](main.py), which parses the experiment config file and runs the msn pre-training locally on a multi-GPU (or single-GPU) machine. For example, to run on GPUs "0","1", and "2" on a local machine, use the command:
```
python main.py
  --fname configs/pretrain/msn_vits16.yaml
  --devices cuda:0 cuda:1 cuda:2
```

### Multi-GPU training
In the multi-GPU setting, the implementation starts from [main_distributed.py](main_distributed.py), which, in addition to parsing the config file, also allows for specifying details about distributed training. For distributed training, we use the popular open-source [submitit](https://github.com/facebookincubator/submitit) tool and provide examples for a SLURM cluster. Feel free to edit [main_distributed.py](main_distributed.py) for your purposes to specify a different procedure for launching a multi-GPU job on a cluster.

For example, to pre-train with MSN on 16 GPUs using the pre-training experiment configs specificed inside [configs/pretrain/msn_vits16.yaml](configs/pretrain/msn_vits16.yaml), run:
```
python main_distributed.py
  --fname configs/pretrain/msn_vits16.yaml
  --folder $path_to_save_submitit_logs
  --partition $slurm_partition
  --nodes 2 --tasks-per-node 8
  --time 1000
```

## ImageNet-1K Logistic Regression Evaluation

##### Labeled Training Splits
For reproducibilty, we have pre-specified the labeled training images as `.txt` files in the [imagenet_subsets/](imagenet_subsets/) directory.
Based on your specifications in your experiment's config file, our implementation will automatically use the images specified in one of these `.txt` files as the set of labeled images.

To run logistic regression on a pre-trained model using some labeled training split you can directly call the script from the command line:
```
python logistic_eval.py
  --subset-path imagenet_subsets/5imgs_class.txt
  --root-path /datasets/ --image-folder imagenet_full_size/061417/
  --device cuda:0
  --pretrained $directory_containing_your_model
  --fname $model_filename
  --model-name vit_small
  --penalty l2
  --lambd 0.0025
```

## ImageNet-1K Linear Evaluation
To run linear evaluation on the entire ImageNet-1K dataset, use the `main_distributed.py` script and specify the `--linear-eval` flag.

For example, to evaluate MSN on 32 GPUs using the linear evaluation config specificed inside [configs/eval/lineval_msn_vits16.yaml](configs/eval/lineval_msn_vits16.yaml), run:
```
python main_distributed.py
  --linear-eval
  --fname configs/eval/lineval_msn_vits16.yaml
  --folder $path_to_save_submitit_logs
  --partition $slurm_partition
  --nodes 4 --tasks-per-node 8
  --time 1000
```

## ImageNet-1K Fine-Tuning Evaluation
For fine-tuning evaluation, we use the [MAE codebase](https://github.com/facebookresearch/mae).

## License
See the [LICENSE](./LICENSE) file for details about the license under which this code is made available.

## Citation
If you find this repository useful in your research, please consider giving a star :star: and a citation
```
@article{assran2022masked,
  title={Masked Siamese Networks for Label-Efficient Learning}, 
  author={Assran, Mahmoud, and Caron, Mathilde, and Misra, Ishan, and Bojanowski, Piotr, and Bordes, Florian and Vincent, Pascal, and Joulin, Armand, and Rabbat, Michael, and Ballas, Nicolas},
  journal={arXiv preprint arXiv:2204.07141},
  year={2022}
}
```
