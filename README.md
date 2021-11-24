# A ConvNet for the 2020s

This is a PyTorch implementation for the anonymous CVPR 2022 submission "A ConvNet for the 2020s" (Paper ID: 123).

## Setup
Please install [Pytorch](https://pytorch.org/)>=1.8.0, [torchvision](https://pytorch.org/vision/stable/index.html)>=0.9.0 following official instructions. For example:
```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Clone the repo and install required packages:
```
git clone https://github.com/anonymous20222022/convnext
cd convnext
pip install -r requirements.txt
```

## Dataset Preparation

Download and extract ImageNet-1k from http://image-net.org/.

The directory structure is the standard layout of torchvision's [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder). The training and validation data are expected to be in the `train/` folder and `val` folder, respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

## Training on ImageNet-1k (image classification)

To reproduce the results in the "modernization" of ResNet (Section 2) on ImageNet-1K, please run the following commands, one for each step change. 

The commands use [submitit](https://github.com/facebookincubator/submitit) for submitting jobs to SLURM clusters. You may need to change cluster specific arguments in `run_with_submitit.py`. If you wish to run on a single machine, replace the first line of each command with 
```bash
python -m torch.distributed.launch --nproc_per_node=8 run_classification.py --data_path IMNET_PATH --update_freq 4 \
```

Please set `JOB_DIR` to a desired folder name under `checkpoint/` for saving the log and checkpoints, and `IMNET_PATH` to path to ImageNet-1K.

Original ResNet-50:
```bash
python run_with_submitit.py --job_dir JOB_DIR --data_path IMNET_PATH --nodes 4 \
--model resnet --batch_size 128  --drop_path 0.1
```

Change stage ratio:
```bash
python run_with_submitit.py --job_dir JOB_DIR --data_path IMNET_PATH --nodes 4 \
--model resnet --batch_size 128  --drop_path 0.1 \
--resnet_layers 3-3-9-3
```

"Patchify" stem:
```bash
python run_with_submitit.py --job_dir JOB_DIR --data_path IMNET_PATH --nodes 4 \
--model resnet --batch_size 128  --drop_path 0.1 \
--resnet_layers 3-3-9-3 --resnet_stem patch_embed
```

Depthwise Conv:
```bash
python run_with_submitit.py --job_dir JOB_DIR --data_path IMNET_PATH --nodes 4 \
--model resnet --batch_size 128  --drop_path 0.1 \
--resnet_layers 3-3-9-3 --resnet_stem patch_embed --resnet_depthwise true
```

Increase width:
```bash
python run_with_submitit.py --job_dir JOB_DIR --data_path IMNET_PATH --nodes 4 \
--model resnet --batch_size 128  --drop_path 0.1 \
--resnet_layers 3-3-9-3 --resnet_stem patch_embed --resnet_depthwise true \
--resnet_embed_dim 96
```

Inverting dimensions:
```bash
python run_with_submitit.py --job_dir JOB_DIR --data_path IMNET_PATH --nodes 4 \
--model resnet --batch_size 128  --drop_path 0.1 \
--resnet_layers 3-3-9-3 --resnet_stem patch_embed --resnet_depthwise true \
--resnet_embed_dim 96 --resnet_block inverted
```

Move up depthwise conv: 
```bash
python run_with_submitit.py --job_dir JOB_DIR --data_path IMNET_PATH --nodes 4 \
--model resnet --batch_size 128  --drop_path 0.1 \
--resnet_layers 3-3-9-3 --resnet_stem patch_embed --resnet_depthwise true \
--resnet_embed_dim 96 --resnet_block inverted  --resnet_conv_first true
```

Kernel size --> 7, or change `--resnet_kernel_size` for other values:
```bash
python run_with_submitit.py --job_dir JOB_DIR --data_path IMNET_PATH --nodes 4 \
--model resnet --batch_size 128  --drop_path 0.1 \
--resnet_layers 3-3-9-3 --resnet_stem patch_embed --resnet_depthwise true \
--resnet_embed_dim 96 --resnet_block inverted  --resnet_conv_first true \
--resnet_kernel_size 7
```

ReLU --> GELU:
```bash
python run_with_submitit.py --job_dir JOB_DIR --data_path IMNET_PATH --nodes 4 \
--model resnet --batch_size 128  --drop_path 0.1 \
--resnet_layers 3-3-9-3 --resnet_stem patch_embed --resnet_depthwise true \
--resnet_embed_dim 96 --resnet_block inverted  --resnet_conv_first true \
--resnet_kernel_size 7  --resnet_act_layer gelu
```

Fewer activations:
```bash
python run_with_submitit.py --job_dir JOB_DIR --data_path IMNET_PATH --nodes 4 \
--model resnet --batch_size 128  --drop_path 0.1 \
--resnet_layers 3-3-9-3 --resnet_stem patch_embed --resnet_depthwise true \
--resnet_embed_dim 96 --resnet_block inverted  --resnet_conv_first true \
--resnet_kernel_size 7  --resnet_act_layer gelu --resnet_del_act true
```

Fewer norms:
```bash
python run_with_submitit.py --job_dir JOB_DIR --data_path IMNET_PATH --nodes 4 \
--model resnet --batch_size 128  --drop_path 0.1 \
--resnet_layers 3-3-9-3 --resnet_stem patch_embed --resnet_depthwise true \
--resnet_embed_dim 96 --resnet_block inverted  --resnet_conv_first true \
--resnet_kernel_size 7  --resnet_act_layer gelu --resnet_del_act true \
--resnet_reorg_norm true
```

BN --> LN:
```bash
python run_with_submitit.py --job_dir JOB_DIR --data_path IMNET_PATH --nodes 4 \
--model resnet --batch_size 128  --drop_path 0.1 \
--resnet_layers 3-3-9-3 --resnet_stem patch_embed --resnet_depthwise true \
--resnet_embed_dim 96 --resnet_block inverted  --resnet_conv_first true \
--resnet_kernel_size 7  --resnet_act_layer gelu  --resnet_del_act true \
--resnet_reorg_norm true  --resnet_norm_layer ln
```

Final norm:
```bash
python run_with_submitit.py --job_dir JOB_DIR --data_path IMNET_PATH --nodes 4 \
--model resnet --batch_size 128  --drop_path 0.1 \
--resnet_layers 3-3-9-3 --resnet_stem patch_embed --resnet_depthwise true \
--resnet_embed_dim 96 --resnet_block inverted  --resnet_conv_first true \
--resnet_kernel_size 7  --resnet_act_layer gelu  --resnet_del_act true \
--resnet_reorg_norm true  --resnet_norm_layer ln  --resnet_final_norm true 
```

Seperate downsampling convs (or equivalently, ConvNeXt-T):
```bash
python run_with_submitit.py --job_dir JOB_DIR --data_path IMNET_PATH --nodes 4 \
--model resnet --batch_size 128  --drop_path 0.1 \
--resnet_layers 3-3-9-3 --resnet_stem patch_embed --resnet_depthwise true \
--resnet_embed_dim 96 --resnet_block inverted  --resnet_conv_first true \
--resnet_kernel_size 7  --resnet_act_layer gelu  --resnet_del_act true \
--resnet_reorg_norm true  --resnet_norm_layer ln  --resnet_final_norm true \
--resnet_sep_downsample true
```

## Acknowledgement

This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) library and [BeiT](https://github.com/microsoft/unilm/tree/master/beit) repository.
