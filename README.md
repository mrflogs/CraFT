<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> Connecting the Dots: Collaborative Fine-tuning for Black-Box Vision-Language Models </h1>

<p align='center' style="text-align:center;font-size:1.25em;">
    <a href="https://zhengbo.wang/" target="_blank" style="text-decoration: none;">Zhengbo Wang<sup>1,2</sup></a>&nbsp;,&nbsp;
    <a href="https://liangjian.xyz/" target="_blank" style="text-decoration: none;">Jian Liang<sup>2,3†</sup></a>&nbsp;,&nbsp;
    <a href="https://scholar.google.com/citations?user=ayrg9AUAAAAJ&hl=en" target="_blank" style="text-decoration: none;">Ran He<sup>2,3</sup></a>&nbsp;,&nbsp; 
    <a href="http://staff.ustc.edu.cn/~zlwang/index_en.html" target="_blank" style="text-decoration: none;">Zilei Wang<sup>1</sup></a>&nbsp;,&nbsp; 
	<a href="https://scholar.google.com/citations?user=W-FGd_UAAAAJ&hl=en" target="_blank" style="text-decoration: none;">Tieniu Tan<sup>2,3,4</sup></a>&nbsp;&nbsp;
	<br>
<sup>1</sup>University of Science and Technology of China&nbsp;&nbsp;&nbsp;
<sup>2</sup>NLPR & MAIS, Institute of Automation, Chinese Academy of Sciences&nbsp;&nbsp;&nbsp;
<sup>3</sup>School of Artificial Intelligence, University of Chinese Academy of Sciences&nbsp;&nbsp;&nbsp;
<sup>4</sup>Nanjing University 
</p>


<p align='center';>
<b>
<em>ICML, 2024</em> <br>
</b>
</p>
<p align='center' style="text-align:center;font-size:2.5 em;">
<b>
    <a href="https://arxiv.org/abs/2402.04050" target="_blank" style="text-decoration: none;">[Paper]</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://github.com/mrflogs/CraFT" target="_blank" style="text-decoration: none;">[Code]</a>
</b>
</p>


## Requirements
### Installation
Create a conda environment and install dependencies:
```
conda create -n carrot python=3.9
conda activate carrot

pip install -r requirements.txt

# Install the according versions of torch and torchvision
conda install pytorch torchvision cudatoolkit
```

### Dataset
Follow [DATASET.md](DATASET.md) to install ImageNet and other 10 datasets referring to CoOp.

## Get Started
### Configs
The running configurations can be modified in `carrot_configs/dataset.yaml`, including shot numbers, visual encoders, and hyperparamters. 

### Running
For ImageNet dataset:
```bash
CUDA_VISIBLE_DEVICES=0 python main_carrot_imagenet.py --config carrot_configs/imagenet.yaml
```
For other 10 datasets:
```bash
CUDA_VISIBLE_DEVICES=0 python main_carrot.py --config carrot_configs/dataset.yaml
```

## Acknowledgement

This repo benefits from [CLIP](https://github.com/openai/CLIP), [CoOp](https://github.com/KaiyangZhou/CoOp), [SHIP](https://github.com/mrflogs/SHIP), and [H2B](https://github.com/mrflogs/ICLR24). Thanks for their wonderful work.

## Citation
```latex
@inproceedings{wang2024craft,
  title={Connecting the Dots: Collaborative Fine-tuning for Black-Box Vision-Language Models},
  author={Wang, Zhengbo and Liang, Jian and He, Ran and Wang, Zilei and Tan, Tieniu},
  booktitle={Proceedings of International Conference on Machine Learning (ICML)},
  year={2024}
}
```

## Contact

If you have any question, feel free to contact 📫zhengbowang@mail.ustc.edu.cn.
