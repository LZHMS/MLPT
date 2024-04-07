# Mutual Learning Between Two Prompt Tuning Mechanisms for Vision-Language Models

## Introduction
Large-scale vision-language models such as CLIP, CoOP, CoCoOP, MaPLE, etc., are capable of aligning representations of a wide range of natural images with their textual descriptions. They have demonstrated exceptional abilities to efficiently tackle various tasks with data. Research findings suggest that self-supervised learning itself exhibits a certain level of robustness to noise. However, this robustness does not necessarily enable models to adapt autonomously to downstream datasets. Therefore, we explore the mutual learning of vision-language models to further enhance their robustness to noise.

## How to Install
+ Setup conda environment:
```bash
############ Conda Environment Installation ############

# Fetch the miniconda script
export HOME=$PWD
wget -q https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh -O miniconda.sh
sh miniconda.sh -b -p $HOME/miniconda3
rm miniconda.sh
export PATH=$HOME/miniconda3/bin:$PATH

# Initialize conda
source $HOME/miniconda3/etc/profile.d/conda.sh
hash -r
conda config --set always_yes yes --set changeps1 yes
conda create -n dassl python=3.7
conda activate dassl
```
+ Install the the awesome toolbox [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch)
```bash
############ Dassl Installation ############

# Clone the Dassl repository
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/

# Activate the existed environment
conda activate dassl

# Install the required dependencies
pip install -r requirements.txt

# Install PyTorch (version 1.11.0 or above) and torchvision
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch

# Set up the Dassl library (No need to rebuild even if the source code changes)
python setup.py develop
```
+ Install the MLPT
```bash
############ MLPT Installation ############

# Navigate back to the parent directory
cd ..

# Clone the MLPT repository
git clone https://github.com/LZHMS/MLPT.git
cd RMaPLe/

# Install necessary packages for CLIP
pip install -r requirements.txt

######## Note ########
# Two symbolic links, `data` and `dassl`, exist in the repository. It is recommended that these be pointed to locations with sufficient storage capacity.

rm data dassl # remove the existing links
ln -s ${your_data_path} ./data
ln -s ${your_dassl_path} ./dassl

# Installation complete
```
## Datasets

Please follow [CoOp Datasets Instructions](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) to install the datasets.

## Model Intuitive
<img src="https://cdn.jsdelivr.net/gh/LZHMS/picx-images-hosting@master/Profile/Baoyan/intuitive.lvibpg3b8.webp" alt="intuitive" />

## Model Piplines
<img src="https://cdn.jsdelivr.net/gh/LZHMS/picx-images-hosting@master/Profile/Baoyan/pipline.2doh6lycre.webp" alt="pipline" />

## Algorithm
<img src="https://cdn.jsdelivr.net/gh/LZHMS/picx-images-hosting@master/Profile/Baoyan/image.101y2kpzxe.webp" alt="algorithm" />


## Results			
<img src="https://cdn.jsdelivr.net/gh/LZHMS/picx-images-hosting@master/Profile/Baoyan/image.wic4z17h4.webp" alt="image" />

### Raw Materials
Model traning logs can be found in the `log.txt` under each experiment directory.

Parsing results can be found in the following files:

+ [MLPT RN50_EP50_16SHOTS ON Dtd](./output/dtd/MLPT/rn50_ep50_16shots/parse_results.txt)
+ [MLPT VT16_EP50_16SHOTS ON Dtd](./output/dtd/MLPT/vit_b16_ep50_16shots/parse_results.txt)
+ [MLPT RN50_EP50_16SHOTS ON Caltech101](./output/caltech101/MLPT/rn50_ep50_16shots/parse_results.txt)
+ [MLPT VT16_EP50_16SHOTS ON Caltech101](./output/caltech101/MLPT/vit_b16_ep50_16shots/parse_results.txt)
+ [MLPT RN50_EP50_16SHOTS ON Fgvc_aircraft](./output/fgvc_aircraft/MLPT/rn50_ep50_16shots/parse_results.txt)
+ [MLPT VT16_EP50_16SHOTS ON Fgvc_aircraft](./output/fgvc_aircraft/MLPT/vit_b16_ep50_16shots/parse_results.txt)
+ [MLPT RN50_EP50_16SHOTS ON Oxford_flowers](./output/oxford_flowers/MLPT/rn50_ep50_16shots/parse_results.txt)
+ [MLPT VT16_EP50_16SHOTS ON Oxford_flowers](./output/oxford_flowers/MLPT/vit_b16_ep50_16shots/parse_results.txt)
+ [MLPT RN50_EP50_16SHOTS ON Oxford_pets](./output/oxford_pets/MLPT/rn50_ep50_16shots/parse_results.txt)
+ [MLPT VT16_EP50_16SHOTS ON Oxford_pets](./output/oxford_pets/MLPT/vit_b16_ep50_16shots/parse_results.txt)
+ [MLPT RN50_EP50_16SHOTS ON Ucf101](./output/ucf101/MLPT/rn50_ep50_16shots/parse_results.txt)
+ [MLPT VT16_EP50_16SHOTS ON Ucf101](./output/ucf101/MLPT/vit_b16_ep50_16shots/parse_results.txt)

## Conclusions
By contrasting experimental results, we observe that our proposed mutual learning mechanism enables vision-language models to achieve higher accuracy across various noise scenarios. This enhanced mutual learning paradigm ensures more efficient model adaptation to downstream tasks.

## References
+ [CoOp](https://github.com/KaiyangZhou/CoOp)
+ [multimodal-prompt-learning](https://github.com/muzairkhattak/multimodal-prompt-learning)
