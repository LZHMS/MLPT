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
### Combination Between CoOP and GCE					
| Dataset|Backbone| Method|Noise Rate|Noise Rate|Noise Rate|Noise Rate|			
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
||||0| 12.5|25	|50|
|Dtd|RN50|CoOP|63.13|59.40|54.83|44.80| 
|Dtd|RN50|CoOP(CLS)|63.87|57.17 |	49.77 |	33.63|
|Dtd |RN50|RCoOP(+GCE)	|62.87 |	60.80 |	**61.13** |	**56.27**| 
|Dtd|RN50|CoOP(CLS+GCE)	|**63.67** |	**61.90** |	58.37 	|46.30|
|Dtd|VT16|CoOP|69.27|63.73|60.53|48.77| 
|Dtd|VT16|CoOP(CLS)|69.20|61.47 |	54.47 |	36.97|
|Dtd |VT16|RCoOP(+GCE)	|**70.07** |	**67.93**|	**67.50** |	**61.50**| 
|Dtd|VT16|CoOP(CLS+GCE)	|68.30|	66.87 |	64.00 	|52.50|
|Caltech-101|RN50|CoOP|91.93|86.40|83.17|76.80|
|Caltech-101|RN50|CoOP(CLS)|91.37|76.93|65.87|46.07|
|Caltech-101|RN50|RCoOP(+GCE)|**92.03**|**91.53**|**91.43**|**87.83**|
|Caltech-101|RN50|CoOP(CLS+GCE)|91.53|89.87|84.37|69.90|
|Caltech-101|VT16|CoOP|95.70|92.43|89.70|84.67|
|Caltech-101|VT16|CoOP(CLS)|94.80|81.50|71.73|52.63|
|Caltech-101|VT16|RCoOP(+GCE)|**95.80**|**95.47**|**95.63**|**91.43**|
|Caltech-101|VT16|CoOP(CLS+GCE)|95.23|92.43|88.40|73.80|

### Comparation Between MaPLe and RMaPLe
|Dataset| Method|Noise Rate|Noise Rate|Noise Rate|Noise Rate|
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|Dtd|MaPLe|70.70|62.93|55.43|39.73|
|Dtd|MaPLe+GCE|**71.53**|**69.63**|**67.77**|59.33|
|Dtd|RMaPLe|69.20|66.93|64.87|55.90|
|Dtd|RMaPLe+GCE|68.47|68.83|66.90|**61.10**|


### Raw Materials
Model traning logs can be found in the `log.txt` under each experiment directory.

Parsing results can be found in the following files:
+ [CoOP RN50_EP50_16SHOTS ON Caltech-101](./output/caltech101/CoOp/rn50_ep50_16shots/parse_results.txt)
+ [CoOP VT16_EP50_16SHOTS ON Caltech-101](./output/caltech101/CoOp/vit_b16_ep50_16shots/parse_results.txt)
+ [CoOP RN50_EP50_16SHOTS ON Dtd](./output/dtd/CoOp/rn50_ep50_16shots/parse_results.txt)
+ [CoOP VT16_EP50_16SHOTS ON Dtd](./output/dtd/CoOp/vit_b16_ep50_16shots/parse_results.txt)
+ [MaPLe VT16_EP50_16SHOTS ON Dtd](./output/dtd/MaPLe/vit_b16_c2_ep50_batch4_16shots/parse_results.txt)
+ [RMaPLe VT16_EP50_16SHOTS ON Dtd](./output/dtd/RMaPLe/vit_b16_c2_ep50_batch4_16shots/parse_results.txt)


## Conclusions
### CoOP and RCoOP
The factors influencing robustness in prompt learning within Visual-Language Models(VLMs), as well as directions for improvement, can be inferred from the experiments above:

+ Loss Function
By analyzing the experiments above, comparing the Generalized Cross Entropy (GCE) loss function with the Cross Entropy (CE) loss function in terms of robustness to noise, it can be concluded that GCE significantly enhances model robustness. This suggests that optimizing loss function design under prompt learning may lead to stronger robustness and transferability in downstream tasks. So further exploration is required in loss function design.

+ Class Specific Context
From the experiments, it is evident that the transfer performance of the `dtd` dataset varies across different backbone structures, indicating that the design of specific category prompts relies on both the dataset and model architecture. Therefore, there is no unified criterion for whether to select specific category prompts or not.

+ Backbone
From the results, it can be analyzed that the ViT-B/16 architecture consistently achieves higher accuracy compared to ResNet50 when used as the backbone visual encoder. This suggests that the ViT-B/16 structure helps to resist label noise and enhance the robustness of the model.

### MaPLe and RMaPLe
Based on the experimental results of CoOP and RCoOP, we further optimize the loss function design and adopt unified curriculum learning. We conduct experiments using ViT16 as the backbone network and derive corresponding conclusions:
+ Comparing the experimental results above, our improved RMaPLe shows significantly greater improvement over MaPLe at different noise ratios, indicating its stronger noise robustness;
+ Additionally, further experimentation reveals that the incorporation of the GCE loss function leads to significant improvements in both cases. However, MaPLe seems to perform better in low-noise scenarios, while RMaPLe demonstrates the ability to withstand more severe noise.

## References
+ [CoOp](https://github.com/KaiyangZhou/CoOp)
+ [multimodal-prompt-learning](https://github.com/muzairkhattak/multimodal-prompt-learning)
