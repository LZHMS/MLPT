# ------------------------------------------- MLPT Trainer ------------------------------------------
###### Model backbone ---- vit_b32
## -- research the robustness of MLPT under different noise rates
### Experiment 1: On dtd
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/main.sh MLPT dtd vit_b32_ep50 end 16 16 False False 0 BiLoss
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/main.sh MLPT dtd vit_b32_ep50 end 16 16 False False 2 BiLoss
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/main.sh MLPT dtd vit_b32_ep50 end 16 16 False False 4 BiLoss
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/main.sh MLPT dtd vit_b32_ep50 end 16 16 False False 8 BiLoss

### Experiment 2: On caltech101
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/main.sh MLPT caltech101 vit_b32_ep50 end 16 16 False False 0 BiLoss
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/main.sh MLPT caltech101 vit_b32_ep50 end 16 16 False False 2 BiLoss
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/main.sh MLPT caltech101 vit_b32_ep50 end 16 16 False False 4 BiLoss
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/main.sh MLPT caltech101 vit_b32_ep50 end 16 16 False False 8 BiLoss

### Experiment 3: On oxford_flowers
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/main.sh MLPT oxford_flowers vit_b32_ep50 end 16 16 False False 0 BiLoss
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/main.sh MLPT oxford_flowers vit_b32_ep50 end 16 16 False False 2 BiLoss
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/main.sh MLPT oxford_flowers vit_b32_ep50 end 16 16 False False 4 BiLoss
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/main.sh MLPT oxford_flowers vit_b32_ep50 end 16 16 False False 8 BiLoss

### Experiment 4: On ucf101
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/main.sh MLPT ucf101 vit_b32_ep50 end 16 16 False False 0 BiLoss
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/main.sh MLPT ucf101 vit_b32_ep50 end 16 16 False False 2 BiLoss
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/main.sh MLPT ucf101 vit_b32_ep50 end 16 16 False False 4 BiLoss
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/main.sh MLPT ucf101 vit_b32_ep50 end 16 16 False False 8 BiLoss

### Experiment 5: On fgvc_aircraft
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/main.sh MLPT fgvc_aircraft vit_b32_ep50 end 16 16 False False 0 BiLoss
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/main.sh MLPT fgvc_aircraft vit_b32_ep50 end 16 16 False False 2 BiLoss
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/main.sh MLPT fgvc_aircraft vit_b32_ep50 end 16 16 False False 4 BiLoss
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/main.sh MLPT fgvc_aircraft vit_b32_ep50 end 16 16 False False 8 BiLoss

### Experiment 6: On oxford_pets
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/main.sh MLPT oxford_pets vit_b32_ep50 end 16 16 False False 0 BiLoss
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/main.sh MLPT oxford_pets vit_b32_ep50 end 16 16 False False 2 BiLoss
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/main.sh MLPT oxford_pets vit_b32_ep50 end 16 16 False False 4 BiLoss
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/main.sh MLPT oxford_pets vit_b32_ep50 end 16 16 False False 8 BiLoss

## Experiments Analysis
### Experiment 1
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/parse_results.sh MLPT dtd vit_b32_ep50 end 16 16 False False 0 BiLoss
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/parse_results.sh MLPT dtd vit_b32_ep50 end 16 16 False False 2 BiLoss
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/parse_results.sh MLPT dtd vit_b32_ep50 end 16 16 False False 4 BiLoss
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/parse_results.sh MLPT dtd vit_b32_ep50 end 16 16 False False 8 BiLoss

### Experiment 2
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/parse_results.sh MLPT caltech101 vit_b32_ep50 end 16 16 False False 0 BiLoss
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/parse_results.sh MLPT caltech101 vit_b32_ep50 end 16 16 False False 2 BiLoss
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/parse_results.sh MLPT caltech101 vit_b32_ep50 end 16 16 False False 4 BiLoss
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/parse_results.sh MLPT caltech101 vit_b32_ep50 end 16 16 False False 8 BiLoss

### Experiment 3
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/parse_results.sh MLPT oxford_flowers vit_b32_ep50 end 16 16 False False 0 BiLoss
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/parse_results.sh MLPT oxford_flowers vit_b32_ep50 end 16 16 False False 2 BiLoss
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/parse_results.sh MLPT oxford_flowers vit_b32_ep50 end 16 16 False False 4 BiLoss
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/parse_results.sh MLPT oxford_flowers vit_b32_ep50 end 16 16 False False 8 BiLoss

### Experiment 4
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/parse_results.sh MLPT ucf101 vit_b32_ep50 end 16 16 False False 0 BiLoss
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/parse_results.sh MLPT ucf101 vit_b32_ep50 end 16 16 False False 2 BiLoss
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/parse_results.sh MLPT ucf101 vit_b32_ep50 end 16 16 False False 4 BiLoss
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/parse_results.sh MLPT ucf101 vit_b32_ep50 end 16 16 False False 8 BiLoss

### Experiment 5
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/parse_results.sh MLPT fgvc_aircraft vit_b32_ep50 end 16 16 False False 0 BiLoss
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/parse_results.sh MLPT fgvc_aircraft vit_b32_ep50 end 16 16 False False 2 BiLoss
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/parse_results.sh MLPT fgvc_aircraft vit_b32_ep50 end 16 16 False False 4 BiLoss
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/parse_results.sh MLPT fgvc_aircraft vit_b32_ep50 end 16 16 False False 8 BiLoss

### Experiment 6
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/parse_results.sh MLPT oxford_pets vit_b32_ep50 end 16 16 False False 0 BiLoss
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/parse_results.sh MLPT oxford_pets vit_b32_ep50 end 16 16 False False 2 BiLoss
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/parse_results.sh MLPT oxford_pets vit_b32_ep50 end 16 16 False False 4 BiLoss
CUDA_VISIBLE_DEVICES=1 bash scripts/mlpt/parse_results.sh MLPT oxford_pets vit_b32_ep50 end 16 16 False False 8 BiLoss