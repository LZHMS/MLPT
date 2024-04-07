# ------------------------------------------- RMaPLe Trainer ------------------------------------------
###### Model backbone ---- vit_b16
## Experiment 1 -- research the robustness of RMaPLe under different noise rates
# Conditions: 2 n_context, 9 learning depth, 16 shots, False GCE
CUDA_VISIBLE_DEVICES=1 bash scripts/rmaple/main.sh RMaPLe dtd vit_b16_c2_ep50_batch4 2 9 16 False 0
CUDA_VISIBLE_DEVICES=1 bash scripts/rmaple/main.sh RMaPLe dtd vit_b16_c2_ep50_batch4 2 9 16 False 2
CUDA_VISIBLE_DEVICES=1 bash scripts/rmaple/main.sh RMaPLe dtd vit_b16_c2_ep50_batch4 2 9 16 False 4
CUDA_VISIBLE_DEVICES=1 bash scripts/rmaple/main.sh RMaPLe dtd vit_b16_c2_ep50_batch4 2 9 16 False 8

## Experiment 2 -- research the robustness of RMaPLe with GCE under different noise rates
# Conditions: 2 n_context, 9 learning depth, 16 shots, True GCE
CUDA_VISIBLE_DEVICES=1 bash scripts/rmaple/main.sh RMaPLe dtd vit_b16_c2_ep50_batch4 2 9 16 True 0
CUDA_VISIBLE_DEVICES=1 bash scripts/rmaple/main.sh RMaPLe dtd vit_b16_c2_ep50_batch4 2 9 16 True 2
CUDA_VISIBLE_DEVICES=1 bash scripts/rmaple/main.sh RMaPLe dtd vit_b16_c2_ep50_batch4 2 9 16 True 4
CUDA_VISIBLE_DEVICES=1 bash scripts/rmaple/main.sh RMaPLe dtd vit_b16_c2_ep50_batch4 2 9 16 True 8

## Experiments Analysis
# Experiment 1
CUDA_VISIBLE_DEVICES=1 bash scripts/rmaple/parse_results.sh RMaPLe dtd vit_b16_c2_ep50_batch4 2 9 16 False 0
CUDA_VISIBLE_DEVICES=1 bash scripts/rmaple/parse_results.sh RMaPLe dtd vit_b16_c2_ep50_batch4 2 9 16 False 2
CUDA_VISIBLE_DEVICES=1 bash scripts/rmaple/parse_results.sh RMaPLe dtd vit_b16_c2_ep50_batch4 2 9 16 False 4
CUDA_VISIBLE_DEVICES=1 bash scripts/rmaple/parse_results.sh RMaPLe dtd vit_b16_c2_ep50_batch4 2 9 16 False 8

# Experiment 2
CUDA_VISIBLE_DEVICES=1 bash scripts/rmaple/parse_results.sh RMaPLe dtd vit_b16_c2_ep50_batch4 2 9 16 True 0
CUDA_VISIBLE_DEVICES=1 bash scripts/rmaple/parse_results.sh RMaPLe dtd vit_b16_c2_ep50_batch4 2 9 16 True 2
CUDA_VISIBLE_DEVICES=1 bash scripts/rmaple/parse_results.sh RMaPLe dtd vit_b16_c2_ep50_batch4 2 9 16 True 4
CUDA_VISIBLE_DEVICES=1 bash scripts/rmaple/parse_results.sh RMaPLe dtd vit_b16_c2_ep50_batch4 2 9 16 True 8


CUDA_VISIBLE_DEVICES=1 bash scripts/rmaple/main.sh RMaPLe caltech101 vit_b16_c2_ep50_batch4 2 9 16 True 2