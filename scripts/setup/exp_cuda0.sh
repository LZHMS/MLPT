: '
# running coop
bash scripts/setup/exp_coop_dataset.sh oxford_flowers 0
bash scripts/setup/exp_coop_dataset.sh caltech101 0
bash scripts/setup/exp_coop_dataset.sh ucf101 0
'
# running rcoop
CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp caltech101 rn50_ep50 end 16 16 False True 0
CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp caltech101 rn50_ep50 end 16 16 False True 2
CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp caltech101 rn50_ep50 end 16 16 False True 4
CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp caltech101 rn50_ep50 end 16 16 False True 8

CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp oxford_flowers rn50_ep50 end 16 16 False True 0
CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp oxford_flowers rn50_ep50 end 16 16 False True 2
CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp oxford_flowers rn50_ep50 end 16 16 False True 4
CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp oxford_flowers rn50_ep50 end 16 16 False True 8

CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp ucf101 rn50_ep50 end 16 16 False True 0
CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp ucf101 rn50_ep50 end 16 16 False True 2
CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp ucf101 rn50_ep50 end 16 16 False True 4
CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp ucf101 rn50_ep50 end 16 16 False True 8

CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp caltech101 vit_b32_ep50 end 16 16 False True 0
CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp caltech101 vit_b32_ep50 end 16 16 False True 2
CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp caltech101 vit_b32_ep50 end 16 16 False True 4
CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp caltech101 vit_b32_ep50 end 16 16 False True 8

CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp oxford_flowers vit_b32_ep50 end 16 16 False True 0
CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp oxford_flowers vit_b32_ep50 end 16 16 False True 2
CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp oxford_flowers vit_b32_ep50 end 16 16 False True 4
CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp oxford_flowers vit_b32_ep50 end 16 16 False True 8

CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp ucf101 vit_b32_ep50 end 16 16 False True 0
CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp ucf101 vit_b32_ep50 end 16 16 False True 2
CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp ucf101 vit_b32_ep50 end 16 16 False True 4
CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp ucf101 vit_b32_ep50 end 16 16 False True 8