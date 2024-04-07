: '
# running coop
bash scripts/setup/exp_coop_dataset.sh dtd 1
bash scripts/setup/exp_coop_dataset.sh fgvc_aircraft 1
bash scripts/setup/exp_coop_dataset.sh oxford_pets 1
'
# running rcoop
: '
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp dtd rn50_ep50 end 16 16 False True 0
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp dtd rn50_ep50 end 16 16 False True 2
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp dtd rn50_ep50 end 16 16 False True 4
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp dtd rn50_ep50 end 16 16 False True 8

CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp fgvc_aircraft rn50_ep50 end 16 16 False True 0
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp fgvc_aircraft rn50_ep50 end 16 16 False True 2
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp fgvc_aircraft rn50_ep50 end 16 16 False True 4
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp fvc_aircraft rn50_ep50 end 16 16 False True 8

CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp oxford_pets rn50_ep50 end 16 16 False True 0
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp oxford_pets rn50_ep50 end 16 16 False True 2
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp oxford_pets rn50_ep50 end 16 16 False True 4
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp oxford_pets rn50_ep50 end 16 16 False True 8

CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp dtd vit_b32_ep50 end 16 16 False True 0
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp dtd vit_b32_ep50 end 16 16 False True 2
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp dtd vit_b32_ep50 end 16 16 False True 4
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp dtd vit_b32_ep50 end 16 16 False True 8

CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp fgvc_aircraft vit_b32_ep50 end 16 16 False True 0
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp fgvc_aircraft vit_b32_ep50 end 16 16 False True 2
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp fgvc_aircraft vit_b32_ep50 end 16 16 False True 4
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp fgvc_aircraft vit_b32_ep50 end 16 16 False True 8

CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp oxford_pets vit_b32_ep50 end 16 16 False True 0
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp oxford_pets vit_b32_ep50 end 16 16 False True 2
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp oxford_pets vit_b32_ep50 end 16 16 False True 4
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp oxford_pets vit_b32_ep50 end 16 16 False True 8
'
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp ucf101 rn50_ep50 end 16 16 False True 2
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp ucf101 rn50_ep50 end 16 16 False True 4
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp ucf101 rn50_ep50 end 16 16 False True 8

CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp caltech101 vit_b32_ep50 end 16 16 False True 0
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp caltech101 vit_b32_ep50 end 16 16 False True 2
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp caltech101 vit_b32_ep50 end 16 16 False True 4
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp caltech101 vit_b32_ep50 end 16 16 False True 8

CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp oxford_flowers vit_b32_ep50 end 16 16 False True 0
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp oxford_flowers vit_b32_ep50 end 16 16 False True 2
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp oxford_flowers vit_b32_ep50 end 16 16 False True 4
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp oxford_flowers vit_b32_ep50 end 16 16 False True 8

CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp ucf101 vit_b32_ep50 end 16 16 False True 0
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp ucf101 vit_b32_ep50 end 16 16 False True 2
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp ucf101 vit_b32_ep50 end 16 16 False True 4
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp ucf101 vit_b32_ep50 end 16 16 False True 8