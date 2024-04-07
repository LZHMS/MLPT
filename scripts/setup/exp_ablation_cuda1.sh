# for all datasets

CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp dtd vit_b32_ep50 end 16 16 False False 0 OneHot
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp dtd vit_b32_ep50 end 16 16 False False 2 OneHot
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp dtd vit_b32_ep50 end 16 16 False False 4 OneHot
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp dtd vit_b32_ep50 end 16 16 False False 8 OneHot

CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp caltech101 vit_b32_ep50 end 16 16 False False 0 OneHot
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp caltech101 vit_b32_ep50 end 16 16 False False 2 OneHot
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp caltech101 vit_b32_ep50 end 16 16 False False 4 OneHot
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp caltech101 vit_b32_ep50 end 16 16 False False 8 OneHot

CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp oxford_flowers vit_b32_ep50 end 16 16 False False 0 OneHot
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp oxford_flowers vit_b32_ep50 end 16 16 False False 2 OneHot
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp oxford_flowers vit_b32_ep50 end 16 16 False False 4 OneHot
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp oxford_flowers vit_b32_ep50 end 16 16 False False 8 OneHot

CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp ucf101 vit_b32_ep50 end 16 16 False False 0 OneHot
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp ucf101 vit_b32_ep50 end 16 16 False False 2 OneHot
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp ucf101 vit_b32_ep50 end 16 16 False False 4 OneHot
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp ucf101 vit_b32_ep50 end 16 16 False False 8 OneHot

CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp fgvc_aircraft vit_b32_ep50 end 16 16 False False 0 OneHot
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp fgvc_aircraft vit_b32_ep50 end 16 16 False False 2 OneHot
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp fgvc_aircraft vit_b32_ep50 end 16 16 False False 4 OneHot
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp fgvc_aircraft vit_b32_ep50 end 16 16 False False 8 OneHot

CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp oxford_pets vit_b32_ep50 end 16 16 False False 0 OneHot
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp oxford_pets vit_b32_ep50 end 16 16 False False 2 OneHot
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp oxford_pets vit_b32_ep50 end 16 16 False False 4 OneHot
CUDA_VISIBLE_DEVICES=1 bash scripts/rcoop/main.sh RCoOp oxford_pets vit_b32_ep50 end 16 16 False False 8 OneHot