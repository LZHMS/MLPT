# for all datasets
CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp dtd rn50_ep50 end 16 16 False False 0 OneHot
CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp dtd rn50_ep50 end 16 16 False False 2 OneHot
CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp dtd rn50_ep50 end 16 16 False False 4 OneHot
CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp dtd rn50_ep50 end 16 16 False False 8 OneHot
CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp caltech101 rn50_ep50 end 16 16 False False 0 OneHot
CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp caltech101 rn50_ep50 end 16 16 False False 2 OneHot
CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp caltech101 rn50_ep50 end 16 16 False False 4 OneHot
CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp caltech101 rn50_ep50 end 16 16 False False 8 OneHot

CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp oxford_flowers rn50_ep50 end 16 16 False False 0 OneHot
CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp oxford_flowers rn50_ep50 end 16 16 False False 2 OneHot
CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp oxford_flowers rn50_ep50 end 16 16 False False 4 OneHot
CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp oxford_flowers rn50_ep50 end 16 16 False False 8 OneHot

CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp ucf101 rn50_ep50 end 16 16 False False 0 OneHot
CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp ucf101 rn50_ep50 end 16 16 False False 2 OneHot
CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp ucf101 rn50_ep50 end 16 16 False False 4 OneHot
CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp ucf101 rn50_ep50 end 16 16 False False 8 OneHot

CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp fgvc_aircraft rn50_ep50 end 16 16 False False 0 OneHot
CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp fgvc_aircraft rn50_ep50 end 16 16 False False 2 OneHot
CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp fgvc_aircraft rn50_ep50 end 16 16 False False 4 OneHot
CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp fgvc_aircraft rn50_ep50 end 16 16 False False 8 OneHot

CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp oxford_pets rn50_ep50 end 16 16 False False 0 OneHot
CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp oxford_pets rn50_ep50 end 16 16 False False 2 OneHot
CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp oxford_pets rn50_ep50 end 16 16 False False 4 OneHot
CUDA_VISIBLE_DEVICES=0 bash scripts/rcoop/main.sh RCoOp oxford_pets rn50_ep50 end 16 16 False False 8 OneHot