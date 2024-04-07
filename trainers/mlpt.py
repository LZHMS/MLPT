import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
import time
import datetime
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from datasets.data_manager import COOPDataManager

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'MLPT',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class GeneralizedCrossEntropy(nn.Module):
    """Computes the generalized cross-entropy loss, from `
    "Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels"
    <https://arxiv.org/abs/1805.07836>`_
    Args:
        q: Box-Cox transformation parameter, :math:`\in (0,1]`
    Shape:
        - Input: the raw, unnormalized score for each class.
                tensor of size :math:`(minibatch, C)`, with C the number of classes
        - Target: the labels, tensor of size :math:`(minibatch)`, where each value
                is :math:`0 \leq targets[i] \leq C-1`
        - Output: scalar
    """

    def __init__(self, q: float = 0.7) -> None:
        super().__init__()
        self.q = q
        self.epsilon = 1e-6
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self.softmax(input)
        p = p[torch.arange(p.shape[0]), target]
        # Avoid undefined gradient for p == 0 by adding epsilon
        p += self.epsilon
        loss = (1 - p ** self.q) / self.q
        return torch.mean(loss)
    

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.MLPT.N_CTX
        ctx_init = cfg.TRAINER.MLPT.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.MLPT.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.MLPT.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.GCE_loss = GeneralizedCrossEntropy(q=0.5)

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
 
        return logits


@TRAINER_REGISTRY.register()
class MLPT(TrainerX):
    """Context Optimization (MLPT).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.MLPT.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.MLPT.PREC == "fp32" or cfg.TRAINER.MLPT.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP (Two CLIP models)")
        self.model1 = CustomCLIP(cfg, classnames, clip_model)
        self.model2 = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model1.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        for name, param in self.model2.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model1.prompt_learner, cfg.MODEL.INIT_WEIGHTS)
            load_pretrained_weights(self.model2.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model1.to(self.device)
        self.model2.to(self.device)

        # NOTE: only give prompt_learner to the optimizer
        self.optim1 = build_optimizer(self.model1.prompt_learner, cfg.OPTIM)
        self.sched1 = build_lr_scheduler(self.optim1, cfg.OPTIM)
        self.register_model("prompt_learner1", self.model1.prompt_learner, self.optim1, self.sched1)

        # NOTE: only give prompt_learner to the optimizer
        self.optim2 = build_optimizer(self.model2.prompt_learner, cfg.OPTIM)
        self.sched2 = build_lr_scheduler(self.optim2, cfg.OPTIM)
        self.register_model("prompt_learner2", self.model2.prompt_learner, self.optim2, self.sched2)

        self.scaler1 = GradScaler() if cfg.TRAINER.MLPT.PREC == "amp" else None
        self.scaler2 = GradScaler() if cfg.TRAINER.MLPT.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model1 = nn.DataParallel(self.model1)
            self.model2 = nn.DataParallel(self.model2)

    def forward_backward(self, batch, model, optim, scaler, modelname):
        image, label, _ = self.parse_batch_train(batch)
        
        #print([image.shape, label.shape])
        prec = self.cfg.TRAINER.MLPT.PREC
        if prec == "amp":
            with autocast():
                output = model(image)
                loss = F.cross_entropy(output, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            output = model(image)
            if self.cfg.DATASET.USE_ROBUSTLOSS:
                loss = model.GCE_loss(output, label)
            else:
                loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss, modelname)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr(modelname)

        return loss_summary

    def training_forward_backward(self, batch_x, batch_u, model, optim, scaler, modelname):
        image_x, label_x, _ = self.parse_batch_train(batch_x)
        image_u, label_u, _ = self.parse_batch_train(batch_u)
        
        #print([image.shape, label.shape])
        prec = self.cfg.TRAINER.MLPT.PREC
        if prec == "amp":
            with autocast():
                output_x, output_u = model(image_x), model(image_u)
                loss_x, loss_u = F.cross_entropy(output_x, label_x), F.cross_entropy(output_u, label_u)
            
            loss = torch.cat([loss_x, loss_u], dim=0)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            output_x, output_u = model(image_x), model(image_u)
            if self.cfg.DATASET.USE_ROBUSTLOSS:
                loss_x, loss_u = model.GCE_loss(output_x, label_x), model.GCE_loss(output_u, label_u)
            else:
                probs_u = torch.softmax(output_u, dim=1)
                Lx = -torch.mean(torch.sum(F.log_softmax(output_x, dim=1) * label_x, dim=1))
                Lu = torch.mean((probs_u - label_u)**2)
                loss = Lx + Lu
            self.model_backward_and_update(loss, modelname)

        _, label_x_id = torch.max(label_x, 1)
        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output_x, label_x_id)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr(modelname)

        return loss_summary
    
    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dm = COOPDataManager(self.cfg)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm

    def parse_batch_train(self, batch):
        inputs = batch["img"]
        label = batch["label"]
        wx = batch["wx"]
        inputs = inputs.to(self.device)
        label = label.to(self.device)
        return inputs, label, wx

    def parse_batch_with_index(self, batch):
        inputs = batch["img"]
        label = batch["label"]
        index = batch["index"]
        inputs = inputs.to(self.device)
        label = label.to(self.device)
        index = index.to(self.device)
        return inputs, label, index

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    def train(self):
        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            
            if self.epoch < self.cfg.WARMUP.EPOCH:
                print("Warmup CLIP1")
                self.warmup_epoch(self.model1, self.optim1, self.scaler1, "prompt_learner1")
                print("Warmup CLIP2")
                self.warmup_epoch(self.model2, self.optim2, self.scaler2, "prompt_learner2")
            else:
                prob1, self.all_loss_m1 = self.eval_train(self.model1, "prompt_learner1")
                prob2, self.all_loss_m2 = self.eval_train(self.model2, "prompt_learner2")
                
                pred1 = (prob1 > self.cfg.WARMUP.THRESHOLD)      
                pred2 = (prob2 > self.cfg.WARMUP.THRESHOLD)
                
                print('Train CLIP1')
                labeled_trainloader, unlabeled_trainloader = self.dm.split_data_loader(pred2, prob2) # co-divide
                self.run_clips_epoch(self.model1, self.model2, "prompt_learner1", "prompt_learner2", self.optim1, self.scaler1, labeled_trainloader, unlabeled_trainloader)

                print('Train CLIP2')
                labeled_trainloader, unlabeled_trainloader = self.dm.split_data_loader(pred1, prob1) # co-divide
                self.run_clips_epoch(self.model2, self.model1, "prompt_learner2", "prompt_learner1", self.optim2, self.scaler2, labeled_trainloader, unlabeled_trainloader)
                # self.update_labels()
                
                if (self.epoch+1) % 10 == 0:
                    loss_data = {'clip1_loss': np.squeeze(self.all_loss_m1.cpu().numpy()),
                                 'clip1_prob': np.squeeze(prob1),
                                 'clip2_loss': np.squeeze(self.all_loss_m2.cpu().numpy()),
                                 'clip2_prob': np.squeeze(prob2)}
                    loss_data_df = pd.DataFrame(loss_data, columns=['clip1_loss', 'clip1_prob', 'clip2_loss', 'clip2_prob'])

                    output_layers_dir = os.path.join(self.cfg.OUTPUT_DIR, 'ClipLoss')  # 获取layers子目录路径
                    if not os.path.exists(output_layers_dir):
                        os.makedirs(output_layers_dir)
                    loss_data_df.to_excel(f'{output_layers_dir}/clip_{self.epoch+1}.xlsx', index=False)
                
            self.after_epoch()
            
        self.after_train_clips()

    def warmup_epoch(self, model, optim, scaler, modelname):
        self.set_model_mode("train", modelname)
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch, model, optim, scaler, modelname)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def run_clips_epoch(self, train_model, eval_model, train_model_name, eval_model_name, optim, scaler, labeled_trainloader, unlabeled_trainloader):
        self.set_model_mode("train", train_model_name)
        self.set_model_mode("eval", eval_model_name)
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(labeled_trainloader)

        end = time.time()
        unlabeled_train_iter = iter(unlabeled_trainloader)  
        for self.batch_idx, batch_x in enumerate(labeled_trainloader):
            data_time.update(time.time() - end)
            try:
                batch_u = unlabeled_train_iter.next()
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                batch_u = unlabeled_train_iter.next()

            inputs_x, labels_x, wx = self.parse_batch_train(batch_x)
            inputs_u, _ , _ = self.parse_batch_train(batch_u)

            batch_size = inputs_x.size(0)
            # Transform label to one-hot
            labels_x = torch.zeros(batch_size, self.num_classes).scatter_(1, labels_x.view(-1,1).cpu(), 1) 
            
            with torch.no_grad():
                # label refinement of labeled samples
                outputs_x1 = train_model(inputs_x)
                outputs_x2 = eval_model(inputs_x)

                px = (torch.softmax(outputs_x1, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
                #print([px.shape, wx.shape, labels_x.shape])
                wx = wx.reshape(-1, 1).to(self.device)
                # label refinement
                px = wx * labels_x.to(self.device) + (1 - wx) * px
                # no label refinement
                # px = labels_x.to(self.device)

                ptx = px**(1 / self.cfg.TRAINER.MLPT.TEM) # temparature sharpening 
                        
                targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
                targets_x = targets_x.detach()  

                # co-guessing for unlabeled samples
                outputs_u1 = train_model(inputs_u)
                outputs_u2 = eval_model(inputs_u)

                pu = (torch.softmax(outputs_u1, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2       
                ptu = pu**(1 / self.cfg.TRAINER.MLPT.TEM)    # temparature sharpening

                targets_u = ptu / ptu.sum(dim=1, keepdim=True)   # normalize
                targets_u = targets_u.detach()

            # mixmatch
            l = np.random.beta(self.cfg.TRAINER.MLPT.ALPHA, self.cfg.TRAINER.MLPT.ALPHA)        
            l = max(l, 1-l)
                    
            #all_inputs = torch.cat([inputs_x, inputs_u], dim=0)
            #all_targets = torch.cat([targets_x, targets_u], dim=0)

            '''
            # using for MixMatch
            idx = torch.randperm(all_inputs.size(0))
            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]
            
            mixed_input = l * input_a + (1 - l) * input_b        
            mixed_target = l * target_a + (1 - l) * target_b
            
            _, mixed_target = torch.max(mixed_target, 1)

            mixed_inputs = {'img': mixed_input, 'label': mixed_target, 'wx': None}
            loss_summary = self.forward_backward(mixed_inputs, train_model, optim, scaler, train_model_name)
            '''
            '''
            # using for no MixMatch
            _, all_target_id = torch.max(all_targets, 1)
            all_inputs_labels = {'img': all_inputs, 'label': all_target_id, 'wx': None}     
            loss_summary = self.forward_backward(all_inputs_labels, train_model, optim, scaler, train_model_name)
            '''
            '''
            _, targets_x_id = torch.max(targets_x, 1)
            _, targets_u_id = torch.max(targets_u, 1)
            targets_x_onehot = torch.zeros(targets_x_id.size(0), self.num_classes).scatter_(1, targets_x_id.view(-1,1).cpu(), 1)
            targets_u_onehot = torch.zeros(targets_u_id.size(0), self.num_classes).scatter_(1, targets_u_id.view(-1,1).cpu(), 1)
            '''
            inputs_labels_x = {'img': inputs_x, 'label': targets_x, 'wx': None}
            inputs_labels_u = {'img': inputs_u, 'label': targets_u, 'wx': None}

            loss_summary = self.training_forward_backward(inputs_labels_x, inputs_labels_u, train_model, optim, scaler, train_model_name)


            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def after_train_clips(self):
        print("Finish training")

        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")
            self.test_clips()

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        # Close writer
        self.close_writer()

    @torch.no_grad()
    def test_clips(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval", ["prompt_learner1", "prompt_learner2"])
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            inputs, label = self.parse_batch_test(batch)
            outputs1, outputs2 = self.model1(inputs), self.model2(inputs)
            outputs = outputs1 + outputs2

            self.evaluator.process(outputs, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
    
    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar"
                )

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)
    
    def eval_train(self, model, modelname):
        self.set_model_mode("eval", modelname)
        data_loader = self.val_loader

        print(f"Do warmup evaluation on the few_shots_data set")
        
        losses_id = []
        CE = nn.CrossEntropyLoss(reduction='none')
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader)):
                inputs, labels, index = self.parse_batch_with_index(batch)
                output = model(inputs)
                loss = CE(output, labels)
                for b in range(inputs.size(0)):
                    losses_id.append([index[b], loss[b]])

        losses = torch.zeros(len(losses_id))
        for unit in losses_id:
            losses[unit[0]] = unit[1]
        
        losses = (losses - losses.min()) / (losses.max() - losses.min())

        input_loss = losses.reshape(-1, 1)

        # fit a two-component GMM to the loss
        gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss) 
        prob = prob[:,gmm.means_.argmin()]         
        return prob, input_loss

    def update_labels(self):
        self.set_model_mode("eval", ["prompt_learner1", "prompt_learner2"])
        self.evaluator.reset()

        data_loader = self.val_loader

        print(f"Update lables on the *val* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            inputs, labels = self.parse_batch_test(batch)
            outputs1, outputs2 = self.model1(inputs), self.model2(inputs)
            px = (torch.softmax(outputs1, dim=1) + torch.softmax(outputs2, dim=1)) / 2

            # Transform label to one-hot
            labels_x = torch.zeros(inputs.shape[0], self.num_classes).scatter_(1, labels.view(-1,1).cpu(), 1)
            
            wx = 0.5
            # label refinement
            px = wx * labels_x.to(self.device) + (1 - wx) * px

            ptx = px**(1 / self.cfg.TRAINER.MLPT.TEM) # temparature sharpening
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()
    
            _, new_labels = torch.max(targets_x, 1)
            
            batch["label"] = new_labels
            
