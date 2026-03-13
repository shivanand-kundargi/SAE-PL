import os
import os.path as osp
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


class SAEConceptProjector(nn.Module):
    """Projects pretrained SAE sparse features into prompt-shaped modulations.

    Encodes ViT activations through a frozen SAE encoder, pools across tokens,
    then projects directly to [B, n_ctx, ctx_dim] for additive prompt fusion.
    Also computes a concentration loss on top-k SAE features.
    """

    def __init__(self, sae_path, n_ctx, ctx_dim, prompt_depth, hidden_dim=256):
        super().__init__()
        self.n_ctx = n_ctx
        self.ctx_dim = ctx_dim  # 512 (text dim)
        self.prompt_depth = prompt_depth

        # Load pretrained SAE checkpoint
        checkpoint = torch.load(sae_path, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        d_in = state_dict["W_enc"].shape[0]
        d_sae = state_dict["W_enc"].shape[1]
        self.d_sae = d_sae

        # Register SAE encoder weights as frozen buffers
        self.register_buffer("W_enc", state_dict["W_enc"])       # (d_in, d_sae)
        self.register_buffer("b_enc", state_dict["b_enc"])       # (d_sae,)
        self.register_buffer("b_dec", state_dict["b_dec"])       # (d_in,)
        # Project pooled SAE features through bottleneck to prompt shape
        self.ctx_proj = nn.Sequential(
            nn.Linear(d_sae, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_ctx * ctx_dim),
        )

        total_params = sum(p.numel() for p in self.ctx_proj.parameters())
        print(f"SAEConceptProjector: d_in={d_in}, d_sae={d_sae}, "
              f"bottleneck: {d_sae} -> {hidden_dim} -> {n_ctx}x{ctx_dim} "
              f"({total_params:,} params)")

    def forward(self, vit_activations, return_stats=False):
        """
        Args:
            vit_activations: ViT intermediate activations [B, seq_len, d_in]

        Returns:
            h: [B, n_ctx, ctx_dim] prompt-shaped SAE modulation
            alignment_loss: scalar concentration loss
            (optional) sae_stats, feature_acts if return_stats=True
        """
        # SAE encode (frozen)
        x = vit_activations.float()
        sae_in = x - self.b_dec
        hidden_pre = torch.einsum("bsd,dk->bsk", sae_in, self.W_enc) + self.b_enc
        feature_acts = torch.relu(hidden_pre)  # [B, seq_len, d_sae]

        # Mean-pool across sequence
        pooled = feature_acts.mean(dim=1)  # [B, d_sae]

        # Project to prompt shape: [B, n_ctx * ctx_dim] -> [B, n_ctx, ctx_dim]
        h = self.ctx_proj(pooled)  # [B, n_ctx * ctx_dim]
        h = h.view(h.shape[0], self.n_ctx, self.ctx_dim)  # [B, n_ctx, ctx_dim]

        if return_stats:
            with torch.no_grad():
                sparsity = (feature_acts == 0).float().mean().item()
                active_per_token = (feature_acts > 0).float().sum(dim=-1).mean().item()
                h_norm = h.norm(dim=-1).mean().item()
                pooled_norm = pooled.norm(dim=-1).mean().item()

            sae_stats = {
                "sae_sparsity": sparsity,
                "sae_active_features": active_per_token,
                "sae_h_norm": h_norm,
                "sae_pooled_norm": pooled_norm,
            }
            return h, sae_stats, feature_acts.detach()
        return h



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

    n_ctx = cfg.TRAINER.MAPLE.N_CTX
    design_details = {"trainer": 'MaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": n_ctx}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        combined = [x, compound_prompts_deeper_text, 0]
        outputs = self.transformer(combined)
        x = outputs[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.MAPLE.N_CTX
        self.use_sae = cfg.TRAINER.MAPLE.USE_SAE
        ctx_init = cfg.TRAINER.MAPLE.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg.TRAINER.MAPLE.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.TRAINER.MAPLE.PROMPT_DEPTH
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        if self.use_sae:
            print(f"SAE enabled: h-based fusion (n_ctx={n_ctx} unchanged)")

        self.proj = nn.Linear(ctx_dim, 768)
        self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)

        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, sae_h=None):
        ctx = self.ctx  # [n_ctx, 512]

        if sae_h is not None:
            # sae_h: [n_ctx, ctx_dim] — batch-averaged SAE projection
            ctx = ctx + sae_h.type(ctx.dtype)  # [n_ctx, ctx_dim]

        # Project shallow ctx to 768-dim for vision
        shared_ctx = self.proj(ctx)  # [n_ctx, 768]

        # Expand for text prompts
        ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # [n_cls, n_ctx, 512]

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        # Build deep prompts (text and vision)
        visual_deep_prompts = []
        text_deep_prompts = []

        for index, layer in enumerate(self.compound_prompt_projections):
            text_prompt = self.compound_prompts_text[index]  # [n_ctx, 512]
            vision_prompt = layer(text_prompt)               # [n_ctx, 768]

            text_deep_prompts.append(text_prompt)
            visual_deep_prompts.append(vision_prompt)

        return prompts, shared_ctx, text_deep_prompts, visual_deep_prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # SAE integration
        self.use_sae = cfg.TRAINER.MAPLE.USE_SAE
        if self.use_sae:
            ctx_dim = clip_model.ln_final.weight.shape[0]
            self.sae_projector = SAEConceptProjector(
                sae_path=cfg.TRAINER.MAPLE.SAE_PATH,
                n_ctx=cfg.TRAINER.MAPLE.N_CTX,
                ctx_dim=ctx_dim,
                prompt_depth=cfg.TRAINER.MAPLE.PROMPT_DEPTH,
                hidden_dim=cfg.TRAINER.MAPLE.SAE_HIDDEN_DIM,
            )
            # Register hook on ViT at the SAE's target layer
            self.image_encoder.register_sae_hook(cfg.TRAINER.MAPLE.SAE_LAYER)

    def forward(self, image, label=None, return_sae_info=False):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        sae_stats = None
        sae_feature_acts = None

        if self.use_sae:
            # Stage 1: Run normal MaPLe forward to capture ViT activations via hook
            prompts, shared_ctx, deep_text, deep_vision = self.prompt_learner()
            with torch.no_grad():
                _ = self.image_encoder(image.type(self.dtype), shared_ctx, deep_vision)
            hooked_acts = self.image_encoder._hooked_activations  # [B, seq_len, 768]

            # SAE projection: get h [B, n_ctx, ctx_dim]
            if return_sae_info:
                h, sae_stats, sae_feature_acts = self.sae_projector(hooked_acts, return_stats=True)
            else:
                h = self.sae_projector(hooked_acts)

            # Average h over batch: [B, n_ctx, ctx_dim] -> [n_ctx, ctx_dim]
            h_mean = h.mean(dim=0)  # [n_ctx, ctx_dim]

            # Stage 2: Re-run with SAE-modulated prompts
            prompts, shared_ctx, deep_text, deep_vision = self.prompt_learner(sae_h=h_mean)
            text_features = self.text_encoder(prompts, tokenized_prompts, deep_text)
            image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_vision)
        else:
            # Original MaPLe forward (no SAE)
            prompts, shared_ctx, deep_text, deep_vision = self.prompt_learner()
            text_features = self.text_encoder(prompts, tokenized_prompts, deep_text)
            image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_vision)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        if self.prompt_learner.training:
            loss = F.cross_entropy(logits, label)
            if return_sae_info:
                if sae_stats is not None:
                    sae_stats["ce_loss"] = loss.item()
                return loss, sae_stats, sae_feature_acts
            return loss

        return logits


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@TRAINER_REGISTRY.register()
class MaPLe(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.MAPLE.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.MAPLE.PREC == "fp32" or cfg.TRAINER.MAPLE.PREC == "amp":
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        names_to_update = ["prompt_learner"]
        if cfg.TRAINER.MAPLE.USE_SAE:
            names_to_update.append("sae_projector")

        for name, param in self.model.named_parameters():
            if not any(n in name for n in names_to_update):
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("MultiModalPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.MAPLE.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        use_sae = self.cfg.TRAINER.MAPLE.USE_SAE
        should_log_sae = use_sae and (self.batch_idx % self.cfg.TRAIN.PRINT_FREQ == 0)
        should_save_sae = use_sae and ((self.batch_idx + 1) == self.num_batches)

        prec = self.cfg.TRAINER.MAPLE.PREC
        if prec == "amp":
            with autocast():
                if should_log_sae or should_save_sae:
                    loss, sae_stats, sae_feature_acts = model(image, label, return_sae_info=True)
                else:
                    loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            if should_log_sae or should_save_sae:
                loss, sae_stats, sae_feature_acts = model(image, label, return_sae_info=True)
            else:
                loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        # SAE logging
        if should_log_sae and sae_stats is not None:
            print(f"  [SAE] sparsity={sae_stats['sae_sparsity']:.4f}, "
                  f"active={sae_stats['sae_active_features']:.1f}, "
                  f"ce={sae_stats['ce_loss']:.4f}")
            loss_summary.update(sae_stats)

        # SAE activation saving (at end of each epoch)
        if should_save_sae and sae_feature_acts is not None:
            save_dir = osp.join(self.cfg.OUTPUT_DIR, "sae_activations")
            os.makedirs(save_dir, exist_ok=True)
            save_path = osp.join(save_dir, f"sae_acts_epoch{self.epoch + 1}.pt")
            torch.save({
                "feature_acts": sae_feature_acts.cpu(),
                "sae_stats": sae_stats,
                "epoch": self.epoch + 1,
            }, save_path)
            print(f"  [SAE] Saved activations to {save_path}")

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def resume_model_if_exist(self, directory):
        """Override to use strict=False when loading checkpoints."""
        names = self.get_model_names()
        file_missing = False

        for name in names:
            path = osp.join(directory, name)
            if not osp.exists(path):
                file_missing = True
                break

        if file_missing:
            print("No checkpoint found, train from scratch")
            return 0

        print(f"Found checkpoint at {directory} (will resume training)")

        for name in names:
            path = osp.join(directory, name)
            with open(osp.join(path, "checkpoint"), "r") as f:
                model_name = f.readlines()[0].strip("\n")
                fpath = osp.join(path, model_name)

            print(f'Loading checkpoint from "{fpath}"')
            checkpoint = load_checkpoint(fpath)

            missing, unexpected = self._models[name].load_state_dict(
                checkpoint["state_dict"], strict=False
            )
            if missing:
                print(f"  Missing keys (ok if SAE config changed): {missing}")
            if unexpected:
                print(f"  Unexpected keys (ok if SAE config changed): {unexpected}")
            print("Loaded model weights")

            if self._optims[name] is not None and "optimizer" in checkpoint:
                try:
                    self._optims[name].load_state_dict(checkpoint["optimizer"])
                    print("Loaded optimizer")
                except ValueError:
                    print("Optimizer state mismatch (SAE config changed), resetting optimizer")

            if self._scheds[name] is not None and "scheduler" in checkpoint:
                self._scheds[name].load_state_dict(checkpoint["scheduler"])
                print("Loaded scheduler")

            start_epoch = checkpoint["epoch"]
            print(f"Previous epoch: {start_epoch}")

        return start_epoch

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

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

            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)
