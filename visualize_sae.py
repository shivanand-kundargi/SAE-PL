"""
Comprehensive SAE + MaPLe visualization script.

Generates 6 visualizations:
1. Projection analysis — ctx_proj bottleneck weight analysis (per-token norms, feature importance)
2. Modulation magnitude — ||SAE h|| vs ||learned ctx|| per token (requires data)
3. Per-class SAE feature activations heatmap
4. Prompt space t-SNE with vs without SAE modulation
5. SAE feature activation sparsity histogram
6. Concept attribution — interpretable SAE feature → prompt token decomposition

Usage:
    cd /p/lustre1/kundargi1/multimodal-prompt-learning
    PYTHONPATH=/p/lustre1/kundargi1/Dassl.pytorch:$PYTHONPATH python3 visualize_sae.py \
        --model-dir output/base2new_sae/train_base/fgvc_aircraft/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx/seed0 \
        --output-dir viz_output
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from dassl.utils import setup_logger, set_random_seed
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# Import custom datasets and trainers (same as train.py)
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet
import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r
import datasets.domainnet

import trainers.coop
import trainers.cocoop
import trainers.zsclip
import trainers.maple
import trainers.independentVL
import trainers.vpt


def extend_cfg(cfg):
    """Same as train.py extend_cfg — register all custom config fields."""
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16
    cfg.TRAINER.COOP.CSC = False
    cfg.TRAINER.COOP.CTX_INIT = ""
    cfg.TRAINER.COOP.PREC = "fp16"
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16
    cfg.TRAINER.COCOOP.CTX_INIT = ""
    cfg.TRAINER.COCOOP.PREC = "fp16"

    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = 2
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"
    cfg.TRAINER.MAPLE.PREC = "fp16"
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 9
    cfg.TRAINER.MAPLE.USE_SAE = False
    cfg.TRAINER.MAPLE.SAE_PATH = ""
    cfg.TRAINER.MAPLE.SAE_LAYER = -2
    cfg.TRAINER.MAPLE.SAE_HIDDEN_DIM = 256
    cfg.TRAINER.MAPLE.SAE_ALIGN_WEIGHT = 0.1

    cfg.TRAINER.IVLP = CN()
    cfg.TRAINER.IVLP.N_CTX_VISION = 2
    cfg.TRAINER.IVLP.N_CTX_TEXT = 2
    cfg.TRAINER.IVLP.CTX_INIT = "a photo of a"
    cfg.TRAINER.IVLP.PREC = "fp16"
    cfg.TRAINER.IVLP.PROMPT_DEPTH_VISION = 9
    cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT = 9

    cfg.TRAINER.VPT = CN()
    cfg.TRAINER.VPT.N_CTX_VISION = 2
    cfg.TRAINER.VPT.CTX_INIT = "a photo of a"
    cfg.TRAINER.VPT.PREC = "fp16"
    cfg.TRAINER.VPT.PROMPT_DEPTH_VISION = 1

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # Dataset config
    cfg.merge_from_file(args.dataset_config_file)
    # Trainer config
    cfg.merge_from_file(args.config_file)

    # Override with CLI args
    cfg.DATASET.ROOT = args.data_root
    cfg.DATASET.NUM_SHOTS = 16
    cfg.DATASET.SUBSAMPLE_CLASSES = "base"
    cfg.TRAINER.NAME = "MaPLe"
    cfg.TRAINER.MAPLE.USE_SAE = True
    cfg.TRAINER.MAPLE.SAE_PATH = args.sae_path
    cfg.OUTPUT_DIR = args.model_dir
    cfg.SEED = 0

    cfg.freeze()
    return cfg


def load_model(cfg, model_dir, epoch=5):
    """Build trainer and load checkpoint using the trainer's own load_model."""
    set_random_seed(cfg.SEED)
    trainer = build_trainer(cfg)

    # Use the trainer's load_model (same path as train.py --eval-only)
    trainer.load_model(model_dir, epoch=epoch)
    trainer.model.eval()
    trainer.model.cuda()
    return trainer


def viz1_projection_analysis(model, output_dir):
    """Analyze the ctx_proj bottleneck weights: how SAE features map to prompt tokens."""
    print("\n=== Viz 1: Projection Weight Analysis ===")
    proj = model.sae_projector

    # ctx_proj is Sequential: Linear(d_sae, hidden) -> GELU -> Linear(hidden, n_ctx*ctx_dim)
    W_in = proj.ctx_proj[0].weight.detach().cpu()   # [hidden_dim, d_sae]
    W_out = proj.ctx_proj[2].weight.detach().cpu()   # [n_ctx*ctx_dim, hidden_dim]
    n_ctx = proj.n_ctx
    ctx_dim = proj.ctx_dim
    d_sae = proj.d_sae
    hidden_dim = W_in.shape[0]

    # Per-SAE-feature importance: norm of input projection columns
    feature_importance = W_in.norm(dim=0).numpy()  # [d_sae]

    # Per-token output weight norms
    W_out_per_token = W_out.view(n_ctx, ctx_dim, hidden_dim)
    token_norms = W_out_per_token.norm(dim=(1, 2)).numpy()  # [n_ctx]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Per-token output projection weight norm
    axes[0].bar(range(n_ctx), token_norms, color='#3498db', alpha=0.8, edgecolor='black')
    axes[0].set_xlabel("Prompt Token Index")
    axes[0].set_ylabel("Weight Norm")
    axes[0].set_title(f"Per-Token Output Weight Norm (n_ctx={n_ctx})")

    # Panel 2: Top SAE feature importance (input layer, sorted)
    top_k = min(50, d_sae)
    sorted_idx = np.argsort(feature_importance)[::-1][:top_k]
    axes[1].bar(range(top_k), feature_importance[sorted_idx], color='#e74c3c', alpha=0.8)
    axes[1].set_xlabel(f"SAE Feature (top {top_k} by input weight norm)")
    axes[1].set_ylabel("Weight Norm")
    axes[1].set_title(f"Top SAE Feature Importance (d_sae={d_sae})")
    axes[1].set_xticks(range(0, top_k, max(1, top_k // 10)))
    axes[1].set_xticklabels([str(sorted_idx[i]) for i in range(0, top_k, max(1, top_k // 10))],
                             rotation=45, fontsize=7)

    # Panel 3: Bottleneck hidden unit activation norms (via input weights)
    hidden_norms = W_in.norm(dim=1).numpy()  # [hidden_dim]
    axes[2].bar(range(hidden_dim), sorted(hidden_norms, reverse=True),
                color='#2ecc71', alpha=0.8, edgecolor='black')
    axes[2].set_xlabel(f"Bottleneck Unit (sorted, hidden_dim={hidden_dim})")
    axes[2].set_ylabel("Input Weight Norm")
    axes[2].set_title("Bottleneck Unit Importance")

    plt.tight_layout()
    path = os.path.join(output_dir, "1_projection_analysis.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")
    print(f"  Per-token norms: {token_norms}")
    print(f"  Top-5 SAE features: {sorted_idx[:5]} (norms: {feature_importance[sorted_idx[:5]]})")


def viz2_modulation_magnitude(trainer, output_dir, max_batches=20):
    """Compare ||SAE modulation h|| vs ||learned prompt ctx|| per token."""
    print("\n=== Viz 2: Modulation Magnitude Ratio ===")
    model = trainer.model
    model.eval()
    pl = model.prompt_learner
    proj = model.sae_projector

    # Learned prompt norms per token
    ctx = pl.ctx.detach().cpu()  # [n_ctx, ctx_dim]
    ctx_token_norms = ctx.norm(dim=1).numpy()  # [n_ctx]
    n_ctx = ctx.shape[0]

    # Collect SAE modulation norms from real data
    loader = trainer.test_loader
    h_accum = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            images = batch["img"].cuda()
            prompts, shared_ctx, deep_text, deep_vision = pl()
            _ = model.image_encoder(images.type(model.dtype), shared_ctx, deep_vision)
            hooked_acts = model.image_encoder._hooked_activations
            h = proj(hooked_acts)  # [B, n_ctx, ctx_dim]
            h_accum.append(h.cpu())

    h_all = torch.cat(h_accum, dim=0)  # [N, n_ctx, ctx_dim]
    h_mean = h_all.mean(dim=0)  # [n_ctx, ctx_dim]
    h_token_norms = h_mean.norm(dim=1).numpy()  # [n_ctx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Per-token comparison
    x = np.arange(n_ctx)
    width = 0.35
    axes[0].bar(x - width/2, ctx_token_norms, width, label='Learned ctx', color='#3498db')
    axes[0].bar(x + width/2, h_token_norms, width, label='SAE h (mean)', color='#e74c3c')
    axes[0].set_xlabel("Prompt Token Index")
    axes[0].set_ylabel("Norm")
    axes[0].set_title("Learned Prompt vs SAE Modulation per Token")
    axes[0].legend()
    for i in range(n_ctx):
        if ctx_token_norms[i] > 0:
            ratio = h_token_norms[i] / ctx_token_norms[i]
            axes[0].text(i, max(ctx_token_norms[i], h_token_norms[i]) * 1.05,
                         f'{ratio:.2f}x', ha='center', fontsize=9)

    # Panel 2: Distribution of per-sample h norms
    per_sample_norms = h_all.norm(dim=2).numpy()  # [N, n_ctx]
    axes[1].boxplot([per_sample_norms[:, i] for i in range(n_ctx)],
                     labels=[f"t{i}" for i in range(n_ctx)])
    axes[1].set_xlabel("Prompt Token Index")
    axes[1].set_ylabel("SAE h Norm")
    axes[1].set_title("SAE Modulation Norm Distribution per Token")

    plt.tight_layout()
    path = os.path.join(output_dir, "2_modulation_magnitude.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")
    print(f"  Learned ctx norms: {ctx_token_norms}")
    print(f"  SAE h norms (mean): {h_token_norms}")


def viz3_per_class_features(trainer, output_dir, top_k=50, max_batches=50):
    """Heatmap of top SAE feature activations per class."""
    print("\n=== Viz 3: Per-Class SAE Feature Activations ===")
    model = trainer.model
    model.eval()

    classnames = trainer.dm.dataset.classnames
    class_activations = defaultdict(list)

    loader = trainer.test_loader
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            images = batch["img"].cuda()
            labels = batch["label"]

            # Run through image encoder to get hooked activations
            prompts, shared_ctx, deep_text, deep_vision = model.prompt_learner()
            _ = model.image_encoder(images.type(model.dtype), shared_ctx, deep_vision)
            hooked_acts = model.image_encoder._hooked_activations

            # SAE encode
            x = hooked_acts.float()
            sae_in = x - model.sae_projector.b_dec
            hidden_pre = torch.einsum("bsd,dk->bsk", sae_in, model.sae_projector.W_enc) + model.sae_projector.b_enc
            feature_acts = torch.relu(hidden_pre)
            pooled = feature_acts.mean(dim=1)  # [B, d_sae]

            for i in range(len(labels)):
                cls_idx = labels[i].item()
                class_activations[cls_idx].append(pooled[i].cpu())

    # Average per class
    class_mean_acts = {}
    for cls_idx, acts_list in class_activations.items():
        class_mean_acts[cls_idx] = torch.stack(acts_list).mean(dim=0)

    if len(class_mean_acts) == 0:
        print("  No data collected, skipping.")
        return

    sorted_classes = sorted(class_mean_acts.keys())
    all_acts = torch.stack([class_mean_acts[c] for c in sorted_classes])

    # Find top-k most variable features across classes
    feature_variance = all_acts.var(dim=0)
    _, top_features = feature_variance.topk(top_k)

    heatmap_data = all_acts[:, top_features].numpy()
    class_labels = [classnames[c][:25] for c in sorted_classes]

    fig, ax = plt.subplots(figsize=(16, max(8, len(class_labels) * 0.35)))
    sns.heatmap(heatmap_data, xticklabels=[f"F{f.item()}" for f in top_features],
                yticklabels=class_labels, cmap="viridis", ax=ax)
    ax.set_xlabel(f"Top-{top_k} Most Variable SAE Features")
    ax.set_ylabel("Class")
    ax.set_title("Per-Class SAE Feature Activations (top variable features)")
    plt.tight_layout()
    path = os.path.join(output_dir, "3_per_class_features.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    print(f"  Classes found: {len(sorted_classes)}, top features plotted: {top_k}")


def viz4_tsne_prompts(trainer, output_dir, max_batches=30):
    """t-SNE of text features (after text encoder) with vs without SAE, colored by class."""
    print("\n=== Viz 4: Prompt Space t-SNE ===")
    model = trainer.model
    model.eval()

    classnames = trainer.dm.dataset.classnames
    tokenized_prompts = model.tokenized_prompts

    with torch.no_grad():
        # --- Text features WITH SAE ---
        # Run a forward pass to get hooked activations for SAE
        loader = trainer.test_loader
        batch = next(iter(loader))
        images = batch["img"].cuda()

        # Full forward to trigger hook + SAE
        _ = model(images)
        hooked_acts = model.image_encoder._hooked_activations
        h = model.sae_projector(hooked_acts)  # [B, n_ctx, ctx_dim]
        h_mean = h.mean(dim=0)  # [n_ctx, ctx_dim]

        prompts_sae, shared_ctx_sae, deep_text_sae, deep_vision_sae = model.prompt_learner(sae_h=h_mean)
        text_features_sae = model.text_encoder(prompts_sae, tokenized_prompts, deep_text_sae)
        text_features_sae = text_features_sae / text_features_sae.norm(dim=-1, keepdim=True)
        text_features_sae = text_features_sae.cpu().numpy()

        # --- Text features WITHOUT SAE ---
        prompts_no, shared_ctx_no, deep_text_no, deep_vision_no = model.prompt_learner()
        text_features_no = model.text_encoder(prompts_no, tokenized_prompts, deep_text_no)
        text_features_no = text_features_no / text_features_no.norm(dim=-1, keepdim=True)
        text_features_no = text_features_no.cpu().numpy()

    n_classes = len(classnames)
    all_features = np.concatenate([text_features_sae, text_features_no], axis=0)

    if n_classes < 3:
        print("  Too few classes for t-SNE, skipping.")
        return

    perplexity = min(30, len(all_features) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embedded = tsne.fit_transform(all_features)

    fig, ax = plt.subplots(figsize=(12, 9))

    # Without SAE (faded x markers)
    ax.scatter(embedded[n_classes:, 0], embedded[n_classes:, 1],
               c=range(n_classes), cmap='tab20', alpha=0.3, s=60, marker='x',
               label='Without SAE')
    # With SAE (solid dots)
    ax.scatter(embedded[:n_classes, 0], embedded[:n_classes, 1],
               c=range(n_classes), cmap='tab20', alpha=0.9, s=80, marker='o',
               label='With SAE')

    # Arrows from without -> with
    for i in range(n_classes):
        ax.annotate("", xy=embedded[i], xytext=embedded[n_classes + i],
                     arrowprops=dict(arrowstyle="->", color="gray", alpha=0.4, lw=0.8))

    # Label a subset of classes
    step = max(1, n_classes // 15)
    for i in range(0, n_classes, step):
        ax.annotate(classnames[i][:15], embedded[i], fontsize=7, alpha=0.7)

    ax.set_title("t-SNE of Text Features: With vs Without SAE Modulation")
    ax.legend(fontsize=11)
    plt.tight_layout()
    path = os.path.join(output_dir, "4_tsne_prompts.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    # Also compute mean shift magnitude
    shifts = np.linalg.norm(text_features_sae - text_features_no, axis=1)
    print(f"  Mean feature shift (L2): {shifts.mean():.4f}, max: {shifts.max():.4f}")


def viz5_sparsity(trainer, output_dir, max_batches=50):
    """Histogram of SAE activation sparsity."""
    print("\n=== Viz 5: SAE Feature Activation Sparsity ===")
    model = trainer.model
    model.eval()

    loader = trainer.test_loader
    all_sparsities = []
    all_active_counts = []
    all_activation_values = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            images = batch["img"].cuda()

            prompts, shared_ctx, deep_text, deep_vision = model.prompt_learner()
            _ = model.image_encoder(images.type(model.dtype), shared_ctx, deep_vision)
            hooked_acts = model.image_encoder._hooked_activations

            x = hooked_acts.float()
            sae_in = x - model.sae_projector.b_dec
            hidden_pre = torch.einsum("bsd,dk->bsk", sae_in, model.sae_projector.W_enc) + model.sae_projector.b_enc
            feature_acts = torch.relu(hidden_pre)

            active_mask = (feature_acts > 0).float()
            sparsity = 1.0 - active_mask.mean(dim=-1)
            active_count = active_mask.sum(dim=-1)

            all_sparsities.extend(sparsity.cpu().flatten().tolist())
            all_active_counts.extend(active_count.cpu().flatten().tolist())

            nonzero = feature_acts[feature_acts > 0].cpu()
            if len(nonzero) > 10000:
                nonzero = nonzero[torch.randperm(len(nonzero))[:10000]]
            all_activation_values.extend(nonzero.tolist())

    d_sae = model.sae_projector.d_sae

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(all_sparsities, bins=50, color='#3498db', alpha=0.8, edgecolor='black')
    axes[0].set_xlabel("Sparsity (fraction of zero features)")
    axes[0].set_ylabel("Count (tokens)")
    mean_sp = np.mean(all_sparsities)
    axes[0].set_title(f"Token-Level Sparsity (mean={mean_sp:.4f})")
    axes[0].axvline(mean_sp, color='red', linestyle='--', label=f"mean={mean_sp:.4f}")
    axes[0].legend()

    axes[1].hist(all_active_counts, bins=50, color='#e74c3c', alpha=0.8, edgecolor='black')
    mean_ac = np.mean(all_active_counts)
    axes[1].set_xlabel(f"# Active Features (out of {d_sae})")
    axes[1].set_ylabel("Count (tokens)")
    axes[1].set_title(f"Active Features per Token (mean={mean_ac:.0f})")
    axes[1].axvline(mean_ac, color='blue', linestyle='--', label=f"mean={mean_ac:.0f}")
    axes[1].legend()

    axes[2].hist(all_activation_values, bins=100, color='#2ecc71', alpha=0.8, edgecolor='black')
    axes[2].set_xlabel("Activation Value")
    axes[2].set_ylabel("Count")
    mean_av = np.mean(all_activation_values)
    axes[2].set_title(f"Nonzero Activation Distribution (mean={mean_av:.3f})")
    axes[2].set_yscale('log')

    plt.suptitle("SAE Feature Activation Sparsity Analysis", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "5_sparsity.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    print(f"  Mean sparsity: {mean_sp:.4f}")
    print(f"  Mean active features: {mean_ac:.0f} / {d_sae}")


def viz6_concept_attribution(trainer, output_dir, top_k_features=10, max_batches=30):
    """Show interpretable concept decomposition of SAE-guided prompts.

    For each prompt token, identifies the top SAE features that contribute
    most to its modulation, then reveals what visual concepts those features
    respond to by showing per-class activation patterns.

    This is the key interpretability advantage over baseline MaPLe:
    SAE-guided prompts = learned ctx + sum of (concept_i * weight_i),
    where each concept_i is a specific, identifiable visual pattern.
    Baseline MaPLe prompts are opaque optimized vectors with no such decomposition.
    """
    print("\n=== Viz 6: Interpretable Concept Attribution ===")
    model = trainer.model
    model.eval()
    proj = model.sae_projector
    classnames = trainer.dm.dataset.classnames

    n_ctx = proj.n_ctx
    ctx_dim = proj.ctx_dim
    d_sae = proj.d_sae

    # --- Step 1: Identify top SAE features per prompt token via bottleneck ---
    # W_in: [hidden_dim, d_sae], W_out: [n_ctx*ctx_dim, hidden_dim]
    W_in = proj.ctx_proj[0].weight.detach().cpu().float()   # [hidden, d_sae]
    W_out = proj.ctx_proj[2].weight.detach().cpu().float()   # [n_ctx*ctx_dim, hidden]
    hidden_dim = W_in.shape[0]

    # Effective weight: approximate the linear mapping from SAE features to each token
    # W_eff[token] = W_out[token] @ W_in  (ignoring GELU nonlinearity for attribution)
    W_out_per_token = W_out.view(n_ctx, ctx_dim, hidden_dim)  # [n_ctx, ctx_dim, hidden]
    # For each token, compute importance of each SAE feature
    # importance[t, f] = ||W_out[t] @ W_in[:, f]||  (norm of contribution to token t from feature f)
    token_feature_importance = torch.zeros(n_ctx, d_sae)
    for t in range(n_ctx):
        # W_out_per_token[t]: [ctx_dim, hidden], W_in: [hidden, d_sae]
        W_eff_t = W_out_per_token[t] @ W_in  # [ctx_dim, d_sae]
        token_feature_importance[t] = W_eff_t.norm(dim=0)  # [d_sae]

    # Top-k features per token
    top_features_per_token = {}
    for t in range(n_ctx):
        vals, idxs = token_feature_importance[t].topk(top_k_features)
        top_features_per_token[t] = (idxs.numpy(), vals.numpy())

    # Union of all top features across tokens
    all_top_features = set()
    for t in range(n_ctx):
        all_top_features.update(top_features_per_token[t][0].tolist())
    all_top_features = sorted(all_top_features)

    # --- Step 2: Collect per-class SAE feature activations for top features ---
    class_activations = defaultdict(list)
    loader = trainer.test_loader

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            images = batch["img"].cuda()
            labels = batch["label"]

            prompts, shared_ctx, deep_text, deep_vision = model.prompt_learner()
            _ = model.image_encoder(images.type(model.dtype), shared_ctx, deep_vision)
            hooked_acts = model.image_encoder._hooked_activations

            # SAE encode
            x = hooked_acts.float()
            sae_in = x - proj.b_dec
            hidden_pre = torch.einsum("bsd,dk->bsk", sae_in, proj.W_enc) + proj.b_enc
            feature_acts = torch.relu(hidden_pre)
            pooled = feature_acts.mean(dim=1)  # [B, d_sae]

            for i in range(len(labels)):
                cls_idx = labels[i].item()
                class_activations[cls_idx].append(pooled[i, all_top_features].cpu())

    # Average per class
    sorted_classes = sorted(class_activations.keys())
    class_mean_acts = {}
    for cls_idx in sorted_classes:
        class_mean_acts[cls_idx] = torch.stack(class_activations[cls_idx]).mean(dim=0)

    if len(class_mean_acts) == 0:
        print("  No data collected, skipping.")
        return

    # --- Step 3: Build the figure ---
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # --- Panel 1 (top-left): Per-token top feature attribution ---
    ax1 = fig.add_subplot(gs[0, 0])
    # Stacked bar: for each token, show contribution of top features
    colors = plt.cm.tab20(np.linspace(0, 1, top_k_features))
    bottom = np.zeros(n_ctx)
    legend_handles = []
    for rank in range(top_k_features):
        vals = []
        feat_ids = []
        for t in range(n_ctx):
            idxs, importance = top_features_per_token[t]
            vals.append(importance[rank])
            feat_ids.append(idxs[rank])
        bars = ax1.bar(range(n_ctx), vals, bottom=bottom, color=colors[rank], alpha=0.85,
                       edgecolor='white', linewidth=0.5)
        bottom += vals
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=colors[rank], alpha=0.85))
    ax1.set_xlabel("Prompt Token Index")
    ax1.set_ylabel("Feature Attribution (weight norm)")
    ax1.set_title("SAE-Guided: Top Feature Contributions per Token\n(each color = one SAE feature)")
    ax1.set_xticks(range(n_ctx))

    # Annotate top-1 feature ID on each bar
    for t in range(n_ctx):
        idxs, importance = top_features_per_token[t]
        ax1.text(t, bottom[t] + 0.002, f"F{idxs[0]}", ha='center', fontsize=7, fontweight='bold')

    # --- Panel 2 (top-right): Baseline MaPLe "black box" contrast ---
    ax2 = fig.add_subplot(gs[0, 1])
    ctx_sae = model.prompt_learner.ctx.detach().cpu().float()
    ctx_norms = ctx_sae.norm(dim=1).numpy()
    ax2.bar(range(n_ctx), ctx_norms, color='#95a5a6', alpha=0.9, edgecolor='black')
    ax2.set_xlabel("Prompt Token Index")
    ax2.set_ylabel("L2 Norm")
    ax2.set_title("Baseline MaPLe: Opaque Learned Prompt Tokens\n(no interpretable decomposition)")
    ax2.set_xticks(range(n_ctx))
    for t in range(n_ctx):
        ax2.text(t, ctx_norms[t] + 0.01, "?", ha='center', fontsize=14, color='#7f8c8d')
    ax2.text(0.5, 0.5, "Black-box\noptimized vectors", ha='center', va='center',
             transform=ax2.transAxes, fontsize=16, color='#bdc3c7', fontweight='bold', alpha=0.5)

    # --- Panel 3 (bottom-left): Concept heatmap — top features x classes ---
    ax3 = fig.add_subplot(gs[1, 0])
    heatmap_data = torch.stack([class_mean_acts[c] for c in sorted_classes]).numpy()
    class_labels = [classnames[c][:20] for c in sorted_classes]
    feature_labels = [f"F{f}" for f in all_top_features]

    sns.heatmap(heatmap_data, xticklabels=feature_labels, yticklabels=class_labels,
                cmap="YlOrRd", ax=ax3, cbar_kws={"label": "Mean Activation"})
    ax3.set_xlabel("SAE Feature (top contributors to prompts)")
    ax3.set_ylabel("Class")
    ax3.set_title("What Each Feature Responds To\n(class-specific activation patterns)")
    ax3.tick_params(axis='x', rotation=45, labelsize=7)
    ax3.tick_params(axis='y', labelsize=7)

    # --- Panel 4 (bottom-right): Feature-to-token flow diagram ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_xlim(-0.5, 2.5)
    ax4.set_ylim(-0.5, max(len(all_top_features), n_ctx * 2) + 0.5)
    ax4.axis('off')
    ax4.set_title("Concept Flow: SAE Features → Bottleneck → Prompt Tokens", fontsize=11)

    # Draw SAE features on left
    n_feat = min(len(all_top_features), 15)  # limit for readability
    feat_y_positions = np.linspace(0, n_feat * 1.2, n_feat)
    for i, (feat_idx) in enumerate(all_top_features[:n_feat]):
        # Find which classes activate this feature most
        col_idx = i
        col_data = heatmap_data[:, col_idx]
        top_cls_idx = np.argmax(col_data)
        top_cls_name = class_labels[top_cls_idx][:12]
        ax4.text(0, feat_y_positions[i], f"F{feat_idx}\n({top_cls_name})",
                 ha='center', va='center', fontsize=6,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#f39c12', alpha=0.7))

    # Draw prompt tokens on right
    token_y_positions = np.linspace(0, n_feat * 1.2, n_ctx)
    for t in range(n_ctx):
        ax4.text(2.5, token_y_positions[t], f"Prompt\nToken {t}",
                 ha='center', va='center', fontsize=8,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#3498db', alpha=0.7))

    # Draw connections (top-3 per token)
    for t in range(n_ctx):
        idxs, importance = top_features_per_token[t]
        max_imp = importance[0]
        for rank in range(min(3, len(idxs))):
            feat_idx = idxs[rank]
            if feat_idx in all_top_features[:n_feat]:
                feat_pos = all_top_features[:n_feat].index(feat_idx)
                alpha = 0.3 + 0.7 * (importance[rank] / max_imp)
                lw = 1 + 3 * (importance[rank] / max_imp)
                ax4.annotate("", xy=(2.2, token_y_positions[t]),
                             xytext=(0.4, feat_y_positions[feat_pos]),
                             arrowprops=dict(arrowstyle="->", color='#2c3e50',
                                             alpha=float(alpha), lw=float(lw)))

    plt.savefig(os.path.join(output_dir, "6_concept_attribution.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.join(output_dir, '6_concept_attribution.png')}")

    # Print concept summary
    print("\n  === Concept Summary ===")
    for t in range(n_ctx):
        idxs, importance = top_features_per_token[t]
        concepts = []
        for rank in range(min(5, len(idxs))):
            feat_idx = idxs[rank]
            if feat_idx in all_top_features:
                col_idx = all_top_features.index(feat_idx)
                col_data = heatmap_data[:, col_idx]
                top3_cls = np.argsort(col_data)[::-1][:3]
                cls_str = ", ".join([class_labels[c] for c in top3_cls])
                concepts.append(f"F{feat_idx}({cls_str})")
        print(f"  Token {t}: {' | '.join(concepts)}")


def main():
    parser = argparse.ArgumentParser(description="SAE + MaPLe Visualization")
    parser.add_argument("--config-file", type=str,
                        default="configs/trainers/MaPLe/vit_b16_c2_ep5_batch4_2ctx.yaml")
    parser.add_argument("--dataset-config-file", type=str,
                        default="configs/datasets/fgvc_aircraft.yaml")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Directory containing trained model (e.g. .../seed0)")
    parser.add_argument("--sae-path", type=str,
                        default="/p/lustre1/kundargi1/patchsae/data/sae_weight/base/out.pt")
    parser.add_argument("--data-root", type=str,
                        default="/p/lustre5/kundargi1/multimodal-prompt-learning/data")
    parser.add_argument("--output-dir", type=str, default="viz_output")
    parser.add_argument("--load-epoch", type=int, default=5,
                        help="Which epoch checkpoint to load")
    parser.add_argument("--viz", nargs="+", type=int, default=[1, 2, 3, 4, 5],
                        help="Which visualizations to run (1-6)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Setting up config...")
    cfg = setup_cfg(args)

    print("Building trainer and loading model...")
    trainer = load_model(cfg, args.model_dir, epoch=args.load_epoch)
    model = trainer.model

    if 1 in args.viz:
        viz1_projection_analysis(model, args.output_dir)
    if 2 in args.viz:
        viz2_modulation_magnitude(trainer, args.output_dir)
    if 3 in args.viz:
        viz3_per_class_features(trainer, args.output_dir)
    if 4 in args.viz:
        viz4_tsne_prompts(trainer, args.output_dir)
    if 5 in args.viz:
        viz5_sparsity(trainer, args.output_dir)
    if 6 in args.viz:
        viz6_concept_attribution(trainer, args.output_dir)

    print(f"\nAll visualizations saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
