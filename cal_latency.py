import gc
from contextlib import nullcontext
import math

import numpy as np
import torch

from models import mar
from models.mar import mask_by_order


class Args:
    img_size = 256
    vae_stride = 16
    patch_size = 1
    vae_embed_dim = 16
    mask_ratio_min = 0.7
    label_drop_prob = 0.1
    class_num = 1000
    attn_dropout = 0.1
    proj_dropout = 0.1
    buffer_size = 64
    grad_checkpointing = False
    bf16 = False
    cfg = 2
    cfg_schedule = "linear"
    temperature = 1.0
    model = "mar_large"
    repetitions = 1
    warmup = 20
    num_sampling_steps = '100'
    num_iter = 64


args = Args()

model = mar.__dict__[args.model](
    img_size=args.img_size,
    vae_stride=args.vae_stride,
    patch_size=args.patch_size,
    vae_embed_dim=args.vae_embed_dim,
    mask_ratio_min=args.mask_ratio_min,
    label_drop_prob=args.label_drop_prob,
    class_num=args.class_num,
    attn_dropout=args.attn_dropout,
    proj_dropout=args.proj_dropout,
    buffer_size=args.buffer_size,
    grad_checkpointing=args.grad_checkpointing,
    num_sampling_steps=args.num_sampling_steps,
)

model.eval()
model.cuda()

batch_size = 1
class_labels = torch.randint(0, args.class_num, (batch_size,)).cuda()

if args.bf16:
    model = model.bfloat16()

gc.collect()
torch.cuda.empty_cache()

ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16) if args.bf16 else nullcontext()

# Warmup
print("GPU WARM-UP...")
bsz = batch_size
mask = torch.ones(bsz, model.seq_len).cuda()
tokens = torch.zeros(bsz, model.seq_len, model.token_embed_dim).cuda()
orders = model.sample_orders(bsz)

if args.bf16:
    tokens = tokens.bfloat16()
    mask = mask.bfloat16()

with torch.no_grad():
    for _ in range(args.warmup):
        with ctx:
            class_embedding = model.class_emb(class_labels)
            tokens_cfg = torch.cat([tokens, tokens], dim=0) if args.cfg != 1.0 else tokens
            class_cfg = torch.cat([class_embedding,
                                   model.fake_latent.repeat(bsz, 1)], dim=0) if args.cfg != 1.0 else class_embedding
            mask_cfg = torch.cat([mask, mask], dim=0) if args.cfg != 1.0 else mask
            x = model.forward_mae_encoder(tokens_cfg, mask_cfg, class_cfg)
            z = model.forward_mae_decoder(x, mask_cfg)
            z_sample = z.reshape(-1, z.shape[-1])[:10]
            _ = model.diffloss.sample(z_sample, args.temperature, args.cfg)

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# 1. Class embedding
timings_class = np.zeros(args.repetitions)
print("Measuring Class Embedding latency...")
with torch.no_grad():
    for rep in range(args.repetitions):
        starter.record()
        with ctx:
            _ = model.class_emb(class_labels)
        ender.record()
        torch.cuda.synchronize()
        timings_class[rep] = starter.elapsed_time(ender)

# 2. Measure Transformer and DiffLoss in the actual loop
print("Measuring Transformer and DiffLoss latency in real loop...")
timings_transformer = []
timings_diff = []
token_counts = []

mask = torch.ones(bsz, model.seq_len).cuda()
tokens = torch.zeros(bsz, model.seq_len, model.token_embed_dim).cuda()
orders = model.sample_orders(bsz)

if args.bf16:
    tokens = tokens.bfloat16()
    mask = mask.bfloat16()

with torch.no_grad():
    for step in range(args.num_iter):
        cur_tokens = tokens.clone()

        # class embedding and CFG
        class_embedding = model.class_emb(class_labels)
        if args.cfg != 1.0:
            tokens_cfg = torch.cat([tokens, tokens], dim=0)
            class_cfg = torch.cat([class_embedding,
                                   model.fake_latent.repeat(bsz, 1)], dim=0)
            mask_cfg = torch.cat([mask, mask], dim=0)
        else:
            tokens_cfg, class_cfg, mask_cfg = tokens, class_embedding, mask

        # Measure Transformer (encoder + decoder)
        timings_trans_iter = np.zeros(args.repetitions)
        for rep in range(args.repetitions):
            starter.record()
            with ctx:
                x = model.forward_mae_encoder(tokens_cfg, mask_cfg, class_cfg)
                _ = model.forward_mae_decoder(x, mask_cfg)
            ender.record()
            torch.cuda.synchronize()
            timings_trans_iter[rep] = starter.elapsed_time(ender)
        timings_transformer.append(timings_trans_iter)

        # Get z for DiffLoss measurement
        with ctx:
            x = model.forward_mae_encoder(tokens_cfg, mask_cfg, class_cfg)
            z = model.forward_mae_decoder(x, mask_cfg)

        # Calculate mask_to_pred following sample_tokens logic
        mask_ratio = np.cos(math.pi / 2 * (step + 1) / args.num_iter)
        mask_len = torch.Tensor([np.floor(model.seq_len * mask_ratio)]).cuda()
        mask_len = torch.maximum(torch.Tensor([1]).cuda(),
                                torch.minimum(torch.sum(mask, dim=1, keepdims=True) - 1, mask_len))

        mask_next = mask_by_order(mask_len[0], orders, bsz, model.seq_len)

        if step >= args.num_iter - 1:
            mask_to_pred = mask[:bsz].bool()
        else:
            mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
        mask = mask_next
        if args.cfg != 1.0:
            mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

        # Get real z for this iteration
        z_real = z[mask_to_pred.nonzero(as_tuple=True)]
        token_counts.append(z_real.shape[0])

        # Measure DiffLoss
        timings_diff_iter = np.zeros(args.repetitions)
        for rep in range(args.repetitions):
            starter.record()
            with ctx:
                _ = model.diffloss.sample(z_real, args.temperature, args.cfg)
            ender.record()
            torch.cuda.synchronize()
            timings_diff_iter[rep] = starter.elapsed_time(ender)
        timings_diff.append(np.mean(timings_diff_iter))

        # Update tokens for next iteration (simplified, just for loop continuation)
        if args.cfg != 1.0:
            sampled_token_latent = model.diffloss.sample(z_real, args.temperature, args.cfg)
            sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)
            mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)
        else:
            sampled_token_latent = model.diffloss.sample(z_real, args.temperature, args.cfg)

        cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
        tokens = cur_tokens.clone()

mean_class = float(np.mean(timings_class))
total_transformer = float(sum(timings_transformer))
total_diff = float(sum(timings_diff))
total_latency = mean_class + total_transformer + total_diff

print(f"\n{'='*60}")
print("Component Latency Breakdown:")
print(f"{'='*60}")
print(
    f"Transformer Encoder ({args.num_iter}x):      {total_transformer:.2f} ms ({100*total_transformer/total_latency:.1f}%)"
)
print(
    f"MLP diffusion sampler:           {total_diff:.2f} ms ({100*total_diff/total_latency:.1f}%)"
)

print()
print(f"Total tokens processed: {sum(token_counts)}")
print(f"Token counts per iter: {token_counts}")
print(f"Total latency: {total_latency:.2f} ms")

print("\nARGS:")
print(f"  num_iter: {args.num_iter}")
print(f"  sampling steps: {args.num_sampling_steps}")
print(f"  CFG: {args.cfg}")
print(f"  bf16: {args.bf16}")
print(f"  model: {args.model}")
print(f"{'='*60}")