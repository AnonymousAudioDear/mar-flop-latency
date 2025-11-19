import sys
from contextlib import nullcontext

import torch

from calflops import calculate_flops
from models import mar


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
    num_iter = 64
    fp16 = False
    model = "mar_large"
    cfg = 2
    cfg_schedule = "linear"
    temperature = 1.0


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
)

model.eval()
model.cuda()

batch_size = 1
class_labels = torch.randint(0, args.class_num, (batch_size,)).cuda()

if args.fp16:
    model = model.half()

ctx = torch.cuda.amp.autocast(dtype=torch.float16) if args.fp16 else nullcontext()


# Wrapper for class embedding
class ClassEmbWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.class_emb = model.class_emb

    def forward(self, labels):
        return self.class_emb(labels)


# Wrapper for Transformer (encoder + decoder merged)
class TransformerWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens, mask, class_embedding):
        x = self.model.forward_mae_encoder(tokens, mask, class_embedding)
        return self.model.forward_mae_decoder(x, mask)


# Wrapper for diffloss
class DiffLossWrapper(torch.nn.Module):
    def __init__(self, model, temperature, cfg_iter):
        super().__init__()
        self.diffloss = model.diffloss
        self.temperature = temperature
        self.cfg_iter = cfg_iter

    def forward(self, z):
        return self.diffloss.sample(z, self.temperature, self.cfg_iter)


print("Measuring FLOPs for each component...\n")

# 1. Class embedding (only once)
class_emb_wrapper = ClassEmbWrapper(model)
with ctx:
    flops_class, macs_class, params_class = calculate_flops(
        model=class_emb_wrapper,
        args=[class_labels],
        forward_mode="forward",
        print_results=False,
        print_detailed=False,
        output_as_string=False,
        output_precision=2,
    )

# 2. Transformer (per iteration, with CFG)
bsz = batch_size
seq_len = model.seq_len
token_embed_dim = model.token_embed_dim
tokens = torch.zeros(bsz, seq_len, token_embed_dim).cuda()
mask = torch.ones(bsz, seq_len).cuda()
class_embedding = model.class_emb(class_labels)

if args.fp16:
    tokens = tokens.half()
    mask = mask.half()

if args.cfg != 1.0:
    tokens_cfg = torch.cat([tokens, tokens], dim=0)
    class_cfg = torch.cat(
        [
            class_embedding,
            model.fake_latent.repeat(bsz, 1)
        ],
        dim=0
    )
    mask_cfg = torch.cat([mask, mask], dim=0)
else:
    tokens_cfg, class_cfg, mask_cfg = tokens, class_embedding, mask

transformer_wrapper = TransformerWrapper(model)
with ctx:
    flops_transformer, macs_transformer, params_transformer = \
        calculate_flops(
            model=transformer_wrapper,
            args=[tokens_cfg, mask_cfg, class_cfg],
            forward_mode="forward",
            print_results=False,
            print_detailed=False,
            output_as_string=False,
            output_precision=2,
        )

print(f"Transformer (per iter): FLOPs={flops_transformer:.2e}, "
      f"MACs={macs_transformer:.2e}")

# 3. DiffLoss sample (per iteration, varies by mask_to_pred size)
with torch.no_grad(), ctx:
    z = transformer_wrapper(tokens_cfg, mask_cfg, class_cfg)

# Use fixed average tokens per iteration
avg_pred_tokens = 4

z_flat = z.reshape(-1, z.shape[-1])
z_sample = z_flat[: avg_pred_tokens * (2 if args.cfg != 1.0 else 1)]

print(f"Using avg_pred_tokens = {avg_pred_tokens}, z_sample.shape = {z_sample.shape}")

diffloss_wrapper = DiffLossWrapper(model, args.temperature, args.cfg)
with ctx:
    flops_diff, macs_diff, params_diff = calculate_flops(
        model=diffloss_wrapper,
        args=[z_sample],
        forward_mode="forward",
        print_results=False,
        print_detailed=False,
        output_as_string=False,
        output_precision=2,
    )

print(f"MLP diffusion sampler (per iter, avg): FLOPs={flops_diff:.2e}, MACs={macs_diff:.2e}")

# Total estimation
total_flops = flops_class + args.num_iter * (flops_transformer + flops_diff)
total_macs = macs_class + args.num_iter * (macs_transformer + macs_diff)

print(f"\nFormula:")
print(
    f"Total FLOPs = {args.num_iter} * ("
    f"{flops_transformer:.2e} + {flops_diff:.2e})"
)
print(
    f"                  = {args.num_iter} * {flops_transformer + flops_diff:.2e}"
)
print(f"                  = {total_flops:.2e}")

print(f"\n{'='*100}")
print("Component Breakdown:")
print(f"{'='*100}")

print(
    f"Transformer Encoder ({args.num_iter} decoding iter):        {args.num_iter*flops_transformer:.2e} FLOPs "
    f"({100*args.num_iter*flops_transformer/total_flops:.1f}%)"
)
print(
    f"MLP diffusion sampler ({args.num_iter} decoding iter x 100 sampling steps):        {args.num_iter*flops_diff:.2e} FLOPs "
    f"({100*args.num_iter*flops_diff/total_flops:.1f}%)"
)

print(f"\n{'='*100}")
print(f"{args.model = }")
print(f"Total FLOPs:      {total_flops:.2e} FLOPs")
print(f"Total MACs:       {total_macs:.2e}")
print(f"CFG: {args.cfg}")
print(f"FP16: {args.fp16}")
print(f"{'='*100}")

