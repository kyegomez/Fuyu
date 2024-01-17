import torch
from fuyu.model import Fuyu

# Initialize model
model = Fuyu(
    num_tokens=20342,
    max_seq_len=4092,
    dim=640,
    depth=8,
    dim_head=128,
    heads=6,
    use_abs_pos_emb=False,
    alibi_pos_bias=True,
    alibi_num_heads=3,
    rotary_xpos=True,
    attn_flash=True,
    attn_kv_heads=2,
    qk_norm=False,
    attn_qk_norm=False,
    attn_qk_norm_dim_scale=False,
    patches=16,
)

# Text shape: [batch, seq_len, dim]
text = torch.randint(0, 20342, (1, 4092))

# Img shape: [batch, channels, height, width]
img = torch.randn(1, 3, 256, 256)

# Apply model to text and img
y = model(text, img)

# Output shape: [batch, seq_len, dim]
print(y)
