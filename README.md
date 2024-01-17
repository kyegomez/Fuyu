[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Fuyu
![FUYU](/architecture.png)

A implementation of Fuyu, the multimodal AI model from Adept in pytorch using a ultra powerful Zeta native decoder with qk_norm, multi grouped query attn, scaled rotary pos embeddings, and a kv_cache


[Blog paper code](https://www.adept.ai/blog/fuyu-8b)

# Appreciation
* Lucidrains
* Agorians
* Adept

# Install
`pip install fuyu`

## Usage
```python
import torch
from fuyu import Fuyu

# Initialize model
model = Fuyu(
    num_tokens=50432,
    max_seq_len=8192,
    dim=2560,
    depth=32,
    dim_head=128,
    heads=24,
    use_abs_pos_emb=False,
    alibi_pos_bias=True,
    alibi_num_heads=12,
    rotary_xpos=True,
    attn_flash=True,
    attn_kv_heads=2,
    qk_norm=True,
    attn_qk_norm=True,
    attn_qk_norm_dim_scale=True,
)

# Text shape: [batch, seq_len, dim]
text = torch.randint(0, 50432, (1, 8192))

# Img shape: [batch, channels, height, width]
img = torch.randn(1, 3, 256, 256)

# Apply model to text and img
y = model(text, img)

# Output shape: [batch, seq_len, dim]
print(y.shape)


```

# Architecture
image patch embeddings -> linear projection -> decoder llm

# License
MIT

# Citations
