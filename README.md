[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Fuyu
![FUYU](/architecture.png)

A implementation of Fuyu, the multimodal AI model from Adept in pytorch and zeta. The architecture is basically instead of using an encoder like VIT or CLIP they just patch the image then project it then feed it into the transformer decoder. The architecture is image patch embeddings -> linear projection -> decoder llm. 


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


```

# License
MIT

# Citations
