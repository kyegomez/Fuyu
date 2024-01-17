import torch
from torch import nn, Tensor
from einops import rearrange, reduce
from torch.nn import Module
from zeta.structs import AutoregressiveWrapper, Decoder, Transformer


def exists(val):
    return val is not None


def patch_img(x: Tensor, patches: int):
    return rearrange(
        x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patches, p2=patches
    )


def threed_to_text(x: Tensor, max_seq_len: int, dim: int, flatten: bool = False):
    """
    Converts a 3D tensor to text representation.

    Args:
        x (Tensor): The input tensor of shape (batch_size, sequence_length, input_dim).
        max_seq_len (int): The maximum sequence length of the output tensor.
        dim (int): The dimension of the intermediate tensor.
        flatten (bool, optional): Whether to flatten the intermediate tensor. Defaults to False.

    Returns:
        Tensor: The output tensor of shape (batch_size, max_seq_len, input_dim).
    """
    b, s, d = x.shape

    x = nn.Linear(d, dim)(x)

    x = rearrange(x, "b s d -> b d s")
    x = nn.Linear(s, max_seq_len)(x)
    x = rearrange(x, "b d s -> b s d")
    return x


def text_to_twod(x: Tensor, dim: int):
    b, s, d = x.shape
    x = reduce(x, "b s d -> b d", "mean")
    x = nn.Linear(d, dim)(x)
    return x


class Fuyu(Module):
    """
    Fuyu model class.


    Args:
    - num_tokens: Number of tokens in the vocabulary
    - max_seq_len: Maximum sequence length
    - dim: Dimension of the model
    - depth: Depth of the model
    - dim_head: Dimension of the model head
    - heads: Number of heads
    - use_abs_pos_emb: Whether to use absolute position embedding
    - alibi_pos_bias: Alibi position bias
    - alibi_num_heads: Number of alibi heads
    - rotary_xpos: Rotary position
    - attn_flash: Attention flash
    - deepnorm: Deep normalization
    - shift_tokens: Number of tokens to shift
    - attn_one_kv_head: Attention one key/value head
    - qk_norm: Query-key normalization
    - attn_qk_norm: Attention query-key normalization
    - attn_qk_norm_dim_scale: Attention query-key normalization dimension scale
    - embedding_provider: Embedding provider module


    Example:
    >>> import torch
    >>> from fuyu import Fuyu
    >>> model = Fuyu()
    >>> x = torch.randn(1, 3, 256, 256)
    >>> y = model(x)
    >>> y.shape
    torch.Size([1, 128, 128])


    """

    def __init__(
        self,
        num_tokens=50432,
        max_seq_len=32052,
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
        patches: int = 16,
        *args,
        **kwargs
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.max_seq_len = max_seq_len
        self.dim = dim
        self.depth = depth
        self.dim_head = dim_head
        self.heads = heads
        self.use_abs_pos_emb = use_abs_pos_emb
        self.alibi_pos_bias = alibi_pos_bias
        self.alibi_num_heads = alibi_num_heads
        self.rotary_xpos = rotary_xpos
        self.attn_flash = attn_flash
        self.attn_kv_heads = attn_kv_heads
        self.qk_norm = qk_norm
        self.attn_qk_norm = attn_qk_norm
        self.attn_qk_norm_dim_scale = attn_qk_norm_dim_scale
        self.patches = patches

        try:
            # Transformer model for the model
            self.Fuyu = Transformer(
                num_tokens=num_tokens,
                max_seq_len=max_seq_len,
                use_abs_pos_emb=use_abs_pos_emb,
                attn_layers=Decoder(
                    dim=dim,
                    depth=depth,
                    dim_head=dim_head,
                    heads=heads,
                    alibi_pos_bias=alibi_pos_bias,
                    alibi_num_heads=alibi_num_heads,
                    rotary_xpos=rotary_xpos,
                    attn_flash=attn_flash,
                    attn_kv_heads=attn_kv_heads,
                    qk_norm=qk_norm,
                    attn_qk_norm=attn_qk_norm,
                    attn_qk_norm_dim_scale=attn_qk_norm_dim_scale,
                    cross_attend=True,
                    *args,
                    **kwargs
                ),
            )

            # Autoregressive wrapper for the model
            self.decoder = AutoregressiveWrapper(self.Fuyu)

        except Exception as e:
            print("Failed to initialize Fuyu: ", e)
            raise

    def forward(self, text: torch.Tensor, img: torch.Tensor = None, *args, **kwargs):
        """
        Forward pass of the model.

        Args:
        - text: Text tensor
        - img: Image tensor

        Returns:
        - torch.Tensor: The output of the model

        Text input shape: [batch, seq_len, dim]
        img input shape: [batch, channels, height, width]
        audio input shape: [batch, audio_seq_len]

        Output shape: [batch, seq_len, dim]


        """
        # print(f"Printing text shape: {text.shape}")
        try:
            # If image is provided, concat it with the text
            if exists(img):
                # Patch the image
                img = patch_img(img, patches=self.patches)
                # print(f"Printing img when patched shape: {img.shape}")
                # img = img_to_text(img, self.max_seq_len, self.dim, True)
                img = threed_to_text(img, self.max_seq_len, self.dim, True)
                # img = text_to_twod(img, self.max_seq_len)
                # print(f"Printing img shape after img_to_text: {img.shape}")
                # Concat the text and image
                # text = torch.cat((text, img), dim=1)
                # x = torch.cat((text, img), dim=1)
            return self.decoder(text, context=img, *args, **kwargs)
        except Exception as e:
            print("Failed in forward method: ", e)
            raise
