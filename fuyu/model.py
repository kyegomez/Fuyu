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


def threed_to_text(
    x: Tensor, max_seq_len: int, dim: int, flatten: bool = False
):
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
    Fuyu is an implementation of the Fuyu model by Adept AI.
    The model is a transformer-based model that can be used for various tasks such as image classification, object detection, and more.

    Args:
    - num_tokens (int): The number of tokens in the input vocabulary
    - max_seq_len (int): The maximum sequence length of the input
    - dim (int): The dimension of the model
    - depth (int): The depth of the model
    - dim_head (int): The dimension of the heads
    - heads (int): The number of heads
    - use_abs_pos_emb (bool): Whether to use absolute positional embeddings
    - alibi_pos_bias (bool): Whether to use alibi positional bias
    - alibi_num_heads (int): The number of heads for alibi positional bias
    - rotary_xpos (bool): Whether to use rotary positional embeddings
    - attn_flash (bool): Whether to use attention flash
    - attn_kv_heads (int): The number of heads for attention key-value
    - qk_norm (bool): Whether to use query key normalization
    - attn_qk_norm (bool): Whether to use attention query key normalization
    - attn_qk_norm_dim_scale (bool): Whether to use attention query key normalization dimension scale
    - patches (int): The number of patches
    - stabilize (bool): Whether to use stabilization
    - use_rmsnorm (bool): Whether to use RMSNorm
    - use_simple_rmsnorm (bool): Whether to use simple RMSNorm
    - ff_glu (bool): Whether to use feedforward GLU
    - ff_swish (bool): Whether to use feedforward Swish
    - macaron (bool): Whether to use macaron
    - rel_pos_bias (bool): Whether to use relative positional bias
    - rotary_pos_emb (bool): Whether to use rotary positional embeddings
    - sandwich_norm (bool): Whether to use sandwich normalization
    - ff_post_act_ln (bool): Whether to use feedforward post-activation layer normalization

    Example:
    >>> import torch
    >>> from fuyu.model import Fuyu
    >>>
    >>> # Initialize model
    >>> model = Fuyu(
    >>>     num_tokens=20342,
    >>>     max_seq_len=4092,
    >>>     dim=640,
    >>>     depth=8,
    >>>     dim_head=128,
    >>>     heads=6,
    >>>     use_abs_pos_emb=False,
    >>>     alibi_pos_bias=True,
    >>>     alibi_num_heads=3,
    >>>     rotary_xpos=True,
    >>>     attn_flash=True,
    >>>     attn_kv_heads=2,
    >>>     qk_norm=False,
    >>>     attn_qk_norm=False,
    >>>     attn_qk_norm_dim_scale=False,
    >>>     patches=16,
    >>> )
    >>>
    >>> # Text shape: [batch, seq_len, dim]
    >>> text = torch.randint(0, 20342, (1, 4092))
    >>>
    >>> # Img shape: [batch, channels, height, width]
    >>> img = torch.randn(1, 3, 256, 256)
    >>>
    >>> # Apply model to text and img
    >>> y = model(text, img)
    >>>
    >>> # Output shape: [batch, seq_len, dim]
    >>> print(y)


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
        stabilize: bool = True,
        use_rmsnorm: bool = True,
        use_simple_rmsnorm: bool = False,
        ff_glu: bool = True,
        ff_swish: bool = True,
        macaron: bool = False,
        rel_pos_bias: bool = False,
        rotary_pos_emb: bool = False,
        sandwich_norm: bool = True,
        ff_post_act_ln: bool = True,
        *args,
        **kwargs,
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
        self.stabilize = stabilize

        # Transformer model for the model
        self.fuyu = Transformer(
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
                use_rmsnorm=use_rmsnorm,
                use_simple_rmsnorm=use_simple_rmsnorm,
                ff_glu=ff_glu,
                ff_swish=ff_swish,
                macaron=macaron,
                rel_pos_bias=rel_pos_bias,
                rotary_pos_emb=rotary_pos_emb,
                sandwich_norm=sandwich_norm,
                ff_post_act_ln=ff_post_act_ln,
                *args,
                **kwargs,
            ),
        )

        # Autoregressive wrapper for the model
        self.decoder = AutoregressiveWrapper(self.fuyu)

        self.s_norm = nn.LayerNorm(dim)
        

    def forward(
        self, text: torch.Tensor, img: torch.Tensor = None, *args, **kwargs
    ):
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
        try:
            # If image is provided, concat it with the text
            if exists(img):
                # Patch the image
                img = patch_img(img, patches=self.patches)
                img = threed_to_text(img, self.max_seq_len, self.dim, True)
                img = self.s_norm(img)
            return self.decoder(text, context=img, *args, **kwargs)
        except Exception as e:
            print("Failed in forward method: ", e)
            raise
