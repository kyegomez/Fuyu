import math
import re
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from transformers.image_processing_utils import BaseImageProcessor
from transformers.processing_utils import ProcessorMixin
from transformers.utils import (
    logging,
    requires_backends,
)

from fuyu.utils import (
    ChannelDimension,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    normalize,
    pad,
    resize,
    to_numpy_array,
)

logger = logging.get_logger(__name__)


class FuyuImageProcessor(BaseImageProcessor):
    """
    This class should handle the image processing part before the main FuyuForCausalLM. In particular, it should
    handle:

    - Processing Images:
        Taking a batch of images as input. If the images are variable-sized, it resizes them based on the desired patch
        dimensions. The image output is always img_h ........................................... 1080 img_w
        ........................................... 1920 Then, it patches up these images using the patchify_image
        function.

    - Creating Image Input IDs:
        For each patch, a placeholder ID is given to identify where these patches belong in a token sequence. For
        variable-sized images, each line of patches is terminated with a newline ID.

    - Image Patch Indices:
        For each image patch, the code maintains an index where these patches should be inserted in a token stream.

    """

    model_input_names = [
        "images",
        "image_input_ids",
        "image_patches",
        "image_patch_indices_per_batch",
        "image_patch_indices_per_subsequence",
    ]

    def __init__(
        self,
        target_height=1080,
        target_width=1920,
        padding_value=1.0,
        padding_mode: str = "constant",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_width = target_width
        self.target_height = target_height
        self.padding_value = padding_value
        self.padding_mode = padding_mode

    def get_num_patches(
        self, img_h: int, img_w: int, patch_dim_h: int, patch_dim_w: int
    ) -> int:
        """Calculate number of patches required to encode an image."""
        if img_h % patch_dim_h != 0:
            raise ValueError(f"{img_h=} must be divisible by {patch_dim_h=}")
        if img_w % patch_dim_w != 0:
            raise ValueError(f"{img_w=} must be divisible by {patch_dim_w=}")

        num_patches_per_dim_h = img_h // patch_dim_h
        num_patches_per_dim_w = img_w // patch_dim_w
        num_patches = num_patches_per_dim_h * num_patches_per_dim_w

        return num_patches

    def patchify_image(
        self, image: "torch.Tensor", patch_dim_h: int, patch_dim_w: int
    ) -> "torch.Tensor":
        """
        Convert an image into a tensor of patches.

        Args:
            image: Image to convert. Shape: [batch, channels, height, width]
            patch_dim_h: Height of each patch.
            patch_dim_w: Width of each patch.
        """
        requires_backends(self, ["torch"])

        # TODO refer to https://github.com/ArthurZucker/transformers/blob/0f0a3fe5ca5697ee58faeb5b53f049af720b5e98/src/transformers/models/vit_mae/modeling_vit_mae.py#L871
        # torch implementation is faster but does not handle non-squares

        batch_size, channels, height, width = image.shape
        unfolded_along_height = image.unfold(2, patch_dim_h, patch_dim_h)
        patches = unfolded_along_height.unfold(3, patch_dim_w, patch_dim_w)

        patches_reshaped = patches.contiguous().view(
            batch_size, channels, -1, patch_dim_h, patch_dim_w
        )

        patches_final = patches_reshaped.permute(0, 2, 3, 4, 1).reshape(
            batch_size, -1, channels * patch_dim_h * patch_dim_w
        )

        return patches_final

    def process_images_for_model_input(
        self,
        image_input: "torch.Tensor",
        image_present: "torch.Tensor",
        image_unpadded_h: "torch.Tensor",
        image_unpadded_w: "torch.Tensor",
        image_patch_dim_h: int,
        image_patch_dim_w: int,
        image_placeholder_id: int,
        image_newline_id: int,
        variable_sized: bool,
    ) -> dict:
        """Process images for model input. In particular, variable-sized images are handled here.

        Args:
            image_input: [batch_size, 1, c, h, w] tensor of images padded to model input size.
            image_present: [batch_size, 1] tensor of 1s and 0s indicating whether an image is present.
            image_unpadded_h: [batch_size, 1] tensor of unpadded image heights.
            image_unpadded_w: [batch_size, 1] tensor of unpadded image widths.
            image_patch_dim_h: The height of the image patches.
            image_patch_dim_w: The width of the image patches.
            image_placeholder_id: The id of the image placeholder token.
            image_newline_id: The id of the image newline token.
            variable_sized: Whether to process images as variable-sized.
        """
        requires_backends(self, ["torch"])
        # Only images that are present.
        images: List[List[torch.Tensor]] = []
        image_patches: List[List[torch.Tensor]] = []
        # Image input ids for every subsequence, including ones with no image present.
        image_input_ids: List[List[torch.Tensor]] = []
        for bi in range(image_input.shape[0]):
            images.append([])
            image_input_ids.append([])
            image_patches.append([])
            for si in range(image_input.shape[1]):
                if image_present[bi, si]:
                    image = image_input[bi, si]
                    if variable_sized:
                        # The min() is required here due to floating point issues:
                        # math.ceil(torch.tensor(300).cuda() / 30) == 11
                        new_h = min(
                            image.shape[1],
                            math.ceil(image_unpadded_h[bi, si] / image_patch_dim_h)
                            * image_patch_dim_h,
                        )
                        new_w = min(
                            image.shape[2],
                            math.ceil(image_unpadded_w[bi, si] / image_patch_dim_w)
                            * image_patch_dim_w,
                        )
                        image = image[:, :new_h, :new_w]
                    images[bi].append(image)
                    num_patches = self.get_num_patches(
                        img_h=image.shape[1],
                        img_w=image.shape[2],
                        patch_dim_h=image_patch_dim_h,
                        patch_dim_w=image_patch_dim_w,
                    )
                    ids = torch.full(
                        [num_patches],
                        image_placeholder_id,
                        dtype=torch.int32,
                        device=image_input.device,
                    )
                    patches = self.patchify_image(
                        image=image.unsqueeze(0),
                        patch_dim_h=image_patch_dim_h,
                        patch_dim_w=image_patch_dim_w,
                    ).squeeze(0)
                    if variable_sized:
                        # Now terminate each line with |NEWLINE|.
                        ids = ids.reshape(-1, new_w // image_patch_dim_w)
                        ids = torch.cat(
                            [
                                ids,
                                torch.full(
                                    [ids.shape[0], 1],
                                    image_newline_id,
                                    dtype=torch.int32,
                                    device=image_input.device,
                                ),
                            ],
                            dim=1,
                        )
                        ids = ids.reshape(-1)
                    image_input_ids[bi].append(ids)
                    image_patches[bi].append(patches)
                else:
                    image_input_ids[bi].append(
                        torch.tensor([], dtype=torch.int32, device=image_input.device)
                    )

        # Create image_patch_input_indices, where non-negative values correspond to image patches to be inserted in
        # the stream.
        image_patch_indices_per_batch: List[List[torch.Tensor]] = []
        image_patch_indices_per_subsequence: List[List[torch.Tensor]] = []
        for bi in range(len(image_input_ids)):
            image_patch_indices_per_batch.append([])
            image_patch_indices_per_subsequence.append([])
            index_offset = 0
            for si in range(len(image_input_ids[bi])):
                # Indices of image patches.
                num_patches = torch.count_nonzero(
                    image_input_ids[bi][si] == image_placeholder_id
                )
                indices = torch.arange(
                    num_patches,
                    dtype=image_input_ids[bi][si].dtype,
                    device=image_input_ids[bi][si].device,
                )

                # Place those indices in the image input ids token stream, with -1 representing non-index tokens.
                indices_in_stream_per_batch = torch.full_like(
                    image_input_ids[bi][si], -1
                )
                indices_in_stream_per_subsequence = torch.full_like(
                    image_input_ids[bi][si], -1
                )
                indices_in_stream_per_batch[
                    torch.nonzero(
                        image_input_ids[bi][si] == image_placeholder_id, as_tuple=True
                    )[0]
                ] = (indices + index_offset)
                indices_in_stream_per_subsequence[
                    torch.nonzero(
                        image_input_ids[bi][si] == image_placeholder_id, as_tuple=True
                    )[0]
                ] = indices

                image_patch_indices_per_batch[bi].append(indices_in_stream_per_batch)
                image_patch_indices_per_subsequence[bi].append(
                    indices_in_stream_per_subsequence
                )
                index_offset += num_patches

        return {
            "images": images,
            "image_input_ids": image_input_ids,
            "image_patches": image_patches,
            "image_patch_indices_per_batch": image_patch_indices_per_batch,
            "image_patch_indices_per_subsequence": image_patch_indices_per_subsequence,
        }

    def _scale_to_target_aspect_ratio(self, image: np.ndarray) -> np.ndarray:
        image_height, image_width, _ = image.shape
        if image_width <= self.target_width and image_height <= self.target_height:
            return image

        height_scale_factor = self.target_height / image_height
        width_scale_factor = self.target_width / image_width
        optimal_scale_factor = min(height_scale_factor, width_scale_factor)

        new_height = int(image_height * optimal_scale_factor)
        new_width = int(image_width * optimal_scale_factor)

        scaled_image = resize(image=image, size=(new_width, new_height))
        return np.array(scaled_image)

    def _pad_to_target_size(self, image: np.ndarray) -> np.ndarray:
        image_height, image_width, _ = image.shape

        padding_top = 0
        padding_left = 0
        padding_bottom = self.target_height - image_height
        padding_right = self.target_width - image_width

        padded_image = pad(
            image,
            ((padding_top, padding_bottom), (padding_left, padding_right)),
            mode=self.padding_mode,
            constant_values=self.padding_value,
        )
        return padded_image

    def apply_transformation(
        self, image: Union[np.ndarray, PIL.Image.Image]
    ) -> np.ndarray:
        if isinstance(image, PIL.Image.Image):
            image = to_numpy_array(image)
        scaled_image = self._scale_to_target_aspect_ratio(image)
        padded_image = self._pad_to_target_size(scaled_image)
        normalized_padded_image = normalize(padded_image, 0.5, 0.5)
        return normalized_padded_image


BBOX_OPEN_STRING = "<0x00>"  # <bbox>
BBOX_CLOSE_STRING = "<0x01>"  # </bbox>
POINT_OPEN_STRING = "<0x02>"  # <point>
POINT_CLOSE_STRING = "<0x03>"  # </point>

TEXT_REPR_BBOX_OPEN = "<box>"
TEXT_REPR_BBOX_CLOSE = "</box>"
TEXT_REPR_POINT_OPEN = "<point>"
TEXT_REPR_POINT_CLOSE = "</point>"

TOKEN_BBOX_OPEN_STRING = BBOX_OPEN_STRING = "<0x00>"  # <bbox>
BBOX_CLOSE_STRING = "<0x01>"  # </bbox>
TOKEN_BBOX_CLOSE_STRING = (
    TOKEN_POINT_OPEN_STRING
) = POINT_OPEN_STRING = "<0x02>"  # <point>
TOKEN_POINT_CLOSE_STRING = POINT_CLOSE_STRING = "<0x03>"  # </point>
BEGINNING_OF_ANSWER_STRING = "<0x04>"  # <boa>


def full_unpacked_stream_to_tensor(
    all_bi_tokens_to_place: List[int],
    full_unpacked_stream: List["torch.Tensor"],
    fill_value: int,
    batch_size: int,
    new_seq_len: int,
    offset: int,
) -> "torch.Tensor":
    """Takes an unpacked stream of tokens (i.e. a list of tensors, one for each item in the batch) and does
    the required padding to create a single tensor for the batch of shape batch_size x new_seq_len.
    """

    assert len(all_bi_tokens_to_place) == batch_size
    assert len(full_unpacked_stream) == batch_size

    # Create padded tensors for the full batch.
    new_padded_tensor = torch.full(
        [batch_size, new_seq_len],
        fill_value=fill_value,
        dtype=full_unpacked_stream[0].dtype,
        device=full_unpacked_stream[0].device,
    )

    # Place each batch entry into the batch tensor.
    for bi in range(batch_size):
        tokens_to_place = all_bi_tokens_to_place[bi]
        new_padded_tensor[bi, :tokens_to_place] = full_unpacked_stream[bi][
            offset : tokens_to_place + offset
        ]

    return new_padded_tensor


def construct_full_unpacked_stream(
    num_real_text_tokens: Union[List[List[int]], "torch.Tensor"],
    input_stream: "torch.Tensor",
    image_tokens: List[List["torch.Tensor"]],
    batch_size: int,
    num_sub_sequences: int,
) -> List["torch.Tensor"]:
    """Takes an input_stream tensor of shape B x S x ?. For each subsequence, adds any required
    padding to account for images and then unpacks the subsequences to create a single sequence per item in the batch.
    Returns a list of tensors, one for each item in the batch."""

    all_bi_stream = []

    for bi in range(batch_size):
        all_si_stream = []

        # First, construct full token stream (including image placeholder tokens) and loss mask for each subsequence
        # and append to lists. We use lists rather than tensors because each subsequence is variable-sized.
        for si in range(num_sub_sequences):
            image_adjustment = image_tokens[bi][si]
            si_stream = torch.cat([image_adjustment, input_stream[bi, si]], dim=0)
            num_real_tokens = image_adjustment.shape[0] + num_real_text_tokens[bi][si]

            all_si_stream.append(si_stream[:num_real_tokens])
        # Combine all subsequences for this batch entry. Still using a list because each batch entry is variable-sized.
        all_bi_stream.append(torch.cat(all_si_stream, dim=0))

    return all_bi_stream


def _replace_string_repr_with_token_tags(prompt: str) -> str:
    prompt = prompt.replace(TEXT_REPR_POINT_OPEN, TOKEN_POINT_OPEN_STRING)
    prompt = prompt.replace(TEXT_REPR_POINT_CLOSE, TOKEN_POINT_CLOSE_STRING)
    prompt = prompt.replace(TEXT_REPR_BBOX_OPEN, TOKEN_BBOX_OPEN_STRING)
    prompt = prompt.replace(TEXT_REPR_BBOX_CLOSE, TOKEN_BBOX_CLOSE_STRING)
    return prompt


def _segment_prompt_into_text_token_conversions(prompt: str) -> List:
    """
    Given a string prompt, converts the prompt into a list of TextTokenConversions.
    """
    # Wherever, we notice the [TOKEN_OPEN_STRING, TOKEN_CLOSE_STRING], we split the prompt
    prompt_text_list: List = []
    regex_pattern = re.compile(
        f"({TOKEN_BBOX_OPEN_STRING}|{TOKEN_BBOX_CLOSE_STRING}|{TOKEN_POINT_OPEN_STRING}|{TOKEN_POINT_CLOSE_STRING})"
    )
    # Split by the regex pattern
    prompt_split = regex_pattern.split(prompt)
    for i, elem in enumerate(prompt_split):
        if len(elem) == 0 or elem in [
            TOKEN_BBOX_OPEN_STRING,
            TOKEN_BBOX_CLOSE_STRING,
            TOKEN_POINT_OPEN_STRING,
            TOKEN_POINT_CLOSE_STRING,
        ]:
            continue
        prompt_text_list.append(
            (
                elem,
                i > 1
                and prompt_split[i - 1]
                in [TOKEN_BBOX_OPEN_STRING, TOKEN_POINT_OPEN_STRING],
            )
        )
    return prompt_text_list


def _transform_coordinates_and_tokenize(
    prompt: str, transformed_image, tokenizer
) -> List[int]:
    """
    This function transforms the prompt in the following fashion:
    - <box> <point> and </box> </point> to their respective token mappings
    - extract the coordinates from the tag
    - transform the coordinates into the transformed image space
    - return the prompt tokens with the transformed coordinates and new tags

    Bounding boxes and points MUST be in the following format: <box>y1, x1, y2, x2</box> <point>x, y</point> The spaces
    and punctuation added above are NOT optional.
    """
    # Make a namedtuple that stores "text" and "is_bbox"

    # We want to do the following: Tokenize the code normally -> when we see a point or box, tokenize using the tokenize_within_tag function
    # When point or box close tag, continue tokenizing normally
    # First, we replace the point and box tags with their respective tokens
    prompt = _replace_string_repr_with_token_tags(prompt)
    # Tokenize the prompt
    # Convert prompt into a list split
    prompt_text_list = _segment_prompt_into_text_token_conversions(prompt)
    transformed_prompt_tokens: List[int] = []
    for elem in prompt_text_list:
        if elem[1]:
            # This is a location, we need to tokenize it
            within_tag_tokenized = _transform_within_tags(
                elem[0], transformed_image, tokenizer
            )
            # Surround the text with the open and close tags
            transformed_prompt_tokens.extend(within_tag_tokenized)
        else:
            transformed_prompt_tokens.extend(
                tokenizer(elem[0], add_special_tokens=False).input_ids
            )
    return transformed_prompt_tokens


def _transform_within_tags(text: str, transformed_image, tokenizer) -> List[int]:
    """
    Given a bounding box of the fashion <box>1, 2, 3, 4</box> | <point>1, 2</point> This function is responsible for
    converting 1, 2, 3, 4 into tokens of 1 2 3 4 without any commas.
    """
    # Convert the text into a list of strings.
    num_int_strs = text.split(",")
    if len(num_int_strs) == 2:
        # If there are any open or close tags, remove them.
        token_space_open_string = tokenizer.vocab[TOKEN_POINT_OPEN_STRING]
        token_space_close_string = tokenizer.vocab[TOKEN_POINT_CLOSE_STRING]
    else:
        token_space_open_string = tokenizer.vocab[TOKEN_BBOX_OPEN_STRING]
        token_space_close_string = tokenizer.vocab[TOKEN_BBOX_CLOSE_STRING]

    # Remove all spaces from num_ints
    num_ints = [float(num.strip()) for num in num_int_strs]
    # scale to transformed image siz
    if len(num_ints) == 2:
        num_ints_translated = scale_point_to_transformed_image(
            x=num_ints[0], y=num_ints[1], transformed_image=transformed_image
        )
    elif len(num_ints) == 4:
        num_ints_translated = scale_bbox_to_transformed_image(
            top=num_ints[0],
            left=num_ints[1],
            bottom=num_ints[2],
            right=num_ints[3],
            transformed_image=transformed_image,
        )
    else:
        raise ValueError(f"Invalid number of ints: {len(num_ints)}")
    # Tokenize the text, skipping the
    tokens = [tokenizer.vocab[str(num)] for num in num_ints_translated]
    return [token_space_open_string] + tokens + [token_space_close_string]


def _tokenize_prompts_with_image_and_batch(
    tokenizer,
    prompts: List[List[str]],
    transformed_images: Optional[List[List["torch.Tensor"]]],
    max_tokens_to_generate: int,
    max_position_embeddings: int,
    add_BOS: bool,  # Same issue with types as above
    add_beginning_of_answer_token: bool,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """
    Given a set of prompts and number of tokens to generate:
    - tokenize prompts
    - set the sequence length to be the max of length of prompts plus the number of tokens we would like to generate
    - pad all the sequences to this length so we can convert them into a 3D tensor.
    """

    # If not tool use, tranform the coordinates while tokenizing
    if transformed_images is not None:
        transformed_prompt_tokens = []
        for prompt_seq, transformed_image_seq in zip(prompts, transformed_images):
            transformed_prompt_tokens.append(
                [
                    _transform_coordinates_and_tokenize(
                        prompt, transformed_image, tokenizer
                    )
                    for prompt, transformed_image in zip(
                        prompt_seq, transformed_image_seq
                    )
                ]
            )
    else:
        transformed_prompt_tokens = [
            [tokenizer.tokenize(prompt) for prompt in prompt_seq]
            for prompt_seq in prompts
        ]

    prompts_tokens = transformed_prompt_tokens

    if add_BOS:
        bos_token = tokenizer.vocab["<s>"]
    else:
        bos_token = tokenizer.vocab["|ENDOFTEXT|"]
    prompts_tokens = [
        [[bos_token] + x for x in prompt_seq] for prompt_seq in prompts_tokens
    ]
    if add_beginning_of_answer_token:
        boa = tokenizer.vocab[BEGINNING_OF_ANSWER_STRING]
        # Only add bbox open token to the last subsequence since that is what will be completed
        for token_seq in prompts_tokens:
            token_seq[-1].append(boa)

    # Now we have a list of list of tokens which each list has a different
    # size. We want to extend this list to:
    #   - incorporate the tokens that need to be generated
    #   - make all the sequences equal length.
    # Get the prompts length.

    prompts_length = [
        [len(x) for x in prompts_tokens_seq] for prompts_tokens_seq in prompts_tokens
    ]
    # Get the max prompts length.
    max_prompt_len: int = np.max(prompts_length)
    # Number of tokens in the each sample of the batch.
    samples_length = min(
        max_prompt_len + max_tokens_to_generate, max_position_embeddings
    )
    if max_prompt_len + max_tokens_to_generate > max_position_embeddings:
        print(
            f"Max subsequence prompt length of {max_prompt_len} + max tokens to generate {max_tokens_to_generate}",
            f"exceeds context length of {max_position_embeddings}. Will generate as many tokens as possible.",
        )
    # Now update the list of list to be of the same size: samples_length.
    for prompt_tokens_seq, prompts_length_seq in zip(prompts_tokens, prompts_length):
        for prompt_tokens, prompt_length in zip(prompt_tokens_seq, prompts_length_seq):
            if len(prompt_tokens) > samples_length:
                raise ValueError(
                    "Length of subsequence prompt exceeds sequence length."
                )
            padding_size = samples_length - prompt_length
            prompt_tokens.extend([tokenizer.vocab["|ENDOFTEXT|"]] * padding_size)

    # Now we are in a structured format, we can convert to tensors.
    prompts_tokens_tensor = torch.tensor(prompts_tokens, dtype=torch.int64)
    prompts_length_tensor = torch.tensor(prompts_length, dtype=torch.int64)

    return prompts_tokens_tensor, prompts_length_tensor


def original_to_transformed_h_coords(self, original_coords):
    # apply crop
    cropped_coords = (
        self._clamp_coords(
            original_coords, min_value=self.crop_top, max_value=self.crop_bottom
        )
        - self.crop_top
    )
    # apply scale
    scaled_coords = self._scale_coords(
        cropped_coords, scale=self.scaled_h / self.original_h
    )
    # apply pad
    return scaled_coords + self.padding_top


def original_to_transformed_w_coords(self, original_coords):
    # apply crop
    cropped_coords = (
        self._clamp_coords(
            original_coords, min_value=self.crop_left, max_value=self.crop_right
        )
        - self.crop_left
    )
    # apply scale
    scaled_coords = self._scale_coords(
        cropped_coords, scale=self.scaled_w / self.original_w
    )
    # apply pad
    return scaled_coords + self.padding_left


def scale_point_to_transformed_image(x: float, y: float) -> List[int]:
    x_scaled = original_to_transformed_w_coords(np.array([x / 2]))[0]
    y_scaled = original_to_transformed_h_coords(np.array([y / 2]))[0]
    return [x_scaled, y_scaled]


def scale_bbox_to_transformed_image(
    top: float, left: float, bottom: float, right: float
) -> List[int]:
    top_scaled = original_to_transformed_w_coords(np.array([top / 2]))[0]
    left_scaled = original_to_transformed_h_coords(np.array([left / 2]))[0]
    bottom_scaled = original_to_transformed_w_coords(np.array([bottom / 2]))[0]
    right_scaled = original_to_transformed_h_coords(np.array([right / 2]))[0]
    return [top_scaled, left_scaled, bottom_scaled, right_scaled]


# Copied from transformers.models.detr.image_processing_detr.max_across_indices
def max_across_indices(values: Iterable[Any]) -> List[Any]:
    """
    Return the maximum value across all indices of an iterable of values.
    """
    return [max(values_i) for values_i in zip(*values)]


# Copied from transformers.models.detr.image_processing_detr.get_max_height_width
def get_max_height_width(
    images: List[np.ndarray],
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> List[int]:
    """
    Get the maximum height and width across all images in a batch.
    """
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(images[0])

    if input_data_format == ChannelDimension.FIRST:
        _, max_height, max_width = max_across_indices([img.shape for img in images])
    elif input_data_format == ChannelDimension.LAST:
        max_height, max_width, _ = max_across_indices([img.shape for img in images])
    else:
        raise ValueError(f"Invalid channel dimension format: {input_data_format}")
    return (max_height, max_width)


# Copied from transformers.models.detr.image_processing_detr.make_pixel_mask
def make_pixel_mask(
    image: np.ndarray,
    output_size: Tuple[int, int],
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> np.ndarray:
    """
    Make a pixel mask for the image, where 1 indicates a valid pixel and 0 indicates padding.

    Args:
        image (`np.ndarray`):
            Image to make the pixel mask for.
        output_size (`Tuple[int, int]`):
            Output size of the mask.
    """
    input_height, input_width = get_image_size(image, channel_dim=input_data_format)
    mask = np.zeros(output_size, dtype=np.int64)
    mask[:input_height, :input_width] = 1
    return mask


class FuyuProcessor(ProcessorMixin):
    r"""
    Constructs a Fuyu processor which wraps a Fuyu image processor and a Llama tokenizer into a single processor.

    [`FuyuProcessor`] offers all the functionalities of [`FuyuImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~FuyuProcessor.__call__`] and [`~FuyuProcessor.decode`] for more information.

    Args:
        image_processor ([`FuyuImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`]):
            The tokenizer is a required input.
    """
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "FuyuImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor=image_processor, tokenizer=tokenizer)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_tokens_to_generate = 10
        self.max_position_embeddings = (
            16384  # TODO Can't derive this from model files: where to set it?
        )
        self.image_processor = FuyuImageProcessor()

    def _process_images(self, images):
        """Utility function to preprocess the images and extract necessary information about original formats."""
        batch_images = []
        image_unpadded_heights = []
        image_unpadded_widths = []

        for image in images:
            image = to_numpy_array(image)
            if not is_scaled_image(image):
                image = image / 255.0
            channel_dimension = infer_channel_dimension_format(image, 3)
            if channel_dimension == ChannelDimension.FIRST:
                width_index = 2
                height_index = 1
            elif channel_dimension == ChannelDimension.LAST:
                width_index = 1
                height_index = 0

            image_unpadded_widths.append([image.shape[width_index]])
            image_unpadded_heights.append([image.shape[height_index]])

            # Reproduct adept padding sampler
            padded_image = self.image_processor.apply_transformation(image)

            tensor_img = torch.Tensor(padded_image).permute(2, 0, 1)
            batch_images.append([tensor_img])

        return (
            batch_images,
            torch.Tensor(image_unpadded_heights),
            torch.Tensor(image_unpadded_widths),
        )

    def __call__(self, text=None, images=None, return_tensors=None, **kwargs):
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to
        encode the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        FuyuImageProcessor's [`~FuyuImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `List[PIL.Image.Image]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.

            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        if text is None and images is None:
            raise ValueError(
                "You have to specify either text or images. Both cannot be none."
            )
        if text is not None and images is not None:
            if isinstance(text, str):
                prompts = [[text]]
            elif isinstance(text, list):
                prompts = [[text_seq] for text_seq in text]
            batch_images = []
            if isinstance(images, PIL.Image.Image):
                images = [images]
            if isinstance(images, list):
                (
                    batch_images,
                    image_unpadded_heights,
                    image_unpadded_widths,
                ) = self._process_images(images)
                # image_unpadded_heights = image_unpadded_heights.unsqueeze(0)
                # image_unpadded_widths = image_unpadded_widths.unsqueeze(0)
            else:
                raise ValueError(
                    "images must be a list of ndarrays or PIL Images to be processed."
                )

            # Note: the original adept code has a handling of image_unpadded_h and w, but it doesn't seem to hold
            # when there are several different size subsequences per batch. The current implementation reflects
            # that limitation and should be documented.
            #
            self.subsequence_length = 1  # Each batch contains only one sequence.
            self.batch_size = len(batch_images)
            # FIXME max_tokens_to_generate is embedded into this processor's call.
            prompt_tokens, prompts_length = _tokenize_prompts_with_image_and_batch(
                tokenizer=self.tokenizer,
                prompts=prompts,
                transformed_images=batch_images,
                max_tokens_to_generate=self.max_tokens_to_generate,
                max_position_embeddings=self.max_position_embeddings,
                add_BOS=True,
                add_beginning_of_answer_token=True,
            )
            # same so far

            # This is 1 if there is an image per subsequence, else 0. [batch, 1, presence]
            # the remainder of current image processing logic assumes subsequence_size = 1.
            # Here it is OK as the model cannot handle > 1 subsequences
            # the image could be absent however and image presence should be inferred from user batch input
            # hence this code assumes the images are present. Use an assert?

            image_present = torch.ones(self.batch_size, 1, 1)

            image_placeholder_id = self.tokenizer(
                "|SPEAKER|", add_special_tokens=False
            )["input_ids"][1]
            image_newline_id = self.tokenizer("|NEWLINE|", add_special_tokens=False)[
                "input_ids"
            ][1]
            tensor_batch_images = torch.stack(
                [img[0] for img in batch_images]
            ).unsqueeze(1)
            model_image_input = self.image_processor.process_images_for_model_input(
                image_input=tensor_batch_images,
                image_present=image_present,
                image_unpadded_h=image_unpadded_heights,
                image_unpadded_w=image_unpadded_widths,
                image_patch_dim_h=30,
                image_patch_dim_w=30,
                image_placeholder_id=image_placeholder_id,
                image_newline_id=image_newline_id,
                variable_sized=True,
            )

            image_padded_unpacked_tokens = construct_full_unpacked_stream(
                num_real_text_tokens=prompts_length,
                input_stream=prompt_tokens,
                image_tokens=model_image_input["image_input_ids"],
                batch_size=self.batch_size,
                num_sub_sequences=self.subsequence_length,
            )
            # Construct inputs for image patch indices.
            unpacked_image_patch_indices_per_batch = construct_full_unpacked_stream(
                num_real_text_tokens=prompts_length,
                input_stream=torch.full_like(prompt_tokens, -1),
                image_tokens=model_image_input["image_patch_indices_per_batch"],
                batch_size=self.batch_size,
                num_sub_sequences=self.subsequence_length,
            )
            max_prompt_length = max(x.shape[-1] for x in image_padded_unpacked_tokens)
            max_seq_len_batch = min(
                max_prompt_length + self.max_tokens_to_generate,
                self.max_position_embeddings,
            )
            all_bi_tokens_to_place = []
            for bi in range(self.batch_size):
                tokens_to_place = min(
                    max_seq_len_batch, max(0, image_padded_unpacked_tokens[bi].shape[0])
                )
                all_bi_tokens_to_place.append(tokens_to_place)

            # Use same packing logic for the image patch indices.
            image_patch_input_indices = full_unpacked_stream_to_tensor(
                all_bi_tokens_to_place=all_bi_tokens_to_place,
                full_unpacked_stream=unpacked_image_patch_indices_per_batch,
                fill_value=-1,
                batch_size=self.batch_size,
                new_seq_len=max_seq_len_batch,
                offset=0,
            )

            image_patches_tensor = torch.stack(
                [img[0] for img in model_image_input["image_patches"]]
            ).unsqueeze(1)
            return {
                "input_ids": image_padded_unpacked_tokens[0].unsqueeze(0),
                "image_patches": image_patches_tensor[0][0].unsqueeze(0),
                "image_patches_indices": image_patch_input_indices,
            }

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BertTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BertTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)


# FuyuImageProcessor example
image_processor = FuyuImageProcessor()
image = PIL.Image.open("agorabanner.png")
test = image_processor.apply_transformation(image)
