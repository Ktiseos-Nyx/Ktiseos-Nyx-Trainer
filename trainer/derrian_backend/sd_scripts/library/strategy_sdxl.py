import os
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from library.strategy_base import TokenizeStrategy, TextEncodingStrategy, TextEncoderOutputsCachingStrategy


from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


TOKENIZER1_PATH = "openai/clip-vit-large-patch14"
TOKENIZER2_PATH = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"


class SdxlTokenizeStrategy(TokenizeStrategy):
    def __init__(
        self,
        max_length: Optional[int],
        tokenizer_cache_dir: Optional[str] = None,
        use_attention_mask: bool = True,
    ) -> None:
        self.tokenizer1 = self._load_tokenizer(CLIPTokenizer, TOKENIZER1_PATH, tokenizer_cache_dir=tokenizer_cache_dir)
        self.tokenizer2 = self._load_tokenizer(CLIPTokenizer, TOKENIZER2_PATH, tokenizer_cache_dir=tokenizer_cache_dir)
        self.tokenizer2.pad_token_id = 0  # use 0 as pad token for tokenizer2
        self.use_attention_mask = use_attention_mask

        if max_length is None:
            self.max_length = self.tokenizer1.model_max_length
        else:
            self.max_length = max_length + 2

    def tokenize(
        self, text: Union[str, List[str]]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
        text = [text] if isinstance(text, str) else text

        tokenizer_1_ids, tokenizer_1_mask = self._tokenize_tags(self.tokenizer1, text)
        tokenizer_2_ids, tokenizer_2_mask = self._tokenize_tags(self.tokenizer2, text)

        B = len(text)
        tokenizer_1_ids = tokenizer_1_ids.view(B, -1, tokenizer_1_ids.shape[-1])
        tokenizer_2_ids = tokenizer_2_ids.view(B, -1, tokenizer_2_ids.shape[-1])

        if self.use_attention_mask:
            tokenizer_1_mask = tokenizer_1_mask.squeeze(0)
            tokenizer_2_mask = tokenizer_2_mask.squeeze(0)
        else:
            tokenizer_1_mask = None
            tokenizer_2_mask = None

        ret = ((tokenizer_1_ids, tokenizer_2_ids), (tokenizer_1_mask, tokenizer_2_mask))

        return ret

    def tokenize_with_weights(self, text: str | List[str]) -> Tuple[List[torch.Tensor]]:
        text = [text] if isinstance(text, str) else text
        tokens1_list, tokens2_list = [], []
        weights1_list, weights2_list = [], []
        for t in text:
            tokens1, weights1 = self._get_input_ids(self.tokenizer1, t, self.max_length, weighted=True)
            tokens2, weights2 = self._get_input_ids(self.tokenizer2, t, self.max_length, weighted=True)
            tokens1_list.append(tokens1)
            tokens2_list.append(tokens2)
            weights1_list.append(weights1)
            weights2_list.append(weights2)
        return [torch.stack(tokens1_list, dim=0), torch.stack(tokens2_list, dim=0)], [
            torch.stack(weights1_list, dim=0),
            torch.stack(weights2_list, dim=0),
        ]


class SdxlTextEncodingStrategy(TextEncodingStrategy):
    def __init__(self) -> None:
        pass

    def _pool_workaround(
        self, text_encoder: CLIPTextModelWithProjection, last_hidden_state: torch.Tensor, input_ids: torch.Tensor, eos_token_id: int
    ):
        r"""
        workaround for CLIP's pooling bug: it returns the hidden states for the max token id as the pooled output
        instead of the hidden states for the EOS token
        If we use Textual Inversion, we need to use the hidden states for the EOS token as the pooled output

        Original code from CLIP's pooling function:

        \# text_embeds.shape = [batch_size, sequence_length, transformer.width]
        \# take features from the eot embedding (eot_token is the highest number in each sequence)
        \# casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
        ]
        """

        # input_ids: b*n,77
        # find index for EOS token

        # Following code is not working if one of the input_ids has multiple EOS tokens (very odd case)
        # eos_token_index = torch.where(input_ids == eos_token_id)[1]
        # eos_token_index = eos_token_index.to(device=last_hidden_state.device)

        # Create a mask where the EOS tokens are
        eos_token_mask = (input_ids == eos_token_id).int()

        # Use argmax to find the last index of the EOS token for each element in the batch
        eos_token_index = torch.argmax(eos_token_mask, dim=1)  # this will be 0 if there is no EOS token, it's fine
        eos_token_index = eos_token_index.to(device=last_hidden_state.device)

        # get hidden states for EOS token
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), eos_token_index
        ]

        # apply projection: projection may be of different dtype than last_hidden_state
        pooled_output = text_encoder.text_projection(pooled_output.to(text_encoder.text_projection.weight.dtype))
        pooled_output = pooled_output.to(last_hidden_state.dtype)

        return pooled_output

    def stitch_cross_attention_mask(
        self,
        cross_attention_mask_logical: Optional[torch.Tensor],
        tokenizer_max_length: int = 77,
    ) -> Optional[torch.Tensor]:
        """
        Mirrors the stitching logic of Kohya's _get_hidden_states_sdxl to correctly
        reshape the cross-attention mask.

        Args:
            cross_attention_mask_logical (torch.Tensor): The granular mask with a
                logical shape of [B, N, S].
            tokenizer_max_length (int, optional): The sequence length of one chunk (S).
                Defaults to 77.

        Returns:
            torch.Tensor | None: The stitched mask when attention masking is enabled, otherwise None.
        """
        if cross_attention_mask_logical is None:
            return None

        B, N, S = cross_attention_mask_logical.shape
        assert S == tokenizer_max_length

        # Reshape to [B, N * S] first, as this is easier to slice
        reshaped_mask = cross_attention_mask_logical.view(B, N * S)

        states_list = []
        # 1. Add the mask for the first token (<BOS>)
        states_list.append(reshaped_mask[:, 0].unsqueeze(1))

        # 2. Loop through the chunks, grabbing the content part of the mask
        for i in range(N):
            chunk_start_index = i * S
            # Slice from the token after <BOS> to the token before <EOS>
            chunk_content_mask = reshaped_mask[:, chunk_start_index + 1: chunk_start_index + S - 1]
            states_list.append(chunk_content_mask)

        # 3. Add the mask for the very last token of the last chunk (<EOS>)
        states_list.append(reshaped_mask[:, -1].unsqueeze(1))

        # 4. Concatenate to get the final stitched mask
        stitched_mask = torch.cat(states_list, dim=1)

        return stitched_mask

    def _get_hidden_states_sdxl(
        self,
        input_ids1: torch.Tensor,
        input_ids2: torch.Tensor,
        attention_mask1: Optional[torch.Tensor],
        attention_mask2: Optional[torch.Tensor],
        tokenizer1: CLIPTokenizer,
        tokenizer2: CLIPTokenizer,
        text_encoder1: Union[CLIPTextModel, torch.nn.Module],
        text_encoder2: Union[CLIPTextModelWithProjection, torch.nn.Module],
        unwrapped_text_encoder2: Optional[CLIPTextModelWithProjection] = None,
    ):
        # input_ids: b,n,77 -> b*n, 77
        b_size = input_ids1.size()[0]
        if input_ids1.size()[1] == 1:
            max_token_length = None
        else:
            max_token_length = input_ids1.size()[1] * input_ids1.size()[2]
        input_ids1 = input_ids1.reshape((-1, tokenizer1.model_max_length))  # batch_size*n, 77
        input_ids2 = input_ids2.reshape((-1, tokenizer2.model_max_length))  # batch_size*n, 77
        input_ids1 = input_ids1.to(text_encoder1.device)
        input_ids2 = input_ids2.to(text_encoder2.device)

        # text_encoder1
        if attention_mask1 is not None:
            attention_mask1_reshaped = attention_mask1.reshape((-1, tokenizer1.model_max_length)).to(
                text_encoder1.device
            )
            enc_out = text_encoder1(
                input_ids1, output_hidden_states=True, return_dict=True, attention_mask=attention_mask1_reshaped
            )
        else:
            enc_out = text_encoder1(input_ids1, output_hidden_states=True, return_dict=True)
        hidden_states1 = enc_out["hidden_states"][11]

        # text_encoder2
        if attention_mask2 is not None:
            attention_mask2_reshaped = attention_mask2.reshape((-1, tokenizer2.model_max_length)).to(
                text_encoder2.device
            )
            enc_out = text_encoder2(
                input_ids2, output_hidden_states=True, return_dict=True, attention_mask=attention_mask2_reshaped
            )
        else:
            enc_out = text_encoder2(input_ids2, output_hidden_states=True, return_dict=True)
        hidden_states2 = enc_out["hidden_states"][-2]  # penuultimate layer

        # pool2 = enc_out["text_embeds"]
        unwrapped_text_encoder2 = unwrapped_text_encoder2 or text_encoder2
        pool2 = self._pool_workaround(unwrapped_text_encoder2, enc_out["last_hidden_state"], input_ids2, tokenizer2.eos_token_id)

        # b*n, 77, 768 or 1280 -> b, n*77, 768 or 1280
        n_size = 1 if max_token_length is None else max_token_length // 75
        hidden_states1 = hidden_states1.reshape((b_size, -1, hidden_states1.shape[-1]))
        hidden_states2 = hidden_states2.reshape((b_size, -1, hidden_states2.shape[-1]))

        if max_token_length is not None:
            # bs*3, 77, 768 or 1024
            # encoder1: <BOS>...<EOS> の三連を <BOS>...<EOS> へ戻す
            states_list = [hidden_states1[:, 0].unsqueeze(1)]  # <BOS>
            for i in range(1, max_token_length, tokenizer1.model_max_length):
                states_list.append(hidden_states1[:, i : i + tokenizer1.model_max_length - 2])  # <BOS> の後から <EOS> の前まで
            states_list.append(hidden_states1[:, -1].unsqueeze(1))  # <EOS>
            hidden_states1 = torch.cat(states_list, dim=1)

            # v2: <BOS>...<EOS> <PAD> ... の三連を <BOS>...<EOS> <PAD> ... へ戻す　正直この実装でいいのかわからん
            states_list = [hidden_states2[:, 0].unsqueeze(1)]  # <BOS>
            for i in range(1, max_token_length, tokenizer2.model_max_length):
                chunk = hidden_states2[:, i : i + tokenizer2.model_max_length - 2]  # <BOS> の後から 最後の前まで
                # this causes an error:
                # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
                # if i > 1:
                #     for j in range(len(chunk)):  # batch_size
                #         if input_ids2[n_index + j * n_size, 1] == tokenizer2.eos_token_id:  # 空、つまり <BOS> <EOS> <PAD> ...のパターン
                #             chunk[j, 0] = chunk[j, 1]  # 次の <PAD> の値をコピーする
                states_list.append(chunk)  # <BOS> の後から <EOS> の前まで
            states_list.append(hidden_states2[:, -1].unsqueeze(1))  # <EOS> か <PAD> のどちらか
            hidden_states2 = torch.cat(states_list, dim=1)

            # pool はnの最初のものを使う
            pool2 = pool2[::n_size]


        attention_mask1 = self.stitch_cross_attention_mask(attention_mask1, tokenizer1.model_max_length)
        attention_mask2 = self.stitch_cross_attention_mask(attention_mask2, tokenizer2.model_max_length)

        return hidden_states1, hidden_states2, pool2, attention_mask1, attention_mask2

    def encode_tokens(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        tokens: list[torch.Tensor],
        attn_masks: Optional[list[Optional[torch.Tensor]]] = None,
    ) -> tuple[List[torch.Tensor], List[Optional[torch.Tensor]]]:
        """
        Args:
            tokenize_strategy: TokenizeStrategy
            models: List of models, [text_encoder1, text_encoder2, unwrapped text_encoder2 (optional)].
                If text_encoder2 is wrapped by accelerate, unwrapped_text_encoder2 is required
            tokens: List of tokens, for text_encoder1 and text_encoder2
        """
        if len(models) == 2:
            text_encoder1, text_encoder2 = models
            unwrapped_text_encoder2 = None
        else:
            text_encoder1, text_encoder2, unwrapped_text_encoder2 = models
        tokens1, tokens2 = tokens
        if attn_masks is None:
            attn_mask1 = None
            attn_mask2 = None
        else:
            attn_mask1, attn_mask2 = attn_masks
        sdxl_tokenize_strategy = tokenize_strategy  # type: SdxlTokenizeStrategy
        tokenizer1, tokenizer2 = sdxl_tokenize_strategy.tokenizer1, sdxl_tokenize_strategy.tokenizer2

        hidden_states1, hidden_states2, pool2, attn_mask1, attn_mask2 = self._get_hidden_states_sdxl(
            tokens1, tokens2, attn_mask1, attn_mask2, tokenizer1, tokenizer2, text_encoder1, text_encoder2, unwrapped_text_encoder2
        )
        return [hidden_states1, hidden_states2, pool2], [attn_mask1, attn_mask2]

    def encode_tokens_with_weights(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        tokens_list: List[torch.Tensor],
        weights_list: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        encoded, attn = self.encode_tokens(tokenize_strategy, models, tokens_list)
        hidden_states1, hidden_states2, pool2 = encoded
        attn1, attn2 = attn

        weights_list = [weights.to(hidden_states1.device) for weights in weights_list]

        # apply weights
        if weights_list[0].shape[1] == 1:  # no max_token_length
            # weights: ((b, 1, 77), (b, 1, 77)), hidden_states: (b, 77, 768), (b, 77, 768)
            hidden_states1 = hidden_states1 * weights_list[0].squeeze(1).unsqueeze(2)
            hidden_states2 = hidden_states2 * weights_list[1].squeeze(1).unsqueeze(2)
        else:
            # weights: ((b, n, 77), (b, n, 77)), hidden_states: (b, n*75+2, 768), (b, n*75+2, 768)
            for weight, hidden_states in zip(weights_list, [hidden_states1, hidden_states2]):
                for i in range(weight.shape[1]):
                    hidden_states[:, i * 75 + 1 : i * 75 + 76] = hidden_states[:, i * 75 + 1 : i * 75 + 76] * weight[
                        :, i, 1:-1
                    ].unsqueeze(-1)

        return [hidden_states1, hidden_states2, pool2]


class SdxlTextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    SDXL_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX = "_te_outputs.npz"

    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: int,
        skip_disk_cache_validity_check: bool,
        is_partial: bool = False,
        is_weighted: bool = False,
        use_attention_mask: bool = True,
    ) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check, is_partial, is_weighted)
        self.use_attention_mask = use_attention_mask

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        return os.path.splitext(image_abs_path)[0] + SdxlTextEncoderOutputsCachingStrategy.SDXL_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX

    def is_disk_cached_outputs_expected(self, npz_path: str):
        if not self.cache_to_disk:
            return False
        if not os.path.exists(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True

        try:
            npz = np.load(npz_path)
            required_keys = {"hidden_state1", "hidden_state2", "pool2"}
            if self.use_attention_mask:
                required_keys.update({"attention_mask1", "attention_mask2"})
            if not required_keys.issubset(npz.files):
                return False
        except Exception as e:
            logger.error(f"Error loading file: {npz_path}")
            raise e

        return True

    def load_outputs_npz(self, npz_path: str) -> List[np.ndarray]:
        data = np.load(npz_path)
        hidden_state1 = data["hidden_state1"]
        hidden_state2 = data["hidden_state2"]
        pool2 = data["pool2"]
        if self.use_attention_mask and "attention_mask1" in data and "attention_mask2" in data:
            attention_mask1 = data["attention_mask1"]
            attention_mask2 = data["attention_mask2"]
            return [hidden_state1, hidden_state2, pool2, attention_mask1, attention_mask2]
        return [hidden_state1, hidden_state2, pool2]

    def cache_batch_outputs(
        self, tokenize_strategy: TokenizeStrategy, models: List[Any], text_encoding_strategy: TextEncodingStrategy, infos: List
    ):
        sdxl_text_encoding_strategy = text_encoding_strategy  # type: SdxlTextEncodingStrategy
        captions = [info.caption for info in infos]

        attn1 = None
        attn2 = None

        if self.is_weighted:
            tokens_list, weights_list = tokenize_strategy.tokenize_with_weights(captions)
            with torch.no_grad():
                hidden_state1, hidden_state2, pool2 = sdxl_text_encoding_strategy.encode_tokens_with_weights(
                    tokenize_strategy, models, tokens_list, weights_list
                )
        else:
            tokens_and_masks = tokenize_strategy.tokenize(captions)
            (tokens1, tokens2), mask_pair = tokens_and_masks
            mask1, mask2 = mask_pair
            if mask1 is not None and mask1.dim() == 2:
                mask1 = mask1.unsqueeze(0)
            if mask2 is not None and mask2.dim() == 2:
                mask2 = mask2.unsqueeze(0)
            mask_args: Optional[list[Optional[torch.Tensor]]] = None
            if mask1 is not None or mask2 is not None:
                mask_args = [mask1, mask2]
            with torch.no_grad():
                encoded, attn = sdxl_text_encoding_strategy.encode_tokens(
                    tokenize_strategy, models, [tokens1, tokens2], mask_args
                )
                hidden_state1, hidden_state2, pool2 = encoded
                attn1, attn2 = attn

        if hidden_state1.dtype == torch.bfloat16:
            hidden_state1 = hidden_state1.float()
        if hidden_state2.dtype == torch.bfloat16:
            hidden_state2 = hidden_state2.float()
        if pool2.dtype == torch.bfloat16:
            pool2 = pool2.float()

        hidden_state1 = hidden_state1.cpu().numpy()
        hidden_state2 = hidden_state2.cpu().numpy()
        pool2 = pool2.cpu().numpy()
        attn1_np = attn1.cpu().numpy() if attn1 is not None else None
        attn2_np = attn2.cpu().numpy() if attn2 is not None else None

        for i, info in enumerate(infos):
            hidden_state1_i = hidden_state1[i]
            hidden_state2_i = hidden_state2[i]
            pool2_i = pool2[i]
            attn1_i = attn1_np[i] if attn1_np is not None else None
            attn2_i = attn2_np[i] if attn2_np is not None else None

            if self.cache_to_disk:
                save_kwargs = {
                    "hidden_state1": hidden_state1_i,
                    "hidden_state2": hidden_state2_i,
                    "pool2": pool2_i,
                }
                if attn1_i is not None and attn2_i is not None:
                    save_kwargs["attention_mask1"] = attn1_i
                    save_kwargs["attention_mask2"] = attn2_i
                np.savez(info.text_encoder_outputs_npz, **save_kwargs)
            else:
                if attn1_i is not None and attn2_i is not None:
                    info.text_encoder_outputs = [hidden_state1_i, hidden_state2_i, pool2_i, attn1_i, attn2_i]
                else:
                    info.text_encoder_outputs = [hidden_state1_i, hidden_state2_i, pool2_i]
