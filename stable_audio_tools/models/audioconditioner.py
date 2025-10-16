from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from .process_mm_info_ import process_mm_info
import math
from diffusers.models.normalization import RMSNorm
import ffmpeg
import io
import random
import time
import sys
import gc
sys.path.append("/data/code/AR/VQ_tok")
from ibq import VQConvProjector
import torch
import torch.nn as nn
import typing as tp
from typing import Any, Callable, Optional, Union

class AudioEncoderConditioner(nn.Module):
    def __init__(
            self,
            output_dim: int,
            enable_connecter_gard: bool = True,
            vq_quant: bool = True,
    ):
        super().__init__()
        self.input_dim = 1280
        self.enable_connecter_gard = enable_connecter_gard
        # random sleep for 0～5s
        time.sleep(random.randint(0,5))
        self.processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")

        model_temp = Qwen2_5OmniForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-3B", torch_dtype="auto", device_map="cpu")
        self.audio_tower = model_temp.thinker.audio_tower
        self.audio_tower.train(False).requires_grad_(False)
        del model_temp
        gc.collect()
        torch.cuda.empty_cache()

        self.connector_in_dim = self.input_dim
        self.connector_out_dim = output_dim
        norm = RMSNorm(self.connector_out_dim, eps=1e-5, elementwise_affine=True)
        with torch.no_grad():
            norm.weight.fill_(math.sqrt(5.5))
        self.connector = nn.Sequential(
            nn.Linear(self.connector_in_dim, self.connector_out_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.connector_out_dim, self.connector_out_dim),
            norm,
        ).train(enable_connecter_gard).requires_grad_(enable_connecter_gard)
        
        self.vq_quant = vq_quant
        if vq_quant:
            print("*********WaveEncoderConditioner using VQ quantization")
            self.ibq_projection = VQConvProjector(
                z_channels=self.connector_out_dim,    # 768
                codebook_size=16384,  # codebook size: 16384
                codebook_dim=self.connector_out_dim,  # 768
                use_transformer=False,
                # config=copy.deepcopy(config),  # use the same config as the model
                recon=False,     # whether to use the recon loss
            )

    def forward(self, prompts: tp.List[str], device: tp.Union[torch.device, str], demo: bool=False) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        self.audio_tower.to(device)
        self.connector.to(device)
        self.ibq_projection.to(device)
        conversation = []
        for temp in prompts:
            if type(temp) is str:
                audio_path = temp
                audio_start = 0
                audio_end = None
            else:
                audio_path, audio_start, audio_end = temp
            if audio_start != 0: print("*******************audio_start != 0")
            if audio_path.split(".")[-1] == 'mp4':
                conversation.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": audio_path, "video_start": audio_start, "video_end": audio_end},
                        ],
                    }
                )
            else:
                conversation.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio": audio_path, "audio_start": audio_start, "audio_end": audio_end},
                        ],
                    }
                )
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
        # /home/yifanyang/miniconda3/envs/sao/lib/python3.10/site-packages/qwen_omni_utils/v2_5/audio_process.py
        inputs = self.processor(text="Hello", audio=audios, images=None, videos=None, return_tensors="pt", padding=True, use_audio_in_video=False)

        audio_features = self.get_audio_features(
            input_features=inputs["input_features"].to(device),
            feature_attention_mask=inputs["feature_attention_mask"].to(device),
            audio_feature_lengths=None,
        )
        # N \times [seq, 1280] -> embeddings [batch, 2000, 1280]  attention_mask [batch, 2000]
        embeddings = torch.zeros((len(prompts), 2000, self.input_dim), device=device)
        attention_mask = torch.zeros((len(prompts), 2000), device=device, dtype=torch.long)
        for i,audio_feature in enumerate(audio_features):
            embeddings[i, :audio_feature.shape[0], :] = audio_feature
            attention_mask[i, :audio_feature.shape[0]] = 1
        # print(f"embeddings {embeddings[0][0]}")
        
        # embeddings = embeddings.permute(0, 2, 1)

        embeddings = self.connector(embeddings)
        embeddings = embeddings * attention_mask.unsqueeze(-1).float()

        if self.vq_quant:
            valid_lengths = attention_mask.sum(dim=1).long()  # [batch]
            cu_seqlens = torch.cat([
                torch.zeros(1, device=device, dtype=torch.long), #[0]
                valid_lengths.cumsum(dim=0) # [batch]
            ], dim=0)  # [batch + 1]
            # embeddings_quant should be [sum(valid_lengths), dim]
            embeddings_quant = torch.zeros((cu_seqlens[-1], embeddings.shape[2]), device=device)
            for i in range(embeddings.shape[0]):
                embeddings_quant[cu_seqlens[i]:cu_seqlens[i+1], :] = embeddings[i, :valid_lengths[i], :]
            # print(f"embeddings_quant {embeddings_quant.shape, embeddings_quant[0][:10]}, cu_seqlens {cu_seqlens}, valid_lengths {valid_lengths}")
            quant_code, code_idx, vq_loss = self.ibq_projection(
                # embeddings,
                embeddings_quant,
                cu_seqlens=cu_seqlens,
                position_embeddings=None,
                demo=demo,
            )
            # back to [batch, seqlen, dim]
            # print(f"quant_code before reshape {quant_code.shape, quant_code[0][:10]}")
            quant_code_batch = torch.zeros_like(embeddings)
            for i in range(embeddings.shape[0]):
                quant_code_batch[i, :valid_lengths[i], :] = quant_code[cu_seqlens[i]:cu_seqlens[i+1], :]
            quant_code = quant_code_batch
            # print(f"quant_code {quant_code.shape, quant_code[0][0][:10]}, code_idx {code_idx.shape}, vq_loss {vq_loss}")
            out_dtype = next(self.connector.parameters()).dtype
            quant_code = quant_code.to(out_dtype)
            return quant_code, attention_mask

        out_dtype = next(self.connector.parameters()).dtype
        embeddings = embeddings.to(out_dtype)

        return embeddings, attention_mask

    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        feature_attention_mask: Optional[torch.LongTensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
    ):
        """
        Encodes audios into continuous embeddings that can be forwarded to the language model.

        Args:
            input_features (`torch.FloatTensor`):
                The tensors corresponding to the input audios.
            feature_attention_mask (`torch.LongTensor`, *optional*):
                Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:
            audio_feature_lengths (`torch.LongTensor` of shape `(num_audios)`, *optional*):
                The length of feature shape of each audio in LLM.
        """
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        else:
            audio_feature_lengths = None

        audio_feat_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
            audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
        )
        feature_lens = audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
        audio_outputs = self.audio_tower(
            input_features,
            feature_lens=feature_lens,
            aftercnn_lens=audio_feat_lengths,
        )

        audio_features = audio_outputs.last_hidden_state

        # if audio_features.shape[0] != sum(audio_output_lengths.tolist()):
        #     raise ValueError("length of audio_features should match audio_output_lengths")

        return audio_features