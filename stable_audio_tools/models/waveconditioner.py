from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from .process_mm_info import process_mm_info
import math
from diffusers.models.normalization import RMSNorm
import ffmpeg
import io
import random
import time
import sys
sys.path.append("/data/code/AR/VQ_tok")
from ibq import VQConvProjector
class WaveEncoderConditioner(nn.Module):

    def __init__(
            self,
            output_dim: int,
            enable_grad: bool = False,
            enable_connecter_gard: bool = True,
            vq_quant: bool = False,
            project_out: bool = False
    ):
        super().__init__()
        self.input_dim = 128
        self.enable_grad = enable_grad
        self.enable_connecter_gard = enable_connecter_gard
        # super().__init__(input_dim, output_dim, project_out=project_out)
        # random sleep for 0～5s
        time.sleep(random.randint(0,5))
        self.processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-3B", torch_dtype="auto", device_map="auto")

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
                z_channels=768,    # 2048
                codebook_size=8192,  # codebook size: 16384
                codebook_dim=768,  # 2048
                use_transformer=False,
                # config=copy.deepcopy(config),  # use the same config as the model
                recon=False,     # whether to use the recon loss
            )

    def forward(self, prompts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        self.model.to(device)
        self.connector.to(device)
        conversation = []
        for temp in prompts:
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

        attention_mask = inputs["feature_attention_mask"].to(device)
        embeddings = inputs["input_features"].to(device)

        
        embeddings = embeddings.permute(0, 2, 1)

        embeddings = self.connector(embeddings)
        embeddings = embeddings * attention_mask.unsqueeze(-1).float()

        out_dype = next(self.connector.parameters()).dtype
        embeddings = embeddings.to(out_dype)
        
        if self.vq_quant:
            valid_lengths = attention_mask.sum(dim=1).long()  # [batch]
            
            cu_seqlens = torch.cat([
                torch.zeros(1, device=device, dtype=torch.long), #[0]
                valid_lengths.cumsum(dim=0) # [batch]
            ], dim=0)  # [batch + 1]
            # print(f"embeddings: {embeddings.shape}, cu_seqlens: {cu_seqlens.shape}, attention_mask: {attention_mask.shape}")
            # [batch, 30000, hidden_dim] to [batch*30000, hidden_dim]
            embeddings_quant = embeddings.reshape(-1, embeddings.shape[-1])
            quant_code, code_idx, vq_loss = self.ibq_projection(
                # embeddings,
                embeddings_quant,
                cu_seqlens=cu_seqlens,
                position_embeddings=None
            )
            # print(f"quant_code: {quant_code.shape}, code_idx: {code_idx.shape}, vq_loss: {vq_loss}")
            # quant code back to [batch, 30000, hidden_dim]
            quant_code = quant_code.reshape(embeddings.shape[0], embeddings.shape[1], -1)
            # return quant_code, attention_mask, vq_loss
            return quant_code, attention_mask

        return embeddings, attention_mask