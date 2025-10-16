import torch
import soundfile as sf
from diffusers import StableAudioPipeline
import inspect
print(inspect.getfile(StableAudioPipeline))

pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
# print(pipe)

# define the prompts
prompt = "123"
negative_prompt = "Low quality."

# set the seed for generator
generator = torch.Generator("cuda").manual_seed(0)

# run the generation
audio = pipe(
    prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=200,
    audio_end_in_s=10.0,
    num_waveforms_per_prompt=1,
    generator=generator,
).audios

# output = audio[0].T.float().cpu().numpy()
# sf.write("hammer.wav", output, pipe.vae.sampling_rate)