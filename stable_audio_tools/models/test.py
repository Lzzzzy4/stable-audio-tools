from waveconditioner import WaveEncoderConditioner

model = WaveEncoderConditioner(output_dim=768)
path = "/blob/vggsound/vggsound_03/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video/A-ZzvGPcdOI_000011.mp4"
output = model([path, path], device="cuda")
print(output[0].shape, output[1].shape)