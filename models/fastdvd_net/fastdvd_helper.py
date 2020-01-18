import torch
from .fastdvd_net import FastDVDnet


__all__ = ['fastdvdnet_denoise_sequence',]


def fastdvdnet_denoise_sequence(seqn_frames, noise_map, fastdvd_model):
	"""
	"""
	numframes, seq_length, C, H, W = seqn_frames.shape
	window_size = FastDVDnet.NUM_INPUT_FRAMES
	den_stride = 1

	denseq_length = int(seq_length - window_size + den_stride) // den_stride
	midframe_idx = int(seq_length-1)//2

	inframes_batches = tuple(seqn_frames[:, m:m+window_size, :, :, :] for m in range(0, denseq_length))
	denframes = torch.empty((numframes, denseq_length, C, H, W)).to(seqn_frames.device)

	for i, inframes in enumerate(inframes_batches):
		denframes[:, i, :, :, :] = fastdvd_model(inframes, noise_map)

	del inframes_batches
	torch.cuda.empty_cache()

	return denframes
