import torch


__all__ = ['add_video_gaussian_noise', 'variable_to_cv2_image']


def add_video_gaussian_noise(seq_frames, noise_std):
	assert len(seq_frames.shape) == 5, "Dimension of sequence frames must be [N, T, C, H, W]" 

	numframes, seq_length, C, H, W = seq_frames.shape

	# std dev for each sequence
	if isinstance(noise_std, (list, tuple)):
		stdmin, stdmax = noise_std
		stdn = torch.empty((numframes, 1, 1, 1)).uniform_(stdmin, stdmax).to(device=seq_frames.device)
	else:
		stdn = torch.FloatTensor(noise_std)

	noise = torch.zeros_like(seq_frames)
	noise = torch.normal(mean=noise, std=stdn.expand_as(noise)).to(device=seq_frames.device)

	seqn_frames = seq_frames + noise
	noise_map = stdn.expand(numframes, 1, H, W).to(device=seq_frames.device)

	return seqn_frames, noise_map


def variable_to_cv2_image(invar, conv_rgb_to_bgr=True):
	""" Converts a torch.autograd.Variable to an OpenCV image

	Args:
		invar: a torch.autograd.Variable
		conv_rgb_to_bgr: boolean. If True, convert output image from RGB to BGR color space
	Returns:
		a HxWxC uint8 image
	"""
	assert torch.max(invar) <= 1.0, "MAXinvar {}".format(torch.max(invar))

	size4 = len(invar.size()) == 4
	if size4:
		nchannels = invar.size()[1]
	else:
		nchannels = invar.size()[0]

	if nchannels == 1:
		if size4:
			res = invar.data.cpu().numpy()[0, 0, :]
		else:
			res = invar.data.cpu().numpy()[0, :]
		res = (res*255.).clip(0, 255).astype(np.uint8)
	elif nchannels == 3:
		if size4:
			res = invar.data.cpu().numpy()[0]
		else:
			res = invar.data.cpu().numpy()
		res = res.transpose(1, 2, 0)
		res = (res*255.).clip(0, 255).astype(np.uint8)
		if conv_rgb_to_bgr:
			res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
	else:
		raise Exception('Number of color channels not supported')
	return res