import nvidia

import nvidia.dali.ops as ops
import nvidia.dali.types as types

from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin import pytorch


__all__ = ["DaliVideoLoader",]

class VideoReaderPipeline(Pipeline):
	""" Pipeline for reading H264 videos based on NVIDIA DALI.

    Returns a batch of sequences of `sequence_length` frames of shape [N, F, C, H, W] [N, F, H, W, C]
	(N being the batch size and T the number of frames/ sequence length). Frames are RGB uint8.

	Args:
		file_list: (str or list of str)
				File path containing list of the video files path to load.
		batch_size: (int)
				Size of the batches
		sequence_length: (int)
				Frames to load per sequence.
		num_threads: (int)
				Number of threads.
		device_id: (int)
				GPU device ID where to load the sequences.
		random_shuffle: (bool, optional, default=True)
				Whether to randomly shuffle data.
		step: (int, optional, default=-1)
				Frame interval between each sequence (if `step` < 0, `step` is set to `sequence_length`).
		stride: (int, optional, default=-1)
				Distance between consecutive frames in sequence
	"""
	def __init__(self, file_list, batch_size, sequence_length, num_threads, device_id,
				crop_size, step=-1, stride=1, random_shuffle=True):
		super(VideoReaderPipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
		
		# Define VideoReader
		self.reader = ops.VideoReader(device="gpu",
										file_list=file_list,
										sequence_length=sequence_length,
										normalized=False,
										random_shuffle=random_shuffle,
										image_type=types.RGB,
										dtype=types.UINT8,
										step=step,
										stride=stride,
										initial_fill=16)

		# Define crop, mirror and normalisation operations to apply to every sequence
		self.crop = ops.Crop(device="gpu",
							crop = crop_size,
							output_dtype = types.FLOAT)

		self.transpose = ops.Transpose(device="gpu", perm=[0, 3, 1, 2]) # [N F C H W]

		self.uniform = ops.Uniform(range=(0.2, 1.0)) # used for random crop

	def define_graph(self):
		""" Definition of the graph--events that will take place at every sampling of the dataloader.
		The random crop and permute operations will be applied to the sampled sequence.
		"""
		input_frames, labels = self.reader(name="Reader")
		cropped = self.crop(input_frames,
							crop_pos_x=self.uniform(),
							crop_pos_y=self.uniform())
		output_frames = self.transpose(input_frames)
		return output_frames, labels


class DaliVideoLoader(object):
	""" Sequence dataloader.
	Args:
		file_list (str)
			Path to file containing  list
		batch_size: (int)
			Size of the batches
		sequence_length: (int)
			Frames to load per sequence
		epoch_size: (int, optional, default=-1)
			Size of the epoch. If epoch_size <= 0, epoch_size will default to the size of VideoReaderPipeline
		random_shuffle (bool, optional, default=True)
			Whether to randomly shuffle data.
		step: (int, optional, default=-1)
				Frame interval between each sequence (if `step` < 0, `step` is set to `sequence_length`).
		stride: (int, optional, default=1)
				Distance between consecutive frames in sequence
	"""
	def __init__(self, file_list, batch_size, sequence_length, crop_size, epoch_size=-1,
                random_shuffle=True, step=-1, stride=1, device_id=0):
		# Define and build pipeline
		self.pipeline = VideoReaderPipeline(file_list=file_list,
											batch_size=batch_size,
											sequence_length=sequence_length,
											num_threads=2,
											device_id=device_id,
											crop_size = crop_size,
											random_shuffle=random_shuffle,
											step=step,
											stride=stride)
		self.pipeline.build()

		# Define size of epoch
		if epoch_size <= 0:
			self.epoch_size = self.pipeline.epoch_size("Reader")
		else:
			self.epoch_size = epoch_size

		# Define Pytorch tensor iterator
		self.dali_iterator = pytorch.DALIGenericIterator(pipelines=self.pipeline,
														output_map=["data", "labels"],
														size=self.epoch_size,
														auto_reset=True)

	def __len__(self):
		return self.epoch_size

	def __iter__(self):
		self._iterator = self.dali_iterator.__iter__()
		return self

	def __next__(self):
		pipeout = next(self._iterator)
		return pipeout[-1]["data"].float() / 255.0, pipeout[-1]["labels"].squeeze().long()
