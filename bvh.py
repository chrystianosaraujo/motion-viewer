class BVHNode:
	def __init__(self):
		self.name = None
		self.offset = [None, None, None]

		self.offx_idx = None
		self.offy_idx = None
		self.offz_idx = None
		self.rotx_idx = None
		self.roty_idx = None
		self.rotz_idx = None

		self.is_ee = False
		self.ee_offset = [None, None, None]

		self.parent = None
		self.children = []

class BVHFormatError(Exception):
	pass

# -> ([BVHNode], float, [[float]])
# OSError if file does not exist
# BVHFormatError if file is incorrect
def import_bvh(path):
	roots = []
	frame_time = None
	motion_data = []

	with open(path) as f:
		if not next(f).startswith('HIERARCHY'):
			raise BVHFormatError('Missing Hierarchy section')

		line = None
		cur_node = None
		ignore_closing_bracket = False
		tot_channels = 0
		num_frames = None

		# TODO
		# Add expected_next_line to ensure format correctness
		while True:
			line = next(f).strip()
			if line.startswith('ROOT'):
				els = line.split(' ')
				if len(els) != 2: 
					raise BVHFormatError('Root name not specified')

				cur_node = BVHNode()
				cur_node.name = els[1]
				roots.append(cur_node)

			elif line.startswith('OFFSET'):
				els = line.split(' ')
				if len(els) != 4:
					raise BVHFormatError('Invalid offset for joint {}'.format(cur_node.name))

				if cur_node.is_ee:
					cur_node.ee_offset = [float(els[1]), float(els[2]), float(els[3])]
				else:
					cur_node.offset = [float(els[1]), float(els[2]), float(els[3])]

			elif line.startswith('CHANNELS'):
				els = line.split(' ')
				if len(els) < 2:
					raise BVHFormatError('Invalid CHANNELS line for joint {}'.format(cur_node.name))

				num_channels = int(els[1])
				if len(els) != num_channels + 2:
					raise BVHFormatError('Invalid CHANNELS line for joint {}'.format(cur_node.name))
			
				for (ii, el) in enumerate(els[2:]):
					if el == 'Xposition':
						cur_node.offx_idx = tot_channels + ii
					if el == 'Yposition':
						cur_node.offy_idx = tot_channels + ii
					if el == 'Zposition':
						cur_node.offz_idx = tot_channels + ii
					if el == 'Xrotation':
						cur_node.rotx_idx = tot_channels + ii
					if el == 'Yrotation':
						cur_node.roty_idx = tot_channels + ii
					if el == 'Zrotation':
						cur_node.rotz_idx = tot_channels + ii
				tot_channels += num_channels

			elif line.startswith('JOINT'):
				els = line.split(' ', 1)
				if len(els) != 2:
					raise BVHFormatError('Joint name not specified')

				new_node = BVHNode()
				new_node.name = els[1]
				cur_node.children.append(new_node)
				new_node.parent = cur_node
				cur_node = new_node

			elif line.startswith('End Site'):
				cur_node.is_ee = True
				ignore_closing_bracket = True

			elif line.startswith('{'):
				pass

			elif line.startswith('}'):
				if ignore_closing_bracket:
					ignore_closing_bracket = False
					continue

				cur_node = cur_node.parent

			else:
				break

		if not line.startswith('MOTION'):
			raise BVHFormatError('Missing Motion section')

		line = next(f).strip()
		if not line.startswith('Frames:'):
			raise BVHFormatError('Missing number of frames')
		num_frames = int(line[7:]) # ValueError

		line = next(f).strip()
		if not line.startswith('Frame Time:'):
			raise BVHFormatError('Missing frame time')
		frame_time = float(line[11:])		

		frame = 0
		for line in f:
			if frame >= num_frames:
				raise BVHFormatError('More frames than expected. Although not an error perse it means that the file header is wrong')

			values = line.split(' ')
			if len(values) != tot_channels:
				raise BVHFormatError('Unexpected number of channels in frame: {}'.format(frame))

			fvalues = [float(v) for v in values]
			motion_data.append(fvalues)

			frame += 1

	print('Imported {}'.format(path))
	print('Num channels: {}'.format(tot_channels))
	print('Num frames: {}'.format(num_frames))
	print('Frame time: {}'.format(frame_time))
	return (roots, frame_time, motion_data)