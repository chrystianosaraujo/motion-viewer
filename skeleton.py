# Python
import enum
import os

# Motion Viewer
import bvh

# External
import numpy as np

# Here mostly for utility purposes, nodes are not guaranteed to have a name or 
# a type associated with it. Especially useful to identify different body parts
# for rendering
class NodeType(enum.Enum):
	HIP = 0
	ABDOMEN = 1
	CHEST = 2
	NECK = 3
	HEAD = 4
	LEFT_EYE = 5
	RIGHT_EYE = 6
	RIGHT_COLLAR = 7
	RIGHT_SHOULDER = 8
	RIGHT_FOREARM = 9
	RIGHT_HAND = 10
	RIGHT_FINGER_THUMB1 = 11
	RIGHT_FINGER_THUMB2 = 12
	RIGHT_FINGER_INDEX1 = 13
	RIGHT_FINGER_INDEX2 = 14
	RIGHT_FINGER_MID1 = 15
	RIGHT_FINGER_MID2 = 16
	RIGHT_FINGER_RING1 = 17
	RIGHT_FINGER_RING2 = 18
	RIGHT_FINGER_PINKY1 = 19
	RIGHT_FINGER_PINKY2 = 20

	LEFT_COLLAR = 21
	LEFT_SHOULDER = 22
	LEFT_FOREARM = 23
	LEFT_HAND = 24
	LEFT_FINGER_THUMB1 = 25
	LEFT_FINGER_THUMB2 = 26
	LEFT_FINGER_INDEX1 = 27
	LEFT_FINGER_INDEX2 = 28
	LEFT_FINGER_MID1 = 29
	LEFT_FINGER_MID2 = 30
	LEFT_FINGER_RING1 = 31
	LEFT_FINGER_RING2 = 32
	LEFT_FINGER_PINKY1 = 33
	LEFT_FINGER_PINKY2 = 34

class UnsupportedError(Exception):
	pass

class FormatError(Exception):
	pass

# Hierarchical collection of joints with associated motion data
class AnimatedSkeleton:
	def __init__(self):
		self._root = None
		self._frame_time = 0
		self._frames = []

	# Loads the skeleton from file
	# Currently only BVH (Biovision Hierarchy) is supported
	# OSError is raised if <path> couldn't be opened
	# UnsupportedError if the specified file in <path> is not supported
	# FormatError if there were errors parsing the file
	def load_from_file(self, path):
		# Not really needed
		_, ext = os.path.splitext(path)
		if ext == '.bvh':
			try:
				self._root, self._frame_time, self._frames = bvh.import_bvh(path)
			except bvh.BVHFormatError as err:
				raise FormatError(str(err))
		else:
			raise UnsupportedError('Unsupported format (currently only BVH is supported)')

	# Returns the number of frames in the loaded motion data
	@property
	def frame_count(self):
		return len(self._frames)

	# Time in milliseconds between two frames
	@property
	def frame_time(self):
		return self._frame_time

	# Traverses the internal hierarchy top-down calculating the combined 
	# transforms for the specified <frame>.
	# <callback> is called for each encountered node  
	# callback(node_type : NodeType, node_name : str, current_transform : np.array, parent_transform : np.array)
	# Transforms are affine 4x4
	def traverse(self, frame, callback):
		pass