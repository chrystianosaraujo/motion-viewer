import glm

class MotionPlayer:
	def __init__(self, motion=None):
		self._reset()
		self._motion = motion

	@property
	def motion(self):
		return self._motion

	@motion.setter
	def motion(self, motion):
		self._reset()
		if motion is None:
			return

		self._motion = motion

	def current_motion_data(self):
		return (self._motion, self._current_frame, self._transform)

	def update(self, dt):
		if self._motion is None:
			return

		self._current_frame	= (self._current_frame + 1) % self._motion.frame_count		

	def _reset(self):
		self._motioon = None
		self._current_frame = 0
		self._transform = glm.mat4(1.0)
		self._dt = 0