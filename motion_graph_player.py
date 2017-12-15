import glm

from motion_graph import MotionGraph


class MotionGraphPlayer:
	def __init__(self, motion_graph=None):
		self._reset()
		self.motion_graph = motion_graph

	@property
	def motion_graph(self):
		return self._motion_graph

	@motion_graph.setter
	def motion_graph(self, motion_graph):
		self._reset()
		if motion_graph is None:
			return

		self._motion_graph = motion_graph
		self._current_edge, self._transform = self._motion_graph.begin_edge(0)

	def current_motion_data(self):
		return (self._current_edge, self._current_edge.frames[self._current_frame], self._transform)

	def update(self, dt):
		if self._motion_graph is None:
			return

		self._current_frame += 1

		if not self._current_edge.is_valid_frame(self._current_frame):
			if not self._path:
				self._path.append(0)

			next_out_edge = self._path.pop(0)
			self._current_edge, self._transform = self._motion_graph.next_edge(self._current_edge, next_out_edge, self._transform)
			self._current_frame = 0


	def _reset(self):
		self._motion_graph = None
		self._current_edge = None
		self._current_frame = 0
		self._dt = 0
		self._transform = glm.mat4(1.0)
		self._path = []

	def _frame_interval(self):
		if self._current_edge is not None:
			return self._current_edge.motion.frame_time