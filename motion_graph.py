from functools import reduce
from PIL import Image
import multiprocessing
import threading
import logging

import numpy as np

class MotionGraph:
    INVALID_DISTANCE = float("nan")

    def __init__(self, window_length):
        self._motions = []
        self._similarity_mat = None

        if window_length is None or window_length <= 0:
            raise RuntimeError("Window length must be an integer value greater than zero.")

        self._window_length = window_length

    def add_motion(self, motion):
        """ Adds a new motion in the motion graph.
        Since the motion graph construction is time consuming,
        the correct way to used this API is by adding all motions firstly,
        followed by calling build.

        Args:
            motion (AnimatedSkeleton): motion data
        """

        self._motions.append(motion)

    def build(self):
        """ This function creates the motion graph using all motion data
        previously added by calling the function add_motion.

        Notes:
            TODO: Handle more than one motion:
        """

        import cProfile, pstats, io
        pr = cProfile.Profile()
        pr.enable()

        self._build_list_of_poses()

        num_frames = self.num_frames
        self._similarity_mat = np.empty([num_frames, num_frames], dtype=np.float32)

        def parallel_worker(thread_i, first_row, last_row):
            logging.debug("[DEBUG] (Thread {}) Starting - Rows[{}, {}]".format(thread_i, first_row, last_row))

            for i in range(first_row, last_row + 1):
                logging.debug("[DEBUG] (Thread {}) {}/{} - ({:.3f})%".format(thread_i, i,range(first_row, last_row + 1)[-1], i / range(first_row, last_row + 1)[-1]))
                for j in range(num_frames):
                    #self._similarity_mat[i, j] = self._difference_between_frames(i, j)
                    x = self._difference_between_frames(i, j)

            logging.debug("[DEBUG] (Thread {}) Exiting".format(thread_i))

        # Compute the distance between each pair of frame
        num_threads = 3#multiprocessing.cpu_count() // 2
        rows_per_thread = num_frames // num_threads

        #threads = []
        #for i in range(num_threads):
        #    first_row = i * rows_per_thread
        #    last_row  = first_row + rows_per_thread - 1

        #    if i == num_threads - 1: 
        #      last_row  = num_frames - 1

        #    t = threading.Thread(target=parallel_worker, args=(i, first_row, last_row))
        #    t.start()
        #    threads.append(t)

        ## Waiting all threads to finish
        #for thread in threads:
        #    thread.join()

        #logging.debug("[DEBUG] All threads have finished!")
        for i in range(num_frames):
            print("{}/{} - ({:.3f})%".format(i, num_frames, i / num_frames * 100))
            for j in range(num_frames):
                self._similarity_mat[i, j] = self._difference_between_frames(i, j)

        #np.savetxt('d:\\test.out', self._similarity_mat, fmt='%1.4f')

        # Normalizes the distance values and convertes from distance to 
        # similarity (basically similatiry = 1 - distance)
        min_value = np.nanmin(self._similarity_mat)
        max_value = np.nanmax(self._similarity_mat)

        elem_wise_operation = np.vectorize(lambda x : 1.0 if x != x else ((x - min_value) / (max_value - min_value)))
        self._similarity_mat = elem_wise_operation(self._similarity_mat)

        #np.savetxt('d:\\test_normalized.out', self._similarity_mat, fmt='%1.4f')
        pr.disable()
        s = io.StringIO()
        sortby = 'time'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())


    @property
    def num_frames(self):
        summation_func = lambda x, y: len(x) + len(y)
        return reduce(summation_func, self._motions, [])

    def save_as_image(self, fn):
        """ Saves the similarity matrix as a grayscale image.

        Args:
            fn (string): output image filename.

        Notes:
             This function can only be called after building the MotionGraph. Otherwise,
             a RuntimeError will be raised.
        """
        self._motion_graph_image().save(fn)

    def show_as_image(self):
        """ Shows the similarity matrix as a grayscale image.

        Notes:
             This function can only be called after building the MotionGraph. Otherwise,
             a RuntimeError will be raised.
        """
        self._motion_graph_image().show()

    def _difference_between_frames(self, i, j):
        # The distance between a frame and itself is not considered.
        if i == j:
            return MotionGraph.INVALID_DISTANCE

        window_i = self._motion_window(i, i + self._window_length - 1)
        window_j = self._motion_window(j - (self._window_length - 1), j)
  
        if window_i is None or window_j is None:
            return MotionGraph.INVALID_DISTANCE
  
        if len(window_i) != len(window_j):
            raise RuntimeError("Distance metric can only be computed for motion windows " +\
                               "with the same length.")

        # Computes the squared distance between the two windows
        accum_distance = 0.0
        for x in range(len(window_i)):
            diff_vec = np.subtract(window_i[x], window_j[x])
            accum_distance += np.inner(diff_vec, diff_vec)

        #diff_vec = window_i - window_j
        #return np.linalg.norm(diff_vec)

        return accum_distance

    def _motion_window(self, begin_frame, end_frame):
        """ Returns the interval of motion frames given the first and last
        interface indices.

        Args:
            begin_frame (int): index first window frame
            end_frame   (int): index last window frame

        Note:
            All provided indices must be defined considering that all added motions' frames are
            stored as an unique list. In orther words, the first frame of the second will be numbered
            as the last frame of the first motion + 1.

            None will be returned if the given window indices are not valid. To be
            considered as valid, the begin and last window frames must belong to the same
            motion path.
        """

        # Check if the provided indices are valid
        if begin_frame < 0 or end_frame >= self.num_frames:
            return None

        # Check if the whole window lies in the same motion clip
        motion = self._motion_data_containing_frame(begin_frame)
        if motion != self._motion_data_containing_frame(end_frame):
            return None

        if motion is None:
            raise RuntimeError("Invalid motion returned for the given frame index.")

        # Ignoring joint types so far
        #test = [motion.get_all_joint_rotations(i)[0] for i in range(begin_frame, end_frame + 1)]
        #test2 = self._frames_pose[begin_frame : end_frame + 1, ]
        return self._frames_pose[begin_frame : end_frame + 1, ]

    def _motion_data_containing_frame(self, frame_idx):
        """ Returns the motion data related to the given global frame index.
        """
        local_idx = frame_idx

        for motion in self._motions:
            if local_idx < len(motion):
                return motion
            else:
                local_idx -= len(motion)

        return None

    def _motion_graph_image(self):
        pillow_compatible = (self._similarity_mat * 255).astype('uint8')
        return Image.fromarray(pillow_compatible, "L")

    def _build_list_of_poses(self):
        first_motion = self._motion_data_containing_frame(0)
        pose_dimensions = len(first_motion.get_all_joint_rotations(0)[0])

        self._frames_pose = np.empty((self.num_frames, pose_dimensions), dtype=np.float32)

        for i in range(self.num_frames):
            self._frames_pose[i:] = motion.get_all_joint_rotations(i)[0]

if __name__ == "__main__":
    from skeleton import *
    logging.basicConfig(level=logging.DEBUG)
    motion = AnimatedSkeleton()
    motion.load_from_file("F:\\caraujo\\PhD\\Courses\\Computer Animation\\FinalProject\\motion-viewer\\data\\06\\06_10.bvh")
    #motion.load_from_file("F:\\caraujo\\PhD\\Courses\\Computer Animation\\FinalProject\\motion-viewer\\data\\02\\02_02.bvh")

    graph = MotionGraph(30)
    graph.add_motion(motion)
    import time
    start = time.time()
    graph.build()
    print('It took {0:0.1f} seconds'.format(time.time() - start))
    graph.show_as_image()
