from functools import reduce
from PIL import Image
import multiprocessing
import threading
import logging
import scipy
from scipy import ndimage

import numpy as np

LOCAL_MINIMA_TOLERANCE = 1E-2
SAME_MOTION_SIMILARITY_TOL = (0.1, 0.3)
DIFF_MOTION_SIMILARITY_TOL = (0.0, 0.3)

def normalize_np_matrix(mat):
  min_value = np.nanmin(mat)
  max_value = np.nanmax(mat)

  elem_wise_operation = np.vectorize(lambda x : 1.0 if x != x else ((x - min_value) / (max_value - min_value)))
  return elem_wise_operation(mat)

def from_matrix_to_image(mat, normalize = False, border = (0, 0)):
    if normalize:
        mat = normalize_np_matrix(mat, border)

    #im = Image.new("RGB", mat.shape)
    #data = [(int(mat[x, y] * 255), int(mat[x, y] * 255), int(mat[x, y] * 255)) for y in range(im.size[1]) for x in range(im.size[0])]
    #im.putdata(data)

    #return im
    return Image.fromarray((mat * 255).astype('uint8'), "L")

def gradient_magnitude(gx, gy):
    out = np.empty(gx.shape, dtype=np.float32)
    for i in range(gx.shape[0]):
      for j in range(gx.shape[1]):
          out[i, j] = math.sqrt(math.pow(gx[i, j], 2) + math.pow(gy[i, j], 2))

    return out

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

        self._build_similarity_matrix()
        self._find_local_minima()

        pr.disable()
        s = io.StringIO()
        sortby = 'time'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

    def _build_similarity_matrix(self):
        self._build_list_of_poses()

        num_frames = self.num_frames
        self._similarity_mat = np.empty([num_frames, num_frames], dtype=np.float32)

        # Compute the distance between each pair of frame
        for i in range(num_frames):
            print("[DEBUG] (Build MotionGraph {}/{} - ({:.3f})%".format(i, num_frames - 1, i/(num_frames-1) * 100))
            for j in range(num_frames):
                self._similarity_mat[i, j] = self._difference_between_frames(i, j)

        shape = (0, len(self._motions[0]), 0, len(self._motions[0]))
        self._similarity_mat[shape[0]:shape[1], shape[2]:shape[3]] = normalize_np_matrix(self._similarity_mat[shape[0]:shape[1], shape[2]:shape[3]])

        shape = (len(self._motions[0]), self._similarity_mat.shape[0], 0, len(self._motions[0]))
        self._similarity_mat[shape[0]:shape[1], shape[2]:shape[3]] = normalize_np_matrix(self._similarity_mat[shape[0]:shape[1], shape[2]:shape[3]])

        shape = (0, len(self._motions[0]), len(self._motions[0]), self._similarity_mat.shape[1])
        self._similarity_mat[shape[0]:shape[1], shape[2]:shape[3]] = normalize_np_matrix(self._similarity_mat[shape[0]:shape[1], shape[2]:shape[3]])

        shape = (len(self._motions[0]), self._similarity_mat.shape[0], len(self._motions[0]), self._similarity_mat.shape[1])
        self._similarity_mat[shape[0]:shape[1], shape[2]:shape[3]] = normalize_np_matrix(self._similarity_mat[shape[0]:shape[1], shape[2]:shape[3]])

        #self._similarity_mat = normalize_np_matrix(self._similarity_mat)

    def _find_local_minima(self):
        # Compute the gradient of the similarity matrix
        gx, gy = np.gradient(self._similarity_mat)
        gradient_mat = gradient_magnitude(gx, gy)

        all_local_minima = {}
        all_minimum_pixels = np.zeros(gradient_mat.shape)
        for i in range(gradient_mat.shape[0]):
            for j in range(gradient_mat.shape[1]):
                TOLERANCE = DIFF_MOTION_SIMILARITY_TOL

                if self._motion_data_containing_frame(i) ==\
                   self._motion_data_containing_frame(j):
                  TOLERANCE = SAME_MOTION_SIMILARITY_TOL

                if gradient_mat[i, j] < LOCAL_MINIMA_TOLERANCE:
                      if self._similarity_mat[i, j] >= TOLERANCE[0] and self._similarity_mat[i,j] <= TOLERANCE[1]:
                          all_local_minima[(i, j)] = True
                          all_minimum_pixels[i, j] = 1.0


        #import ipdb; ipdb.set_trace()
        if len(all_local_minima) == 0:
            raise RuntimeError("There is no local minima in the constructed motion graph.")

        self._selected_local_minima = []
        regions_pixels = []
        while len(all_local_minima) > 0:
            # Find the local minima among all pixels inside the region containing seed_pixel
            seed_pixel = all_local_minima.popitem()[0]
            print("Starting  Region From: " + str(seed_pixel))
            pixels_stack = [seed_pixel]

            new_region = []
            region_minimum_pixel = ((-1, -1), float("inf"))
            while len(pixels_stack) > 0:
                curr_pixel = pixels_stack.pop()
                new_region.append(curr_pixel)
                print("....Processing: " + str(curr_pixel))

                pixel_value = self._similarity_mat[i, j]

                if pixel_value < region_minimum_pixel[1]:
                    region_minimum_pixel = (curr_pixel, pixel_value)

                # Adding all neighbor local minima
                offsets = ((-1, -1), (0, -1), (+1, -1),
                           (-1,  0),          (+1,  0),
                           (-1,  1), (0,  1), (+1,  1))

                for offset in offsets:
                    neighbor_pixel = (curr_pixel[0] + offset[0],
                                      curr_pixel[1] + offset[1])

                    # Ignore neighbors outside the valid region
                    if neighbor_pixel[0] < 0 or \
                       neighbor_pixel[1] < 0 or \
                       neighbor_pixel[0] >= self._similarity_mat.shape[0] or \
                       neighbor_pixel[1] >= self._similarity_mat.shape[1]:
                        continue

                    # Add pixels that lie in a local minima
                    if all_local_minima.get(neighbor_pixel) is not None:
                        pixels_stack.insert(0, neighbor_pixel)
                        del all_local_minima[neighbor_pixel]

            #import ipdb;ipdb.set_trace()
            regions_pixels.append(new_region)
            self._selected_local_minima.append(region_minimum_pixel[0])

        regions_pixels_img = np.zeros(gradient_mat.shape)
        import ipdb;ipdb.set_trace()
        for i, region in enumerate(regions_pixels):
            print(i)
            for p in region:
                regions_pixels_img[p[0], p[1]] = 1.0

        only_minima_pixels_img = np.zeros(gradient_mat.shape)
        for p in self._selected_local_minima:
            only_minima_pixels_img[p[0], p[1]] = 1.0

        Image.fromarray((all_minimum_pixels * 255).astype('uint8'), "L").save("f:\\all_minimum_pixels.png")
        Image.fromarray((regions_pixels_img * 255).astype('uint8'), "L").save("f:\\regions_pixels.png")
        Image.fromarray((only_minima_pixels_img * 255).astype('uint8'), "L").save("f:\\only_minima_pixels.png")

        print("#REGIONS: " + str(len(regions_pixels)))

    @property
    def num_frames(self):
        #import ipdb; ipdb.set_trace()
        #summation_func = lambda x, y: len(x) + len(y)
        #return reduce(summation_func, self._motions, [])
        count = 0
        for motion in self._motions:
            count += len(motion)
        return count

    def get_similarity_matrix_as_image(self):
        """ Builds a greyscale image using the similarity matrix data.

        Returns:
            Returns the greyscale image (PIL.Image)

        Notes:
             This function can only be called after building the MotionGraph. Otherwise,
             a RuntimeError will be raised.
        """

        if self._similarity_mat is None:
            raise RuntimeError("MotionGraph.get_similarity_matrix_as_image function can only be called after " +\
                               "calling MotionGraph.build function.")

        return self._motion_graph_image()

    def save_similarity_mat_as_txt_file(self, fn):
        """ Saves the similarity matrix in text file.

        Notes:
             This function can only be called after building the MotionGraph. Otherwise,
             a RuntimeError will be raised.
        """

        if self._similarity_mat is None:
            raise RuntimeError("MotionGraph.save_similarity_mat_as_txt_file function can only be called after " +\
                               "calling MotionGraph.build function.")

        np.savetxt(fn, self._similarity_mat, fmt='%1.4f')

    def _difference_between_frames(self, i, j):
        ## Clamp window size to shortest possible. It is important to handle poses at the beginning
        ## and at the end of the animation
        #win_length = self._window_length
        #if (i + win_length) > (self.num_frames - 1) or (j - win_length) < 0:
        #       win_length = min(self.num_frames - i, j + 1)

        window_i = self._motion_window(i, i + self._window_length - 1)
        window_j = self._motion_window(j - (self._window_length - 1), j)
        #window_i = self._motion_window(i, i + win_length - 1)
        #window_j = self._motion_window(j - (win_length- 1), j)

        if window_i is None or window_j is None:
            return MotionGraph.INVALID_DISTANCE

        if len(window_i) != len(window_j):
            raise RuntimeError("Distance metric can only be computed for motion windows " +\
                               "with the same length.")

        # Computes the distance between the two windows
        diff_vec = window_i - window_j
        return np.linalg.norm(diff_vec)

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

        return self._frames_pose[begin_frame : end_frame + 1, ]

    def _motion_data_containing_frame(self, frame_idx):
        """ Returns the motion data related to the given global frame index."""

        local_idx = frame_idx

        for motion in self._motions:
            if local_idx < len(motion):
                return motion
            else:
                local_idx -= len(motion)

        raise RuntimeError("Could not find the motion graph related to the given frame index ({}).".format(frame_idx))

    def _motion_graph_image(self):
        pillow_compatible = (self._similarity_mat * 255).astype('uint8')
        return Image.fromarray(pillow_compatible, "L")

    def _build_list_of_poses(self):
        first_motion = self._motion_data_containing_frame(0)
        pose_dimensions = len(first_motion.get_all_joint_rotations(0)[0])

        self._frames_pose = np.empty((self.num_frames, pose_dimensions), dtype=np.float32)

        offset = 0
        for motion in self._motions:
            for i in range(len(motion)):
                self._frames_pose[offset + i:] = motion.get_all_joint_rotations(i)[0]

            offset += len(motion)

    def _generate_graph(self):
        # self._local_minima = np.array((NUM_MINIMA, 2))
        pass

if __name__ == "__main__":
    from skeleton import *
    logging.basicConfig(level=logging.DEBUG)
    motion0 = AnimatedSkeleton()
    motion0.load_from_file("F:\\caraujo\\PhD\\Courses\\Computer "
            "Animation\\FinalProject\\motion-viewer\\data\\02\\02_02.bvh")
    motion1 = AnimatedSkeleton()
    motion1.load_from_file("F:\\caraujo\\PhD\\Courses\\Computer "
            "Animation\\FinalProject\\motion-viewer\\data\\02\\02_03.bvh")

    graph = MotionGraph(30)
    graph.add_motion(motion0)
    graph.add_motion(motion1)

    import time
    start = time.time()
    graph.build()
    print('It took {0:0.1f} seconds'.format(time.time() - start))
    graph.get_similarity_matrix_as_image().show()
