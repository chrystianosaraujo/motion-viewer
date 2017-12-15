from math_utils import quaternion_from_euler
from skeleton import AnimatedSkeleton

from PIL import Image
from graphviz import Digraph
from pyquaternion import Quaternion
import numpy as np
import glm
import math
import copy
import pickle
import random
import debugger

from functools import reduce

INVALID_DISTANCE = float("nan")
LOCAL_MINIMA_TOLERANCE = 1E-3
SAME_MOTION_SIMILARITY_TOL = (0.05, 0.3)
DIFF_MOTION_SIMILARITY_TOL = (0.0, 0.1)

def normalize_np_matrix(mat):
    min_value = np.nanmin(mat)
    max_value = np.nanmax(mat)

    elem_wise_operation = np.vectorize(lambda x: 1.0 if x != x else ((x - min_value) / (max_value - min_value)))
    return elem_wise_operation(mat)

def gradient_magnitude(gx, gy):
    out = np.empty(gx.shape, dtype=np.float32)
    for i in range(gx.shape[0]):
        for j in range(gx.shape[1]):
            out[i, j] = math.sqrt(math.pow(gx[i, j], 2) + math.pow(gy[i, j], 2))

    return out

class MotionGraph:
    class Node:
        def __init__(self, motion):
            self.label = ''
            self.motion = motion
            self.out = []
            self.iin = []
            self.out_transforms = None

        def add_edge(self, dst, motion, frames=None):
            edge = MotionGraph.Edge(self, dst, motion, frames)
            self.out.append(edge)
            dst.iin.append(edge)

        def finalize(self, compute_alignment):
            print(f'Finalizing node {self.label}')
            self.out_transforms = np.empty((len(self.iin), len(self.out)), dtype=object)

            for iini, in_edge in enumerate(self.iin):
                for outi, out_edge in enumerate(self.out):
                    last_in_positions = in_edge.motion.get_frames_positions(in_edge.frames[-1])
                    first_out_positions = out_edge.motion.get_frames_positions(out_edge.frames[0])

                    #theta, tx, tz = compute_alignment(last_in_positions, first_out_positions)
                    diff = last_in_positions[0] - first_out_positions[0]
                    tx, _, tz = diff
                    theta = 0.0 

                    transform = glm.translate(glm.mat4(), glm.vec3(tx, 0.0, tz))
                    transform = glm.rotate(transform, theta, glm.vec3(0.0, 1.0, 0.0))
                    self.out_transforms[iini, outi] = transform

    class Edge:
        def __init__(self, src, dst, motion, frames):
            self.src = src
            self.dst = dst

            self._is_transition = frames is None
            self.motion = motion
            self.frames = frames
            if self._is_transition:
                self.frames = np.arange(self.motion.frame_count)

        def split(self, frame):
            if frame < self.frames[0] or frame > self.frames[-1]:
                raise Exception('Out of range edge split')

            split = frame - self.frames[0]
            
            # No split to be done
            if split == 0:
                return self.src
            if split == (self.frames[-1] - self.frames[0]):
                return self.dst

            # Splitting adding 1 node and 2 edges
            mid_node = MotionGraph.Node(self.motion)
            edgel = MotionGraph.Edge(self.src, mid_node, self.motion, self.frames[:split])
            edger = MotionGraph.Edge(mid_node, self.dst, self.motion, self.frames[split:])
            self.src.out.append(edgel)
            mid_node.iin.append(edgel)
            mid_node.out.append(edger)
            self.dst.iin.append(edger)

            # Deregistering
            self.src.out.remove(self)
            self.dst.iin.remove(self)
            
            return mid_node

        @property
        def label(self):
            if self.motion is not None and self.frames is not None:
                return f'{self.frames[0]}..{self.frames[-1]}'
            else:
                return 'Transition'

        def is_transition(self):
            return self._is_transition

        def is_valid_frame(self, frame):
            return frame >= 0 and frame < len(self.frames)


    def __init__(self, window_length):
        self._motions = []
        self._similarity_mat = None

        if window_length is None or window_length <= 0:
            raise RuntimeError("Window length must be an integer value greater than zero.")

        self._window_length = window_length
        self._nodes = []

    def add_motion(self, motion):
        """ Adds a new motion in the motion graph.
        Since the motion graph construction is time consuming,
        the correct way to used this API is by adding all motions firstly,
        followed by calling build.

        Args:
            motion (AnimatedSkeleton): motion data
        """

        self._motions.append(motion)

    def build(self, progress_cb=None):
        """ This function creates the motion graph using all motion data
        previously added by calling the function add_motion.
        """
        self._set_progress_cb(progress_cb)

        # pr = debugger.start_profiler()

        self._build_similarity_matrix()
        self._find_local_minima()
        self._generate_graph()

        # debugger.finish_profiler(pr)
        self._set_progress_cb(None)

    def serialize(self):
        # TODO(caraujo): remove this constant string
        return pickle.dump(self, open("cache.p", "wb" ))

    def _build_similarity_matrix(self):
        num_frames = self.num_frames
        self._similarity_mat = np.empty([num_frames, num_frames], dtype=np.float32)

        # Compute the distance between each pair of frame
        for i in range(num_frames):
            self._notify_progress(i / (num_frames - 1))
            for j in range(num_frames):
                self._similarity_mat[i, j] = self._difference_between_frames(i, j)

        # Normalize each matrix quadrant individually
        curr_i = 0
        for i, motion_i in enumerate(self._motions):
            curr_j = 0
            for j, motion_j in enumerate(self._motions):
                print(j, len(motion_j))
                slice_i = slice(curr_i, curr_i + len(motion_i))
                slice_j = slice(curr_j, curr_j + len(motion_j))

                print("Normalizing Slice: ", str(slice_i), str(slice_j))

                self._similarity_mat[slice_i, slice_j] = normalize_np_matrix(self._similarity_mat[slice_i, slice_j])

                curr_j += len(motion_j)
            curr_i += len(motion_i)

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
                    if self._similarity_mat[i, j] >= TOLERANCE[0] and self._similarity_mat[i, j] <= TOLERANCE[1]:
                        all_local_minima[(i, j)] = True
                        all_minimum_pixels[i, j] = 1.0

        if len(all_local_minima) == 0:
            raise RuntimeError("There is no local minima in the constructed motion graph.")

        self._selected_local_minima = []
        regions_pixels = []
        while len(all_local_minima) > 0:
            # Find the local minima among all pixels inside the region containing seed_pixel
            seed_pixel = all_local_minima.popitem()[0]
            pixels_stack = [seed_pixel]

            new_region = []
            region_minimum_pixel = ((-1, -1), float("inf"))
            while len(pixels_stack) > 0:
                curr_pixel = pixels_stack.pop()
                new_region.append(curr_pixel)

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

            regions_pixels.append(new_region)
            self._selected_local_minima.append(region_minimum_pixel[0])

        regions_pixels_img = np.zeros(gradient_mat.shape)
        for i, region in enumerate(regions_pixels):
            for p in region:
                regions_pixels_img[p[0], p[1]] = 1.0

        only_minima_pixels_img = np.zeros(gradient_mat.shape)
        for p in self._selected_local_minima:
            only_minima_pixels_img[p[0], p[1]] = 1.0

        Image.fromarray((all_minimum_pixels * 255).astype('uint8'), "L").save("all_minimum_pixels.png")
        Image.fromarray((regions_pixels_img * 255).astype('uint8'), "L").save("regions_pixels.png")
        Image.fromarray((only_minima_pixels_img * 255).astype('uint8'), "L").save("only_minima_pixels.png")

        print("#REGIONS: " + str(len(regions_pixels)))

    @property
    def num_frames(self):
        def summation_func(x, y):
            return x + len(y)

        return reduce(summation_func, self._motions, 0)

    def get_similarity_matrix_as_image(self):
        """ Builds a greyscale image using the similarity matrix data.

        Returns:
            Returns the greyscale image (PIL.Image)

        Notes:
             This function can only be called after building the MotionGraph. Otherwise,
             a RuntimeError will be raised.
        """

        if self._similarity_mat is None:
            raise RuntimeError("MotionGraph.get_similarity_matrix_as_image function can only be called after " +
                               "calling MotionGraph.build function.")

        return self._motion_graph_image()

    def save_similarity_mat_as_txt_file(self, fn):
        """ Saves the similarity matrix in text file.

        Notes:
             This function can only be called after building the MotionGraph. Otherwise,
             a RuntimeError will be raised.
        """

        if self._similarity_mat is None:
            raise RuntimeError("MotionGraph.save_similarity_mat_as_txt_file function can only be called after " +
                               "calling MotionGraph.build function.")

        np.savetxt(fn, self._similarity_mat, fmt='%1.4f')


    def get_root_nodes(self):
        return self._nodes

    def begin_edge(self, motion_idx):
        return (self._nodes[motion_idx].out[0], glm.mat4())

    def next_edge(self, edge, children_idx, current_transform):
        dst = edge.dst

        if children_idx is None:
            children_idx = random.randrange(0, len(dst.out))

        # Can't go anywhere, restarting current motion
        if children_idx >= len(dst.out):
            return self.begin_edge(self._motions.index(dst.motion))

        in_edge_idx = dst.iin.index(edge)
        return (dst.out[children_idx], current_transform * dst.out_transforms[in_edge_idx, children_idx])

    def _difference_between_frames(self, i, j):
        window_i = self._motion_window(i, i + self._window_length - 1)
        window_j = self._motion_window(j - (self._window_length - 1), j)

        if window_i is None or window_j is None:
            return INVALID_DISTANCE

        if len(window_i) != len(window_j):
            raise RuntimeError("Distance metric can only be computed for motion windows " +
                               "with the same length.")

        # Computes the distance between the two windows
        window_i = window_i.ravel()
        window_j = window_j.ravel()
        diff_vec = window_i - window_j
        return np.linalg.norm(diff_vec)

    def _compute_alignment_between_frames(self, frame_i_positions, frame_j_positions):
        """ This function computes the transformation (Tx, Tz, Theta) that aligns two different poses.
        It basically computes an rotation around the y-axis and a translation on the x-z plane that
        aligns the pose j in respect to pose i so that the squared distance between the corresponding
        positions is minimum.
        Returns
           (theta, tx, ty)
        """

        num_parts = len(frame_i_positions)
        if len(frame_j_positions) != num_parts:
            raise RuntimeError("The alignment between two frames can only be computed when both poses "\
                               "have the same dimensions.")

        # Uniform weights
        weights = np.full(num_parts, 1.0)

        # Computing alignment angle
        # TODO: Weights must be considered in the summations below
        sum_weights = np.sum(weights)

        sum_pos_i = np.sum(frame_i_positions, axis=0)
        sum_pos_j = np.sum(frame_j_positions, axis=0)

        # Numerator   left term: SUM(w_i * (x_i * z'_i - x'_i * z_i))
        # Denominator left term: SUM(w_i * (x_i * x'_i + z_i  * z'_i))
        num_left = 0.0
        den_left = 0.0

        for w, body_pos_i, body_pos_j in zip(weights, frame_i_positions, frame_j_positions):
            num_left += w * (body_pos_i[0] * body_pos_j[2]) - (body_pos_j[0] * body_pos_i[2])
            den_left += w * (body_pos_i[0] * body_pos_j[0]) + (body_pos_i[2] * body_pos_j[2])

        # Numerator right term: 1/SUM(w_i) * (Sum(x)*Sum(z') - Sum(x')*Sum(z))
        num_right = 1.0/sum_weights * (sum_pos_i[0] * sum_pos_j[2] -
                                       sum_pos_j[0] * sum_pos_i[2])

        # Denominator right term: 1/SUM(w_i) * (Sum(x)*Sum(x') + Sum(z)*Sum(z'))
        den_right = 1.0/sum_weights * (sum_pos_i[0] * sum_pos_j[0] +
                                       sum_pos_i[2] * sum_pos_j[2])

        theta = math.atan((num_left - num_right) /
                          (den_left - den_right))

        tx = 1.0/sum_weights * (sum_pos_i[0] - sum_pos_j[0] * math.cos(theta) -
                                               sum_pos_j[2] * math.sin(theta))

        tz = 1.0/sum_weights * (sum_pos_i[2] + sum_pos_j[0] * math.sin(theta) - \
                                               sum_pos_j[2] * math.cos(theta))

        return (theta, tx, tz)

    def _generate_transition(self, window_i, window_j):
        """ This function must be used to create a transition between
        two different motion clips.

        Args:
            window_i, window_j: each window consists in a tuple with two values,
            which define the first and last interval frames.
 
        Returns:
            Returns a new window of frames that creates the transition from the
            window_i to window_j.
        """

        num_frames = window_i[1] - window_i[0] + 1

        frames_i = self._get_motion_window_as_hierarchical_poses(*window_i)
        frames_j = self._get_motion_window_as_hierarchical_poses(*window_j)

        motion_i = self._motion_data_containing_frame(window_i[0])
        motion_j = self._motion_data_containing_frame(window_j[0])
        motion_frame_i = self._motion_frame_number(motion_i, window_i[0])
        motion_frame_j = self._motion_frame_number(motion_j, window_j[0])
        frame_i_positions = motion_i.get_frames_positions(motion_frame_i)
        frame_j_positions = motion_j.get_frames_positions(motion_frame_j)

        diff = frames_i[0].position - frames_j[0].position
        tx, _, tz = diff

        dir_i = frames_i[-1].position - frames_i[0].position
        dir_j = frames_j[-1].position - frames_i[0].position
        dir_i[1] = 0.0
        dir_j[1] = 0.0
        dir_i_len = np.linalg.norm(dir_i)
        dir_j_len = np.linalg.norm(dir_j)
        angle = math.acos(np.dot(dir_i / dir_i_len, dir_j / dir_j_len))

        align_trans = np.array([tx, 0.0, tz])
        align_rot   = Quaternion(axis=[0.0, 1.0, 0.0], radians=angle)

        frames_j = copy.deepcopy(frames_j)
        blended_frames = copy.deepcopy(frames_i)

        for p, poses in enumerate(zip(frames_i, frames_j, blended_frames)):
            # -1 is used just to enforce a_p factor between[0, 1]
            a_p = 2 * math.pow((p + 1 - 1) / (num_frames - 1), 3) -\
                  3 * math.pow((p + 1 - 1) / (num_frames - 1), 2) + 1

            pose_i, pose_j, pose_b = poses

            # Align pose_j in respect to pose_i
            pose_j.position = np.add(pose_j.position, align_trans)
            pose_j.offset   = np.array([0.0, 0.0, 0.0])

            roll, pitch, yaw = pose_j.angles
            root_j_quat  = quaternion_from_euler(pitch, roll, yaw)
            aligned_root = root_j_quat * align_rot

            yaw, pitch, roll = aligned_root.yaw_pitch_roll
            #pose_j.angles = (roll, pitch, yaw)

            # Linear interpolates the root position
            pose_b.position = np.add(a_p * pose_i.position, (1 - a_p) * pose_j.position)

            # Spherical interpolates all joints angles
            stack_nodes = [(pose_i, pose_j, pose_b)]
            while stack_nodes:
                poses = stack_nodes.pop()
                pose_i, pose_j, pose_b = poses

                #roll, pitch, yaw = pose_i.angles
                #quat_i = quaternion_from_euler(pitch, roll, yaw)

                #roll, pitch, yaw = pose_j.angles
                #quat_j = quaternion_from_euler(pitch, roll, yaw)
                #blended_joint = Quaternion.slerp(quat_i, quat_j, 1 - a_p)

                #yaw, pitch, roll = blended_joint.yaw_pitch_roll
                #pose_b.angles = (roll, pitch, yaw)
                pose_b.angles = np.add(a_p * pose_i.angles, (1.0 - a_p) * pose_j.angles)
                #pose_b.angles = np.add(0 * pose_i.angles, (0.0) * pose_j.angles)

                for nodes in zip(pose_i.children, pose_j.children, pose_b.children):
                    stack_nodes.insert(0, nodes)

        transition = AnimatedSkeleton()
        transition.load_from_data(blended_frames)
        return transition

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

        # Transform interval indices to motion local index
        frame_interval = (self._to_motion_local_index(motion, begin_frame),
                          self._to_motion_local_index(motion, end_frame))

        return motion.get_frames_joint_angles(*frame_interval)

    def _get_motion_window_as_hierarchical_poses(self, begin_frame, end_frame):
        # Check if the provided indices are valid
        if begin_frame < 0 or end_frame >= self.num_frames:
            return None

        # Check if the whole window lies in the same motion clip
        motion = self._motion_data_containing_frame(begin_frame)
        if motion != self._motion_data_containing_frame(end_frame):
            return None

        if motion is None:
            raise RuntimeError("Invalid motion returned for the given frame index.")

        motion_begin_frame = self._motion_frame_number(motion, begin_frame)
        motion_end_frame = self._motion_frame_number(motion, end_frame)

        # TODO: Should we copy the motions ? 
        return [motion.get_frame_root(i) for i in range(motion_begin_frame, motion_end_frame + 1)] 

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

    def _to_motion_local_index(self, motion, frame_gid):
        for curr in self._motions:
            if curr is motion:
                return frame_gid
            frame_gid -= curr.frame_count

        raise RuntimeError("Could not found the local frame index.")

    def _motion_frame_number(self, motion, gframe):
        for cur in self._motions:
            if cur is motion:
                return gframe
            gframe -= cur.frame_count

    # Walks along the motion graph
    def _graph_find_frame(self, motion, frame):
        for node in self._nodes:
            if node.motion == motion:
                cur = node
                while True:
                    for out_edge in cur.out:
                        if out_edge.motion == motion:
                            if frame <= out_edge.frames[-1]:
                                return out_edge
                            cur = out_edge.dst
                            break

    def _graph_export_graphviz(self, filename):
        dot = Digraph(comment='Motion Graph')

        visited = {}

        # We like recursive..
        def _export_node_rec(node):
            if str(id(node)) not in visited:
                dot.node(str(id(node)), node.label)
                visited[str(id(node))] = True
                for out_edge in node.out:
                    dot.edge(str(id(out_edge.src)), str(id(out_edge.dst)), label=out_edge.label)
                    _export_node_rec(out_edge.dst)

        for node in self._nodes:
            _export_node_rec(node)
        with open(filename, 'w') as ff:
            ff.write(dot.source)
        #dot.render(filename, view=True)

    def _generate_graph(self):
        self._nodes.clear()

        # Creating one transition for each input motion
        for ii, motion in enumerate(self._motions):
            motion_beg = MotionGraph.Node(motion)
            motion_beg.label = f'Motion {ii} Begin'
            motion_end = MotionGraph.Node(motion)
            motion_end.label = f'Motion {ii} End'
            transition = MotionGraph.Edge(motion_beg, motion_end, motion, np.arange(motion.frame_count))
            motion_beg.out.append(transition)
            motion_end.iin.append(transition)
            self._nodes.append(motion_beg)

        # Insert transitions between motions for each local minima
        for minima in self._selected_local_minima:
            gsrc, gdst = minima
            src_motion = self._motion_data_containing_frame(gsrc)
            dst_motion = self._motion_data_containing_frame(gdst)

            # Getting frame idx relative to motion path
            src = self._motion_frame_number(src_motion, gsrc)
            dst = self._motion_frame_number(dst_motion, gdst)

            print(f'Inserted transition {self._motions.index(src_motion)}:{src} -> {self._motions.index(dst_motion)}:{dst}')

            if src_motion is None:
                raise RuntimeError("Invalid motion returned for the given frame index.")

            src_edge = self._graph_find_frame(src_motion, src)
            dst_edge = self._graph_find_frame(dst_motion, dst)

            if src_edge == dst_edge:
                gfirst, glast = (gsrc, gdst) if gsrc < gdst else (gdst, gsrc)
                first, last = (src, dst) if src < dst else (dst, src)

                new_node1 = src_edge.split(first)
                new_node2 = new_node1.out[0].split(last)

                window_i = (gfirst, gfirst + self._window_length - 1)
                window_j = (glast - (self._window_length - 1), glast)
                transition_motion = self._generate_transition(window_i, window_j)

                if gsrc > gdst:
                    new_node2.add_edge(new_node1, transition_motion)
                else:
                    new_node1.add_edge(new_node2, transition_motion)
            else:
                mid_src = src_edge.split(src)
                mid_dst = dst_edge.split(dst)

                window_i = (gsrc, gsrc + self._window_length - 1)
                window_j = (gdst - (self._window_length - 1), gdst)
                transition_motion = self._generate_transition(window_i, window_j)

                mid_src.add_edge(mid_dst, transition_motion)

        print(f'Inserted {len(self._selected_local_minima)} local minimas')

        # Pruning graph finding strongest connected component for each motion
        # kosaraujo 
        assert(len(self._motions) == len(self._nodes))
        all_comp_nodes = []

        for ii, motion in enumerate(self._motions):
            comp_nodes = []
            visited = {}

            def rec_add(node):
                if node in visited:
                    return

                visited[node] = True
                for out_edge in node.out:
                    if (out_edge.motion == motion or out_edge.dst.motion == motion):
                        rec_add(out_edge.dst)
                comp_nodes.append(node)

            rec_add(self._nodes[ii])
            all_comp_nodes.append(comp_nodes)

        # Tagging nodes and counting connected components
        all_tags = []
        for ii, motion in enumerate(self._motions):
            comp_nodes = all_comp_nodes[ii]
            tags = { }
            tagged_nodes = []
            cur_tag = 0
            # dbg.trace()
            for jj, comp_node in enumerate(reversed(comp_nodes)):
                traverse_stack = [comp_node]
                while traverse_stack:
                    node = traverse_stack.pop(0)
                    if node in tags:
                        continue
                    tags[node] = cur_tag
                    node.label += str(cur_tag)
                    tagged_nodes.append(node)
                    for in_edge in node.iin:
                        if (in_edge.motion == motion or in_edge.src.motion == motion) and in_edge.src not in tags:
                            traverse_stack.insert(0, in_edge.src)    
                cur_tag += 1
            all_tags.append((tags, tagged_nodes))

        self._graph_export_graphviz('components.gv')


        # Finding biggest connected component
        for ii, motion in enumerate(self._motions):
            tags = all_tags[ii]

            # Counting components
            component_nodes = {}
            for node, component in tags[0].items():
                if component not in component_nodes:
                    component_nodes[component] = []
                component_nodes[component].append(node)

            max_comp = max(component_nodes.items(), key=lambda t: len(t[1]))
            tagged_nodes = max_comp[1]

            def remove_rec(node):
                if node not in tagged_nodes:
                    for ie in node.iin:
                        ie.src.out.remove(ie)
                    for oe in node.out:
                        oe.dst.iin.remove(oe)

                for oe in node.out:
                    if oe.motion == node.motion:
                        remove_rec(oe.dst)


            remove_rec(self._nodes[ii])
            self._nodes[ii] = tagged_nodes[0]

        print('Pruned motion graph')

        visited = {}
        def finalize_rec(node):
            if str(id(node)) in visited:
                return
            visited[str(id(node))] = True
            node.finalize(self._compute_alignment_between_frames)
            for out_edge in node.out:
                finalize_rec(out_edge.dst)

        for node in self._nodes:
            finalize_rec(node)

        print('Finished generating motion graph')
        self._graph_export_graphviz('pruned.gv')

    def _set_progress_cb(self, cb):
        self._progress_cb = cb

    def _notify_progress(self, factor):
        if self._progress_cb is not None:
            self._progress_cb(factor)

def progress_cb(factor):
    print("[DEBUG] Building MotionGraph ({:.2f})%".format(factor * 100.0))

if __name__ == "__main__":
    from skeleton import *
    motion0 = AnimatedSkeleton()
    motion0.load_from_file("data\\02\\02_02.bvh")
    #motion1 = AnimatedSkeleton()
    #motion1.load_from_file("data\\02\\02_03.bvh")
    motion2 = AnimatedSkeleton()
    motion2.load_from_file("data\\16\\16_57.bvh")

    graph = MotionGraph(30)
    graph.add_motion(motion0)
    #graph.add_motion(motion1)
    graph.add_motion(motion2)

    import time
    start = time.time()
    graph.build(progress_cb)
    print('It took {0:0.1f} seconds'.format(time.time() - start))
    graph.get_similarity_matrix_as_image().show()
