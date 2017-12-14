# Python
import enum
import os
import math
import copy

# Motion Viewer
import bvh
import debugger as dbg

# External
import glm
import numpy as np


class NodeType(enum.Enum):
    UNKNOWN = -1 # Does -1 make sense in python as negative indices are valid ?
    HEAD = 0
    EYE = 1
    NECK = 2
    TORSO = 3
    UPPER_LEG = 4
    LOWER_LEG = 5
    FOOT = 6
    UPPER_ARM = 7
    LOWER_ARM = 8
    FINGER = 9
    HAND = 10


class UnsupportedError(Exception):
    pass


class FormatError(Exception):
    pass

# Hierarchical collection of joints with associated motion data
class AnimatedSkeleton: 
    class Node:
        # All the private variables (_*) are not needed for creation, but are cached and can be computed from the public ones 
        def __init__(self, name, ntype, offset=np.array([0.0, 0.0, 0.0]), position=np.array([0.0, 0.0, 0.0]), angles=np.array([0.0, 0.0, 0.0])):
            self.name = name
            self.ntype = ntype
            self.offset = offset
            self.position = position
            self.angles = angles # 3 rotations 
            self.children = []
            self.ee_offset = None # In case the Node is an end effector this represents its ending point
            self.rotation_order = None # 3 dimensional array containing order for X Y Z rotation
            self._length = None # 3D length of the joint
            self._rest_rot = None # Rotation at angles 0 with respect to (0, 1, 0). This is needed to correctly render joints
            self._transform = None # Cached transform

        def rec_print(self, indent=0):
            buf = indent * '\t' + f'Name: {self.name} Pos: {self.position} Angles: {self.angles}\n'
            for child in self.children:
                buf += child.rec_print(indent+1)
            return buf

    # Mapping type -> names
    # For readability
    BVH_JOINT_NAMES = {
        NodeType.HEAD: ['head'],
        NodeType.NECK: ['neck'],
        NodeType.EYE: ['eye'],
        NodeType.TORSO: ['hip', 'chest', 'abdomen', 'spine', 'back', 'collar', 'buttock'],
        NodeType.UPPER_LEG: ['upleg', 'thigh'],
        NodeType.LOWER_LEG: ['leg', 'shin'],
        NodeType.FOOT: ['foot', 'toe'],
        NodeType.UPPER_ARM: ['shoulder', 'shldr'],
        NodeType.LOWER_ARM: ['arm'],
        NodeType.FINGER: ['finger', 'thumb', 'index', 'mid', 'ring', 'pinky'],
        NodeType.HAND: ['hand']
    }

    def __init__(self, up=(0.0, 1.0, 0.0)):
        self._frames = [] # [Node] 
        self._frame_time = 0 # ms between each frame
        self._reference_up = up # Reference up vector

        self._identity = glm.mat4()
        self._all_joint_angles =  None # Cached Joints Angles
        self._all_positions =  None # Cached body's parts position
        self._pose_joints_dims = None
        self._pose_positions_dims = None

    def __len__(self):
        return len(self._frames)

    def load_from_file(self, path):
        """ 
        Loads skeleton info and data from file. 
        Currently only BVH (Biovision Hierarchy) is supported.
        OSError is raised if <path> doesn't point to an openable file
        UnsupportedError is raised if the file located at <path> is not of a supported format.
        FormatError is raised if errors occurred while parsing the file
        """
        # Not really needed right not
        _, ext = os.path.splitext(path)
        if ext == '.bvh':
            try:
                print(f'Loading BVH from: {path}')
                bvh_root, bvh_frame_time, bvh_frames = bvh.import_bvh(path)
                
                def process_bvh_rec(frame, node):
                    rotx = bvh_frames[frame][node.rotx_idx] if node.rotx_idx is not None else None
                    roty = bvh_frames[frame][node.roty_idx] if node.roty_idx is not None else None
                    rotz = bvh_frames[frame][node.rotz_idx] if node.rotz_idx is not None else None

                    posx = bvh_frames[frame][node.offx_idx] if node.offx_idx is not None else 0
                    posy = bvh_frames[frame][node.offy_idx] if node.offy_idx is not None else 0
                    posz = bvh_frames[frame][node.offz_idx] if node.offz_idx is not None else 0

                    # Creating internal node representation
                    skeleton_node = AnimatedSkeleton.Node(node.name, self._find_type_bvh(node.name))
                    skeleton_node.offset = np.array(node.offset)
                    skeleton_node.position = np.array([posx, posy, posz])
                    skeleton_node.angles = np.array([rotx, roty, rotz])
                    skeleton_node._length = node.estimated_length
                    skeleton_node._transform = self._compute_transform(node.offset, (rotx, roty, rotz), (posx, posy, posz), node.rotation_order)
                    skeleton_node._rest_rot = self._compute_rest_rotation(node.estimated_length)
                    skeleton_node.rotation_order = node.rotation_order

                    if node.is_ee:
                        skeleton_node.ee_offset = np.array(node.ee_offset)

                    for child in node.children:
                        skeleton_node.children.append(process_bvh_rec(frame, child))

                    return skeleton_node

                num_frames = len(bvh_frames)
                print(f'Processing {num_frames} nodes')
                for frame in range(num_frames):
                    self._frames.append(process_bvh_rec(frame, bvh_root[0]))

                self._frame_time = bvh_frame_time
                self._populate_caches()

                print(f'Finished: {path}')

            except bvh.BVHFormatError as err:
                raise FormatError(str(err))
        else:
            raise UnsupportedError('Unsupported format (currently only BVH is supported)')    


    def load_from_data(self, frames):
        """ Creates an animated skeleton directly from a list of Nodes (one per frame) and
            computes transform and rest rotation.
        """
        if not frames:
            # error()
            return

        print(f'Processing {len(frames)} frame')

        # Same as internal format
        self._frames = frames

        def _process_data_rec(node):
            node._transform = self._compute_transform(node.offset, node.angles, node.position, node.rotation_order)

            if len(node.children) > 1:
                node._length = (0.0, 0.0, 0.0)
            elif len(node.children) == 1:
                node._length = node.children[0].offset
            elif node.is_ee:
                node._length = node.ee_offset

            node._rest_rot = self._compute_rest_rotation(node._length)

        for frame in self._frames:
            _process_data_rec(frame)

        self._populate_caches()

    def traverse(self, frame, callback, root_transform=glm.mat4(1.0)):
        """
            DFS Skeleton traversal which calls callback() on all renderable nodes.
            callback(type, name, transform, length, rest_rotation)
            TODO: This should probably be called traverse_visible() or traverse_renderable()
        """

        if frame not in range(0, self.frame_count):
            # error()
            return

        if not self._frames:
            # error()
            return

        self._traverse(self._frames[frame], root_transform, callback)

    def get_frames_positions(self, first_frame, last_frame=None):
        """ Retrieves the list of global positions (not relative to parent) after transformation
            for the specified frame interval.

        Args:
          first_frame: first frame of the interval
          last_frame : last frame (included) of the interval. In case it is None, the first
                       frame will be considered.
        """

        last_frame = last_frame if last_frame is not None else first_frame


        num_frames = last_frame - first_frame + 1
        interval = (first_frame * self._pose_positions_dims,
                    first_frame * self._pose_positions_dims + (num_frames * self._pose_positions_dims))

        all_positions = self._all_positions[interval[0] : interval[1]]
        return np.array(all_positions)

    def get_frames_joint_angles(self, first_frame, last_frame=None):
        """ Retrieves the list of relative joint rotations for the specified frame interval.

        Args:
          first_frame: first frame of the interval
          last_frame : last frame (included) of the interval. In case it is None, the first
                       frame will be considered.
        """

        last_frame = last_frame if last_frame is not None else first_frame

        num_frames = last_frame - first_frame + 1
        interval = (first_frame * self._pose_joints_dims,
                    first_frame * self._pose_joints_dims + (num_frames * self._pose_joints_dims))

        all_angles = self._all_joint_angles[interval[0] : interval[1]]
        return np.array(all_angles)

    def get_frame_root(self, frame):
        return self._frames[frame]

    @property
    def frame_count(self):
        return len(self._frames)

    # Time in milliseconds between two frames
    @property
    def frame_time(self):
        return self._frame_time

    def _traverse(self, root, root_transform, callback):
        traverse_stack = [root]
        transform_stack = [root_transform]

        while traverse_stack:
            node = traverse_stack.pop(0)
            parent_transform = transform_stack.pop(0)

            transform = parent_transform * self._compute_transform(node.offset, node.angles, node.position, node.rotation_order)

            if len(node.children) <= 1 and callback:
                callback(node.ntype, node.name, transform, glm.length(node._length), node._rest_rot)

            for child in node.children:
                traverse_stack.insert(0, child)
                transform_stack.insert(0, transform)

    def _find_type_bvh(self, name):
        for t, names in AnimatedSkeleton.BVH_JOINT_NAMES.items():
            for cname in names:
                if cname in name.lower():
                    return t
        print(f'Failed to find type for {name}')
        return None

    def _compute_transform(self, offset, rotation, position, rotation_order):
        offx, offy, offz = offset
        rotx, roty, rotz = rotation
        posx, posy, posz = position

        Rx = glm.mat4(1.0)
        if rotx:
            c = math.cos(math.radians(rotx))
            s = math.sin(math.radians(rotx))

            Rx = glm.mat4([1.0, 0.0, 0.0, 0.0],
                          [0.0, c, s, 0.0],
                          [0.0, -s, c, 0.0],
                          [0.0, 0.0, 0.0, 1.0])

        Ry = glm.mat4(1.0)
        if roty:
            c = math.cos(math.radians(roty))
            s = math.sin(math.radians(roty))

            Ry = glm.mat4([c, 0.0, -s, 0.0],
                          [0.0, 1.0, 0.0, 0.0],
                          [s, 0.0, c, 0.0],
                          [0.0, 0.0, 0.0, 1.0])

        Rz = glm.mat4(1.0)
        if rotz:
            c = math.cos(math.radians(rotz))
            s = math.sin(math.radians(rotz))

            Rz = glm.mat4([c, s, 0.0, 0.0],
                          [-s, c, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0])


        # The multiplication order depends on how they are specified in the file
        Rall = (Rx, Ry, Rz)
        R = Rall[rotation_order[0]] * Rall[rotation_order[1]] * Rall[rotation_order[2]]
        T = glm.mat4([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [offx + posx, offy + posy, offz + posz, 1.0]])

        # Composing transforms
        return T * R

    # Calculating rotation needed for the up vector (0, 1, 0) to be orientated correctly
    # along the joint's direction (estimated_length) at rest position.
    def _compute_rest_rotation(self, direction):
        def compute_rotation(a, v):
            c = math.cos(a)
            s = math.sin(a)
            axis = glm.normalize(v)
            tmp = glm.vec4((1.0 - c) * axis)           
            R = glm.mat4(c + ((1) - c)      * axis[0]     * axis[0],
            ((1) - c) * axis[0] * axis[1] + s * axis[2],
            ((1) - c) * axis[0] * axis[2] - s * axis[1],
            (0),
            ((1) - c) * axis[1] * axis[0] - s * axis[2],
            c + ((1) - c) * axis[1] * axis[1],
            ((1) - c) * axis[1] * axis[2] + s * axis[0],
            (0),
            ((1) - c) * axis[2] * axis[0] + s * axis[1],
            ((1) - c) * axis[2] * axis[1] - s * axis[0],
            c + ((1) - c) * axis[2] * axis[2],
            (0),
            0, 0, 0, 1)
            return R


        initial_dir_unnorm = glm.vec3(direction)
        initial_dir = glm.normalize(initial_dir_unnorm)
        ortho = glm.cross(self._reference_up, initial_dir)
        angle = math.acos(glm.dot(self._reference_up, initial_dir))
        if math.isnan(angle):
            return glm.mat4(1.0)
        return compute_rotation(angle, ortho) #glm.rotate(glm.mat4(), angle, ortho)

    def _populate_caches(self):
        self._populate_joint_angles_cache()
        self._populate_positions_cache()

    def _populate_joint_angles_cache(self):
        self._all_joint_angles = []

        def gather_rotations_rec(node):
            angles = node.angles if node.angles is not None \
                                 else np.zeros(3)

            self._all_joint_angles.append(angles)

            for child in node.children:
                gather_rotations_rec(child)

        for frame in self._frames:
            gather_rotations_rec(frame)

        # Initializes pose angles dimension
        self._pose_joints_dims = len(self._all_joint_angles) // \
                                 len(self._frames)

    def _populate_positions_cache(self):
        self._all_positions = []

        def gather_positions_rec(node, parent_transform):
            transform = parent_transform * node._transform
            trans = transform[3, :3]
            self._all_positions.append(np.asarray(trans).ravel())

            for child in node.children:
                # TODO(edoardo): Is this copy really needed?
                gather_positions_rec(child, copy.copy(transform))

        for frame in self._frames:
            gather_positions_rec(frame, glm.mat4(1.0))

        # Initializes pose angles dimension
        self._pose_positions_dims = len(self._all_positions) // \
                                    len(self._frames)
