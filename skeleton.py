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
    # Mapping type -> names
    # For readability
    BVH_JOINT_NAMES = {
        NodeType.HEAD: ['head'],
        NodeType.NECK: ['neck'],
        NodeType.EYE: ['eye'],
        NodeType.TORSO: ['hip', 'chest', 'abdomen', 'spine', 'back', 'collar'],
        NodeType.UPPER_LEG: ['upleg', 'thigh'],
        NodeType.LOWER_LEG: ['leg', 'shin'],
        NodeType.FOOT: ['foot', 'toe'],
        NodeType.UPPER_ARM: ['shoulder', 'shldr'],
        NodeType.LOWER_ARM: ['arm'],
        NodeType.FINGER: ['finger', 'thumb', 'index', 'mid', 'ring', 'pinky']
    }

    def __init__(self):
        self._root = None
        self._frame_time = 0
        self._frames = []

        self._traverse_func = None
        self._traverse_func_nop = None
        self._motion_cache = {}

        self._identity = glm.mat4()

        self._bvh_types = {}
        for t, names in AnimatedSkeleton.BVH_JOINT_NAMES.items():
            for name in names:
                self._bvh_types[name] = t

    def __len__(self):
        return len(self._frames)

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
                self._traverse_func = self._traverse_bvh
                self._traverse_func_nop = self._traverse_bvh_nop
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
    # callback(node_type : NodeType, node_name : str, transform)
    # Transforms are affine 4x4
    def traverse(self, frame, callback):
        if frame not in range(0, self.frame_count):
            # error()
            return

        if not self._root:
            # error()
            return

        transform = glm.mat4(1.0)
        self._traverse_func(frame, callback, self._root[0], transform)

    # Returns a tuple containg all the joint rotations with associated
    # node type for the specified frame. Multiple calls return the angles in
    # the same order
    def get_all_joint_rotations(self, frame):  # -> ([rotations], [types])
        rotations = []
        types = []

        def gather_rotations(type, name, rotx, roty, rotz):
            if rotx is not None:
                rotations.append(rotx)
                types.append(type)
            if roty is not None:
                rotations.append(roty)
                types.append(type)
            if rotz is not None:
                rotations.append(rotz)
                types.append(type)

        self._traverse_func_nop(self._root[0], frame, gather_rotations)
        return (rotations, types)

    def get_all_joints_position(self, frame):  # -> ([positions], [types])
        positions = []
        ntypes = []

        def gather_positions(ntype, name, transform, dim, rest_rot):
            trans = transform[3, :3]
            positions.append(np.asarray(trans).ravel())
            ntypes.append(ntype)

        self.traverse(frame, gather_positions)
        return (np.array(positions), ntypes)

    # TODO: This should be cached
    def _find_type_bvh(self, name):
        for t, names in AnimatedSkeleton.BVH_JOINT_NAMES.items():
            for cname in names:
                if cname in name.lower():
                    return t                    
        print(f'Failed to find type for {name}')
        return None

    def _traverse_bvh(self, frame, callback, node, base_transform):
        traverse_stack = [node]
        transform_stack = [base_transform]

        while traverse_stack:
            node = traverse_stack.pop(0)
            parent_transform = transform_stack.pop(0)

            # Checking if current frame is cached
            # TODO: Currently using names which are not guarantedd to be unique, should be using
            # a unique index
            transform = None
            if node.name not in self._motion_cache or frame not in self._motion_cache[node.name]:
                rotx = self._frames[frame][node.rotx_idx] if node.rotx_idx is not None else None
                roty = self._frames[frame][node.roty_idx] if node.roty_idx is not None else None
                rotz = self._frames[frame][node.rotz_idx] if node.rotz_idx is not None else None

                offx = self._frames[frame][node.offx_idx] if node.offx_idx is not None else 0
                offy = self._frames[frame][node.offy_idx] if node.offy_idx is not None else 0
                offz = self._frames[frame][node.offz_idx] if node.offz_idx is not None else 0

                if rotx:
                    c = math.cos(math.radians(rotx))
                    s = math.sin(math.radians(rotx))

                    Rx = glm.mat4([1.0, 0.0, 0.0, 0.0],
                                  [0.0, c, s, 0.0],
                                  [0.0, -s, c, 0.0],
                                  [0.0, 0.0, 0.0, 1.0])

                else:
                    Rx = copy.copy(self._identity)

                if roty:
                    c = math.cos(math.radians(roty))
                    s = math.sin(math.radians(roty))

                    Ry = glm.mat4([c, 0.0, -s, 0.0],
                                  [0.0, 1.0, 0.0, 0.0],
                                  [s, 0.0, c, 0.0],
                                  [0.0, 0.0, 0.0, 1.0])

                else:
                    Ry = copy.copy(self._identity)

                if rotz:
                    c = math.cos(math.radians(rotz))
                    s = math.sin(math.radians(rotz))

                    Rz = glm.mat4([c, s, 0.0, 0.0],
                                  [-s, c, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 0.0],
                                  [0.0, 0.0, 0.0, 1.0])

                else:
                    Rz = copy.copy(self._identity)                    

                # Right to left multiplication
                # R: Rz * Rx * Ry

                # THe multiplication order depends on how they are specified in the file
                R = node.compose_rotations_ordered(Rx, Ry, Rz)
                T = glm.mat4([[1.0, 0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0],
                              [offx + node.offset[0], offy + node.offset[1], offz + node.offset[2], 1.0]])

                # with open('out.txt', 'a') as ff:
                #     ff.write(f'{node.name}\n {Rx}\n {Ry}\n {Rz}\n {T}\n')

                transform = parent_transform * T * R
                if node.name not in self._motion_cache:
                    self._motion_cache[node.name] = {}
                self._motion_cache[node.name][frame] = transform
            else:
                transform = self._motion_cache[node.name][frame]

            if len(node.children) <= 1 and callback:
                # Need to expand in 3 dimensions
                if not hasattr(node, 'rest_rot'):
                    up = glm.vec3(0.0, 1.0, 0.0)
                    initial_dir_unnorm = glm.vec3(node.estimated_length)
                    initial_dir = glm.normalize(initial_dir_unnorm)
                    ortho = glm.cross(up, initial_dir)
                    angle = math.acos(glm.dot(up, initial_dir))
                    if math.isnan(angle):
                        node.rest_rot = glm.mat4()
                    else:
                        node.rest_rot = glm.rotate(glm.mat4(), angle, ortho)
                    node.dimension = glm.length(initial_dir_unnorm)  # Make it 3d

                callback(self._find_type_bvh(node.name), node.name, transform, node.dimension, node.rest_rot)

            for child in node.children:
                traverse_stack.insert(0, child)
                transform_stack.insert(0, transform)

    def _traverse_bvh_nop(self, root, frame, callback):
        traverse_stack = [root]
        while traverse_stack:
            node = traverse_stack.pop(0)

            rotx = self._frames[frame][node.rotx_idx] if node.rotx_idx is not None else None
            roty = self._frames[frame][node.roty_idx] if node.roty_idx is not None else None
            rotz = self._frames[frame][node.rotz_idx] if node.rotz_idx is not None else None

            callback(self._find_type_bvh(node.name), node.name, rotx, roty, rotz)

            for child in node.children:
                traverse_stack.append(child)
