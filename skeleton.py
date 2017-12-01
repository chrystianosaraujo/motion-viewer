# Python
import enum
import os
import math
import copy

# Motion Viewer
import bvh

# External
import glm

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

    RIGHT_BUTTOCK = 35
    RIGHT_THIGH = 36
    RIGHT_SHIN = 37
    RIGHT_FOOT = 38

    LEFT_BUTTOCK = 39
    LEFT_THIGH = 40
    LEFT_SHIN = 41
    LEFT_FOOT = 42


class UnsupportedError(Exception):
    pass


class FormatError(Exception):
    pass

# Hierarchical collection of joints with associated motion data


class AnimatedSkeleton:
    # Optimized for readability :)
    BVH_JOINT_NAMES = {
        NodeType.HIP: ['hip'],
        NodeType.ABDOMEN: ['abdomen'],
        NodeType.CHEST: ['chest'],
        NodeType.NECK: ['neck'],
        NodeType.HEAD: ['head'],
        NodeType.LEFT_EYE: ['leftEye'],
        NodeType.RIGHT_EYE: ['rightEye'],
        NodeType.RIGHT_COLLAR: ['rCollar'],
        NodeType.RIGHT_SHOULDER: ['rShldr'],
        NodeType.RIGHT_FOREARM: ['rForeArm'],
        NodeType.RIGHT_HAND: ['rHand'],
        NodeType.RIGHT_FINGER_THUMB1: ['rThumb1'],
        NodeType.RIGHT_FINGER_THUMB2: ['rThumb2'],
        NodeType.RIGHT_FINGER_INDEX1: ['rIndex1'],
        NodeType.RIGHT_FINGER_INDEX2: ['rIndex2'],
        NodeType.RIGHT_FINGER_MID1: ['rMid1'],
        NodeType.RIGHT_FINGER_MID2: ['rMid2'],
        NodeType.RIGHT_FINGER_RING1: ['rRing1'],
        NodeType.RIGHT_FINGER_RING2: ['rRing2'],
        NodeType.RIGHT_FINGER_PINKY1: ['rPinky1'],
        NodeType.RIGHT_FINGER_PINKY2: ['rPinky2'],
        NodeType.LEFT_COLLAR: ['lCollar'],
        NodeType.LEFT_SHOULDER: ['lShldr'],
        NodeType.LEFT_FOREARM: ['lForeArm'],
        NodeType.LEFT_HAND: ['lHand'],
        NodeType.LEFT_FINGER_THUMB1: ['lThumb1'],
        NodeType.LEFT_FINGER_THUMB2: ['lThumb2'],
        NodeType.LEFT_FINGER_INDEX1: ['lIndex1'],
        NodeType.LEFT_FINGER_INDEX2: ['lIndex2'],
        NodeType.LEFT_FINGER_MID1: ['lMid1'],
        NodeType.LEFT_FINGER_MID2: ['lMid2'],
        NodeType.LEFT_FINGER_RING1: ['lRing1'],
        NodeType.LEFT_FINGER_RING2: ['lRing2'],
        NodeType.LEFT_FINGER_PINKY1: ['lPinky1'],
        NodeType.LEFT_FINGER_PINKY2: ['lPinky2'],

        NodeType.RIGHT_BUTTOCK: ['rButtock'],
        NodeType.RIGHT_THIGH: ['rThigh'],
        NodeType.RIGHT_SHIN: ['rShin'],
        NodeType.RIGHT_FOOT: ['rFoot'],

        NodeType.LEFT_BUTTOCK: ['lButtock'],
        NodeType.LEFT_THIGH: ['lThigh'],
        NodeType.LEFT_SHIN: ['lShin'],
        NodeType.LEFT_FOOT: ['lFoot']
    }

    def __init__(self):
        self._root = None
        self._frame_time = 0
        self._frames = []

        self._traverse_func = None
        self._motion_cache = {}

        self._identity = glm.mat4()

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

    # TODO: This should be cached
    def _find_type_bvh(self, name):
        for t, names in AnimatedSkeleton.BVH_JOINT_NAMES.items():
            if name in names:
                return t
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
            transform: None
            if node.name not in self._motion_cache or frame not in self._motion_cache[node.name]:
                rotx = self._frames[frame][node.rotx_idx] if node.rotx_idx is not None else None
                roty = self._frames[frame][node.roty_idx] if node.roty_idx is not None else None
                rotz = self._frames[frame][node.rotz_idx] if node.rotz_idx is not None else None

                offx = self._frames[frame][node.offx_idx] if node.offx_idx is not None else 0
                offy = self._frames[frame][node.offy_idx] if node.offy_idx is not None else 0
                offz = self._frames[frame][node.offz_idx] if node.offz_idx is not None else 0

                if rotx:
                    Rx = glm.rotate(self._identity, math.radians(rotx), glm.vec3(1.0, 0.0, 0.0))
                else:
                    Rx = copy.copy(self._identity)

                if roty:
                    Ry = glm.rotate(self._identity, math.radians(roty), glm.vec3(0.0, 1.0, 0.0))
                else:
                    Ry = copy.copy(self._identity)

                if rotz:
                    Rz = glm.rotate(self._identity, math.radians(rotz), glm.vec3(0.0, 0.0, 1.0))
                else:
                    Rz = copy.copy(self._identity)

                # Right to left multiplication
                # R: Rz * Rx * Ry
                R = Rz * Rx * Ry

                # M: TS
                M = glm.translate(R, glm.vec3(offx + node.offset[0], offy + node.offset[1], offz + node.offset[2]))
                transform = parent_transform * M
                if node.name not in self._motion_cache:
                    self._motion_cache[node.name] = {}
                self._motion_cache[node.name][frame] = transform
            else:
                transform = self._motion_cache[node.name][frame]

            up = glm.vec3(0.0, 1.0, 0.0)
            initial_dir_unnorm = glm.vec3(node.estimated_length)
            initial_dir = glm.normalize(initial_dir_unnorm)
            ortho = glm.cross(up, initial_dir)
            angle = math.acos(glm.dot(up, initial_dir))
            rest_rot = glm.rotate(glm.mat4(), angle, ortho)

            if callback:
                callback(self._find_type_bvh(node.name), node.name, transform, glm.length(initial_dir_unnorm), rest_rot)

            for child in node.children:
                traverse_stack.append(child)
                transform_stack.append(transform)

    # Traverses the internal hierarchy top-down calculating the combined
    # transforms for the specified <frame>.
    # <callback> is called for each encountered node
    # callback(node_type : NodeType, node_name : str, transform)
    # Transforms are affine 4x4
    def traverse(self, frame, callback):
        if not callback:
            return

        if frame not in range(0, self.frame_count):
            # error()
            return

        if not self._root:
            # error()
            return

        transform = glm.mat4(1.0)
        self._traverse_func(frame, callback, self._root[0], transform)
