import numpy as np
import math
import collections

Mesh = collections.namedtuple("Mesh", "vertices normals")

def normalize_vector(vec):
    return vec / np.linalg.norm(vec)

def create_cube_mesh():
    """ This function creates a triangle mesh for a cube.
    The created mesh is suited for being used with glDrawArrays.

    Return:
        Array of vertices
        Array of normals
    """

    vertices = [[-0.5, 0.0, -0.5],
                [0.5, 0.0, -0.5],
                [0.5, 0.0, 0.5],
                [-0.5, 0.0, 0.5],
                [-0.5, 1.0, -0.5],
                [0.5, 1.0, -0.5],
                [0.5, 1.0, 0.5],
                [-0.5, 1.0, 0.5]]

    triangles = [[0, 3, 1],  # Bottom Face
                 [1, 3, 2],  # Bottom Face
                 [4, 5, 7],  # Top Face
                 [5, 7, 6],  # Top Face
                 [1, 6, 5],  # Right Face
                 [1, 2, 6],  # Right Face
                 [4, 7, 0],  # Left Face
                 [0, 7, 3],  # Left Face
                 [0, 5, 4],  # Front Face
                 [0, 1, 5],  # Front Face
                 [7, 6, 3],  # Back Face
                 [6, 2, 3]]  # Back Face

    normals = [[0.0, 1.0, 0.0],  # Bottom Face
               [0.0, -1.0, 0.0],  # Top Face
               [-1.0, 0.0, 0.0],  # Right Face
               [1.0, 0.0, 0.0],  # Left Face
               [0.0, 0.0, 1.0],  # Front Face
               [0.0, 0.0, -1.0]]  # Back Face

    all_faces_vertices = []
    all_vertices_normal = []
    for i, triangle in enumerate(triangles):
        v0 = vertices[triangle[0]]
        v1 = vertices[triangle[1]]
        v2 = vertices[triangle[2]]

        face_id = i // 2
        normal = normals[face_id]

        all_faces_vertices.extend(v0 + v1 + v2)
        all_vertices_normal.extend(normal * 3)

    return Mesh(vertices = np.asarray(all_faces_vertices , dtype=np.float32),
                normals  = np.asarray(all_vertices_normal, dtype=np.float32))

def create_cylinder_mesh(num_steps):
    """ This function creates a triangle mesh for a cylinder with caps.
    The created mesh is suited for being used with glDrawArrays.

    Return:
        Array of vertices
        Array of normals
    """

    bottom_cap_vertices = []
    top_cap_vertices    = []

    # Generating caps vertices
    half_length = 0.5

    curr_angle = 0.0
    step = math.pi * 2.0 / (num_steps - 1)
    for i in range(num_steps):
        print(i, curr_angle)
        x = math.cos(curr_angle)
        z = math.sin(curr_angle)

        top_cap_vertices.append(np.array([x, half_length * 2.0, z]))
        bottom_cap_vertices.append(np.array([x, 0.0, z]))

        curr_angle += step

    all_vertices = []
    all_normals  = []

    # Generating side faces
    for i in range(num_steps):
        indices = (i, (i + 1) % num_steps)

        bottom_verts = (bottom_cap_vertices[indices[0]],
                        bottom_cap_vertices[indices[1]])

        top_verts = (top_cap_vertices[indices[0]],
                     top_cap_vertices[indices[1]])

        all_vertices.extend(bottom_verts[0])
        all_vertices.extend(bottom_verts[1])
        all_vertices.extend(top_verts[0])

        all_vertices.extend(bottom_verts[1])
        all_vertices.extend(top_verts[1])
        all_vertices.extend(top_verts[0])

        all_normals.extend(normalize_vector(bottom_verts[0]))
        all_normals.extend(normalize_vector(bottom_verts[1]))
        all_normals.extend(normalize_vector(top_verts[0]))

        all_normals.extend(normalize_vector(bottom_verts[1]))
        all_normals.extend(normalize_vector(top_verts[1]))
        all_normals.extend(normalize_vector(top_verts[0]))

    # Generating caps vertices
    caps_center = [np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0])]
    caps_vertices = [top_cap_vertices, bottom_cap_vertices]
    for center, vertices in zip(caps_center, caps_vertices):
        normal = normalize_vector(center)

        for i in range(num_steps):
            indices = (i, (i + 1) % num_steps)

            curr_verts = (vertices[indices[0]],
                          vertices[indices[1]])

            all_vertices.extend(center)
            all_vertices.extend(curr_verts[0])
            all_vertices.extend(curr_verts[1])

            all_normals.extend(normal)
            all_normals.extend(normal)
            all_normals.extend(normal)

    return Mesh(vertices = np.asarray(all_vertices, dtype=np.float32),
                normals  = np.asarray(all_normals , dtype=np.float32))

def create_sphere_mesh(num_steps = 50):
    """ This function creates a triangle mesh for a sphere.
    The created mesh is suited for being used with glDrawArrays.

    Return:
        Array of vertices
        Array of normals
    """

    all_vertices = []
    all_normals  = []

    d_theta = math.pi / (num_steps + 1)
    d_phi = 2.0 * math.pi / num_steps

    for i in range(1, num_steps + 1):
        for j in range(num_steps):
            curr_theta = i * d_theta
            curr_phi   = j * d_phi

            next_theta = (i + 1) * d_theta
            next_phi   = (j + 1) * d_phi

            # Triangle 1
            all_vertices.extend((math.sin(curr_theta) * math.cos(curr_phi),
                                 math.sin(curr_theta) * math.sin(curr_phi),
                                 math.cos(curr_theta)))

            all_vertices.extend((math.sin(next_theta) * math.cos(curr_phi),
                                 math.sin(next_theta) * math.sin(curr_phi),
                                 math.cos(next_theta)))

            all_vertices.extend((math.sin(next_theta) * math.cos(next_phi),
                                 math.sin(next_theta) * math.sin(next_phi),
                                 math.cos(next_theta)))

            # Triangle 2
            all_vertices.extend((math.sin(curr_theta) * math.cos(curr_phi),
                                 math.sin(curr_theta) * math.sin(curr_phi),
                                 math.cos(curr_theta)))

            all_vertices.extend((math.sin(next_theta) * math.cos(next_phi),
                                 math.sin(next_theta) * math.sin(next_phi),
                                 math.cos(next_theta)))

            all_vertices.extend((math.sin(curr_theta) * math.cos(next_phi),
                                 math.sin(curr_theta) * math.sin(next_phi),
                                 math.cos(curr_theta)))

            # Triangle 1
            all_normals.extend((math.sin(curr_theta) * math.cos(curr_phi),
                                math.sin(curr_theta) * math.sin(curr_phi),
                                math.cos(curr_theta)))

            all_normals.extend((math.sin(next_theta) * math.cos(curr_phi),
                                math.sin(next_theta) * math.sin(curr_phi),
                                math.cos(next_theta)))

            all_normals.extend((math.sin(next_theta) * math.cos(next_phi),
                                math.sin(next_theta) * math.sin(next_phi),
                                math.cos(next_theta)))

            # Triangle 2
            all_normals.extend((math.sin(curr_theta) * math.cos(curr_phi),
                                math.sin(curr_theta) * math.sin(curr_phi),
                                math.cos(curr_theta)))

            all_normals.extend((math.sin(next_theta) * math.cos(next_phi),
                                math.sin(next_theta) * math.sin(next_phi),
                                math.cos(next_theta)))

            all_normals.extend((math.sin(curr_theta) * math.cos(next_phi),
                                math.sin(curr_theta) * math.sin(next_phi),
                                math.cos(curr_theta)))

    for j in range(num_steps):
        all_vertices.extend((0.0, 0.0, 1.0))

        all_vertices.extend((math.sin(d_theta) * math.cos(j * d_phi),
                             math.sin(d_theta) * math.sin(j * d_phi),
                             math.cos(d_theta)))

        all_vertices.extend((math.sin(d_theta) * math.cos((j + 1) * d_phi),
                             math.sin(d_theta) * math.sin((j + 1) * d_phi),
                             math.cos(d_theta)))

        all_normals.extend((0.0, 0.0, 1.0))

        all_normals.extend((math.sin(d_theta) * math.cos(j * d_phi),
                            math.sin(d_theta) * math.sin(j * d_phi),
                            math.cos(d_theta)))

        all_normals.extend((math.sin(d_theta) * math.cos((j + 1) * d_phi),
                            math.sin(d_theta) * math.sin((j + 1) * d_phi),
                            math.cos(d_theta)))

    for j in range(num_steps):
        all_vertices.extend((0.0, 0.0, -1.0))

        all_vertices.extend((math.sin(math.pi - d_theta) * math.cos(j * d_phi),
                             math.sin(math.pi - d_theta) * math.sin(j * d_phi),
                             math.cos(math.pi - d_theta)))

        all_vertices.extend((math.sin(math.pi - d_theta) * math.cos((j + 1) * d_phi),
                             math.sin(math.pi - d_theta) * math.sin((j + 1) * d_phi),
                             math.cos(math.pi - d_theta)))

        all_normals.extend((0.0, 0.0, -1.0))

        all_normals.extend((math.sin(math.pi - d_theta) * math.cos(j * d_phi),
                            math.sin(math.pi - d_theta) * math.sin(j * d_phi),
                            math.cos(math.pi - d_theta)))

        all_normals.extend((math.sin(math.pi - d_theta) * math.cos((j + 1) * d_phi),
                            math.sin(math.pi - d_theta) * math.sin((j + 1) * d_phi),
                            math.cos(math.pi - d_theta)))

    return Mesh(vertices = np.asarray(all_vertices, dtype=np.float32),
                normals  = np.asarray(all_normals , dtype=np.float32))


def create_grid_mesh(width, height, divisions):
    cells_x = width / divisions
    cells_z = height / divisions

    vertices = []
    normals = []
    texcoords = []

    start_z = -height / 2
    for z in range(divisions):
        start_x = -width / 2
        for x in range(divisions):
            v0 = [start_x, 0.0, start_z]
            v1 = [start_x + cells_x, 0.0, start_z]
            v2 = [start_x + cells_x, 0.0, start_z + cells_z]
            v3 = [start_x, 0.0, start_z + cells_z]

            t0 = [0.0, 0.0]
            t1 = [1.0, 0.0]
            t2 = [1.0, 1.0]
            t3 = [0.0, 1.0]

            vertices.extend(v0 + v1 + v3 + v1 + v2 + v3)
            texcoords.extend(t0 + t1 + t3 + t1 + t2 + t3)

            n = [0.0, 1.0, 0.0]
            normals += n * 6
            start_x += cells_x
        start_z += cells_z

    return vertices, normals, texcoords
