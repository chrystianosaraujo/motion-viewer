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

    return all_faces_vertices, all_vertices_normal


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
