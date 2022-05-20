import torch
import numpy as np
import MVRegC
import open3d as o3d



def rigid_fit( X, Y, w, eps=0.0001):
    '''
    @param X: source frame [B, N,3]
    @param Y: target frame [B, N,3]
    @param w: weights [B, N,1]
    @param eps:
    @return:
    '''
    # https://ieeexplore.ieee.org/document/88573

    bsize = X.shape[0]
    device = X.device
    W1 = torch.abs(w).sum(dim=1, keepdim=True)
    w_norm = w / (W1 + eps)
    mean_X = (w_norm * X).sum(dim=1, keepdim=True)
    mean_Y = (w_norm * Y).sum(dim=1, keepdim=True)
    Sxy = torch.matmul( (Y - mean_Y).transpose(1,2), w_norm * (X - mean_X) )
    Sxy = Sxy.cpu().double()
    U, D, V = Sxy.svd() # small SVD runs faster on cpu
    S = torch.eye(3)[None].repeat(bsize,1,1).double()
    UV_det = U.det() * V.det()
    S[:, 2:3, 2:3] = UV_det.view(-1, 1,1)
    svT = torch.matmul( S, V.transpose(1,2) )
    R = torch.matmul( U, svT).float().to(device)
    t = mean_Y.transpose(1,2) - torch.matmul( R, mean_X.transpose(1,2) )
    return R, t



def ED_warp(x, g, R, t, w):
    """ Warp a point cloud using the embeded deformation
    https://people.inf.ethz.ch/~sumnerb/research/embdef/Sumner2007EDF.pdf
    :param x: point location
    :param g: anchor location
    :param R: rotation
    :param t: translation
    :param w: weights
    :return:
    """
    y = ( (R @ (x[:,None] - g)[..., None] ).squeeze() + g + t ) * w[...,None]
    y = y.sum(dim=1)
    return y



def map_pixel_to_pcd(valid_pix_mask):
    ''' establish pixel to point cloud mapping, with -1 filling for invalid pixels
    :param valid_pix_mask:
    :return:
    '''
    image_size = valid_pix_mask.shape
    pix_2_pcd_map = torch.cumsum(valid_pix_mask.view(-1), dim=0).view(image_size).long() - 1
    pix_2_pcd_map [~valid_pix_mask] = -1
    return pix_2_pcd_map

def pc_2_uv_np(pcd, intrin):
    '''
    :param pcd: nx3
    :param intrin: 3x3 mat
    :return:
    '''

    X, Y, Z = pcd[:, 0], pcd[:, 1], pcd[:, 2]
    fx, cx, fy, cy = intrin[0, 0], intrin[0, 2], intrin[1, 1], intrin[1, 2]
    u = (fx * X / Z + cx).astype(int)
    v = (fy * Y / Z + cy).astype(int)
    return np.stack([u,v], -1 )

def pc_2_uv(pcd, intrin):
    '''
    :param pcd: nx3
    :param intrin: 3x3 mat
    :return:
    '''

    X, Y, Z = pcd[:, 0], pcd[:, 1], pcd[:, 2]
    fx, cx, fy, cy = intrin[0, 0], intrin[0, 2], intrin[1, 1], intrin[1, 2]
    u = (fx * X / Z + cx).to(torch.long)
    v = (fy * Y / Z + cy).to(torch.long)
    return torch.stack([u,v], -1 )



def depth_2_pc(depth, intrin):
    '''
    :param depth:
    :param intrin: 3x3 mat
    :return:
    '''

    fx, cx, fy, cy = intrin[0,0], intrin[0,2], intrin[1,1], intrin[1,2]
    height, width = depth.shape
    u = np.arange(width) * np.ones([height, width])
    v = np.arange(height) * np.ones([width, height])
    v = np.transpose(v)
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    return np.stack([X, Y, Z])


def depth_to_mesh(depth_image,
                 mask_image,
                 intrin,
                 depth_scale=1000.,
                 max_triangle_distance=0.04):
    """
    :param depth_image:
    :param mask_image:
    :param intrin:
    :param depth_scale:
    :param max_triangle_distance:
    :return:
    """
    width = depth_image.shape[1]
    height = depth_image.shape[0]


    mask_image[mask_image > 0] = 1
    depth_image = depth_image * mask_image

    point_image = depth_2_pc(depth_image / depth_scale, intrin)
    point_image = point_image.astype(np.float32)

    vertices, faces, vertex_pixels = MVRegC.depth_to_mesh(point_image, max_triangle_distance)

    return vertices, faces, vertex_pixels, point_image


def compute_graph_edges(vertices, valid_vertices, faces, node_indices, num_neighbors, node_coverage, USE_ONLY_VALID_VERTICES, ENFORCE_TOTAL_NUM_NEIGHBORS ) :

    num_nodes = node_indices.shape[0]
    num_vertices = vertices.shape[0]

    graph_edges              = -np.ones((num_nodes, num_neighbors), dtype=np.int32)
    graph_edges_weights      =  np.zeros((num_nodes, num_neighbors), dtype=np.float32)
    graph_edges_distances    =  np.zeros((num_nodes, num_neighbors), dtype=np.float32)
    node_to_vertex_distances = -np.ones((num_nodes, num_vertices), dtype=np.float32)

    visible_vertices = np.ones_like(valid_vertices)
    MVRegC.compute_edges_geodesic( vertices, visible_vertices, faces, node_indices,
                                   graph_edges, graph_edges_weights, graph_edges_distances, node_to_vertex_distances,
                                   num_neighbors, node_coverage, USE_ONLY_VALID_VERTICES, ENFORCE_TOTAL_NUM_NEIGHBORS )

    return graph_edges, graph_edges_weights, graph_edges_distances, node_to_vertex_distances


def get_deformation_graph_from_depthmap (depth_image, intrin, config, debug_mode=False):
    '''
    :param depth_image:
    :param intrin:
    :return:
    '''

    #########################################################################
    # Options
    #########################################################################
    # Depth-to-mesh conversion
    max_triangle_distance = config.max_triangle_distance
    # Node sampling and edges computation
    node_coverage = config.node_coverage  # in meters
    USE_ONLY_VALID_VERTICES = config.USE_ONLY_VALID_VERTICES
    num_neighbors = config.num_neighbors
    ENFORCE_TOTAL_NUM_NEIGHBORS = config.ENFORCE_TOTAL_NUM_NEIGHBORS
    SAMPLE_RANDOM_SHUFFLE = config.SAMPLE_RANDOM_SHUFFLE
    REMOVE_NODES_WITH_NOT_ENOUGH_NEIGHBORS = config.REMOVE_NODES_WITH_NOT_ENOUGH_NEIGHBORS



    #########################################################################
    """convert depth to mesh"""
    #########################################################################
    width = depth_image.shape[1]
    height = depth_image.shape[0]
    mask_image=depth_image>0
    # fx, cx, fy, cy = intrin[0,0], intrin[0,2], intrin[1,1], intrin[1,2]
    vertices, faces, vertex_pixels, point_image = depth_to_mesh(depth_image, mask_image, intrin, max_triangle_distance=max_triangle_distance, depth_scale=1000.)
    num_vertices = vertices.shape[0]
    num_faces = faces.shape[0]
    assert num_vertices > 0 and num_faces > 0


    #########################################################################
    """Erode mesh, to not sample unstable nodes on the mesh boundary."""
    #########################################################################
    non_eroded_vertices = MVRegC.erode_mesh(vertices, faces, 0, 0)



    #########################################################################
    """Sample graph nodes"""
    #########################################################################
    valid_vertices = non_eroded_vertices
    node_coords, node_indices = MVRegC.sample_nodes ( vertices, valid_vertices, node_coverage, USE_ONLY_VALID_VERTICES, SAMPLE_RANDOM_SHUFFLE)
    num_nodes = node_coords.shape[0]



    #########################################################################
    """visualize surface and non-eroded points"""
    #########################################################################
    if debug_mode:
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
        mesh.compute_vertex_normals()
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices[non_eroded_vertices.reshape(-1), :]))
        pcd_nodes = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(node_coords))
        o3d.visualization.draw_geometries([mesh,  pcd_nodes], mesh_show_back_face=True)


    #########################################################################
    """Compute graph edges"""
    #########################################################################
    graph_edges, graph_edges_weights, graph_edges_distances, node_to_vertex_distances = \
        compute_graph_edges(vertices, valid_vertices, faces, node_indices, num_neighbors, node_coverage, USE_ONLY_VALID_VERTICES, ENFORCE_TOTAL_NUM_NEIGHBORS )
    # graph_edges = MVRegC.compute_edges_euclidean(node_coords,num_neighbors, 0.05)


    #########################################################################
    "Remove nodes"
    #########################################################################
    valid_nodes_mask = np.ones((num_nodes, 1), dtype=bool)
    node_id_black_list = []
    if REMOVE_NODES_WITH_NOT_ENOUGH_NEIGHBORS:
        MVRegC.node_and_edge_clean_up(graph_edges, valid_nodes_mask)
        node_id_black_list = np.where(valid_nodes_mask == False)[0].tolist()
    # else:
    #     print("You're allowing nodes with not enough neighbors!")
    # print("Node filtering: initial num nodes", num_nodes, "| invalid nodes", len(node_id_black_list),
    #       "({})".format(node_id_black_list))




    #########################################################################
    """Compute pixel anchors"""
    #########################################################################
    pixel_anchors = np.zeros((0), dtype=np.int32)
    pixel_weights = np.zeros((0), dtype=np.float32)
    MVRegC.compute_pixel_anchors_geodesic( node_to_vertex_distances, valid_nodes_mask, vertices, vertex_pixels, pixel_anchors, pixel_weights, width, height, node_coverage)
    # print("Valid pixels:", np.sum(np.all(pixel_anchors != -1, axis=2)))



    #########################################################################
    """filter invalid nodes"""
    #########################################################################
    node_coords = node_coords[valid_nodes_mask.squeeze()]
    node_indices = node_indices[valid_nodes_mask.squeeze()]
    graph_edges = graph_edges[valid_nodes_mask.squeeze()]
    graph_edges_weights = graph_edges_weights[valid_nodes_mask.squeeze()]
    graph_edges_distances = graph_edges_distances[valid_nodes_mask.squeeze()]


    #########################################################################
    """Check that we have enough nodes"""
    #########################################################################
    num_nodes = node_coords.shape[0]
    if (num_nodes == 0):
        print("No nodes! Exiting ...")
        exit()


    #########################################################################
    """Update node ids"""
    #########################################################################
    if len(node_id_black_list) > 0:
        # 1. Mapping old indices to new indices
        count = 0
        node_id_mapping = {}
        for i, is_node_valid in enumerate(valid_nodes_mask):
            if not is_node_valid:
                node_id_mapping[i] = -1
            else:
                node_id_mapping[i] = count
                count += 1

        # 2. Update graph_edges using the id mapping
        for node_id, graph_edge in enumerate(graph_edges):
            # compute mask of valid neighbors
            valid_neighboring_nodes = np.invert(np.isin(graph_edge, node_id_black_list))

            # make a copy of the current neighbors' ids
            graph_edge_copy = np.copy(graph_edge)
            graph_edge_weights_copy = np.copy(graph_edges_weights[node_id])
            graph_edge_distances_copy = np.copy(graph_edges_distances[node_id])

            # set the neighbors' ids to -1
            graph_edges[node_id] = -np.ones_like(graph_edge_copy)
            graph_edges_weights[node_id] = np.zeros_like(graph_edge_weights_copy)
            graph_edges_distances[node_id] = np.zeros_like(graph_edge_distances_copy)

            count_valid_neighbors = 0
            for neighbor_idx, is_valid_neighbor in enumerate(valid_neighboring_nodes):
                if is_valid_neighbor:
                    # current neighbor id
                    current_neighbor_id = graph_edge_copy[neighbor_idx]

                    # get mapped neighbor id
                    if current_neighbor_id == -1:
                        mapped_neighbor_id = -1
                    else:
                        mapped_neighbor_id = node_id_mapping[current_neighbor_id]

                    graph_edges[node_id, count_valid_neighbors] = mapped_neighbor_id
                    graph_edges_weights[node_id, count_valid_neighbors] = graph_edge_weights_copy[neighbor_idx]
                    graph_edges_distances[node_id, count_valid_neighbors] = graph_edge_distances_copy[neighbor_idx]

                    count_valid_neighbors += 1

            # normalize edges' weights
            sum_weights = np.sum(graph_edges_weights[node_id])
            if sum_weights > 0:
                graph_edges_weights[node_id] /= sum_weights
            else:
                print("Hmmmmm", graph_edges_weights[node_id])
                raise Exception("Not good")

        # 3. Update pixel anchors using the id mapping (note that, at this point, pixel_anchors is already free of "bad" nodes, since
        # 'compute_pixel_anchors_geodesic_c' was given 'valid_nodes_mask')
        MVRegC.update_pixel_anchors(node_id_mapping, pixel_anchors)



    #########################################################################
    """Compute clusters."""
    #########################################################################
    graph_clusters = -np.ones((graph_edges.shape[0], 1), dtype=np.int32)
    clusters_size_list = MVRegC.compute_clusters(graph_edges, graph_clusters)
    # print("clusters_size_list", clusters_size_list)


    #########################################################################
    """visualize valid pixels"""
    #########################################################################
    if debug_mode:
        from utils.vis import save_grayscale_image
        pixel_anchors_image = np.sum(pixel_anchors, axis=2)
        pixel_anchors_mask = np.copy(pixel_anchors_image).astype(np.uint8)
        raw_pixel_mask = np.copy(pixel_anchors_image).astype(np.uint8) * 0
        pixel_anchors_mask[pixel_anchors_image == -4] = 0
        raw_pixel_mask[depth_image > 0] = 1
        pixel_anchors_mask[pixel_anchors_image > -4] = 1
        save_grayscale_image("../output/pixel_anchors_mask.jpeg", pixel_anchors_mask)
        save_grayscale_image("../output/depth_mask.jpeg", raw_pixel_mask)


    #########################################################################
    """visualize graph"""
    #########################################################################
    if debug_mode:
        from utils.vis import node_o3d_spheres, merge_meshes

        node_mesh = node_o3d_spheres(node_coords, node_coverage * 0.1, color=[1, 0, 0])
        edges_pairs = []
        for node_id, edges in enumerate(graph_edges):
            for neighbor_id in edges:
                if neighbor_id == -1:
                    break
                edges_pairs.append([node_id, neighbor_id])
        from utils.line_mesh import LineMesh

        line_mesh = LineMesh(node_coords, edges_pairs, radius=0.002)
        edge_mesh = line_mesh.cylinder_segments
        edge_mesh = merge_meshes(edge_mesh)
        edge_mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh, node_mesh, edge_mesh], mesh_show_back_face=True)


    model_data = {
        "graph_nodes": torch.from_numpy( node_coords),
        "graph_edges": torch.from_numpy( graph_edges).long(),
        "graph_edges_weights": torch.from_numpy( graph_edges_weights),
        "graph_clusters": graph_clusters,
        "pixel_anchors": torch.from_numpy( pixel_anchors),
        "pixel_weights": torch.from_numpy( pixel_weights),
        "point_image": torch.from_numpy( point_image).permute(1,2,0)
    }


    return model_data




def partition_arg_topK(matrix, K, axis=0):
    """ find index of K smallest entries along a axis
    perform topK based on np.argpartition
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: 0 or 1. dimension to be sorted.
    :return:
    """
    a_part = np.argpartition(matrix, K, axis=axis)
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        a_sec_argsort_K = np.argsort(matrix[a_part[0:K, :], row_index], axis=axis)
        return a_part[0:K, :][a_sec_argsort_K, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        a_sec_argsort_K = np.argsort(matrix[column_index, a_part[:, 0:K]], axis=axis)
        return a_part[:, 0:K][column_index, a_sec_argsort_K]



def knn_point_np(k, reference_pts, query_pts):
    '''
    :param k: number of k in k-nn search
    :param reference_pts: (N, 3) float32 array, input points
    :param query_pts: (M, 3) float32 array, query points
    :return:
        val: (batch_size, npoint, k) float32 array, L2 distances
        idx: (batch_size, npoint, k) int32 array, indices to input points
    '''

    N, _ = reference_pts.shape
    M, _ = query_pts.shape
    reference_pts = reference_pts.reshape(1, N, -1).repeat(M, axis=0)
    query_pts = query_pts.reshape(M, 1, -1).repeat(N, axis=1)
    dist = np.sum((reference_pts - query_pts) ** 2, -1)
    idx = partition_arg_topK(dist, K=k, axis=1)
    val = np.take_along_axis ( dist , idx, axis=1)
    return np.sqrt(val), idx


def multual_nn_correspondence(src_pcd_deformed, tgt_pcd, search_radius=0.3, knn=1):

    src_idx = np.arange(src_pcd_deformed.shape[0])

    s2t_dists, ref_tgt_idx = knn_point_np (knn, tgt_pcd, src_pcd_deformed)
    s2t_dists, ref_tgt_idx = s2t_dists[:,0], ref_tgt_idx [:, 0]
    valid_distance = s2t_dists < search_radius

    _, ref_src_idx = knn_point_np (knn, src_pcd_deformed, tgt_pcd)
    _, ref_src_idx = _, ref_src_idx [:, 0]

    cycle_src_idx = ref_src_idx [ ref_tgt_idx ]

    is_mutual_nn = cycle_src_idx == src_idx

    mutual_nn = np.logical_and( is_mutual_nn, valid_distance)
    correspondences = np.stack([src_idx [ mutual_nn ], ref_tgt_idx[mutual_nn] ] , axis=0)

    return correspondences

def xyz_2_uv(pcd, intrin):
    ''' np function
    :param pcd: nx3
    :param intrin: 3x3 mat
    :return:
    '''

    X, Y, Z = pcd[:, 0], pcd[:, 1], pcd[:, 2]
    fx, cx, fy, cy = intrin[0, 0], intrin[0, 2], intrin[1, 1], intrin[1, 2]
    u = (fx * X / Z + cx)
    v = (fy * Y / Z + cy)

    if type(u) == np.ndarray:
        return np.stack([u, v], -1).astype(int)
    else :
        return torch.stack( [u,v], -1).long()


