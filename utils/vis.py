import os
import time
import copy
import open3d as o3d
import numpy as np

from skimage import io

import torch

def visualize_pcds(src_pcd=None, tgt_pcd=None, warped_pcd=None, rigidity=None):
    import mayavi.mlab as mlab
    fig = mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))
    c_red = (224. / 255., 0 / 255., 125 / 255.)
    c_pink = (224. / 255., 75. / 255., 232. / 255.)
    c_blue = (0. / 255., 0. / 255., 255. / 255.)
    scale_factor = 0.0125
    # mlab.points3d(s_pc[ :, 0]  , s_pc[ :, 1],  s_pc[:,  2],  scale_factor=scale_factor , color=c_blue)
    if src_pcd is not None:
        if type(src_pcd) == torch.Tensor:
            src_pcd = src_pcd.detach().cpu().numpy()
        mlab.points3d(src_pcd[:, 0], src_pcd[:, 1], src_pcd[:, 2], resolution=6, scale_factor=scale_factor,
                      color=c_pink)

    if warped_pcd is not None:
        if type(warped_pcd) == torch.Tensor:
            warped_pcd = warped_pcd.detach().cpu().numpy()
        mlab.points3d(warped_pcd[:, 0], warped_pcd[:, 1], warped_pcd[:, 2], resolution=6, scale_factor=scale_factor,
                      color=c_pink)

    if tgt_pcd is not None:
        if type(tgt_pcd) == torch.Tensor:
            tgt_pcd = tgt_pcd.detach().cpu().numpy()
        mlab.points3d(tgt_pcd[:, 0], tgt_pcd[:, 1], tgt_pcd[:, 2], resolution=6, scale_factor=scale_factor,
                      color=c_blue)

    if rigidity is not None:
        pcd2 = warped_pcd + 1
        rigidity = rigidity.detach().cpu().numpy()
        rmin, rmax = rigidity.min(), rigidity.max()
        rigidity = (rigidity - rmin) / (rmax - rmin + 1e-6)
        m = mlab.points3d(pcd2[:, 0], pcd2[:, 1], pcd2[:, 2], rigidity, scale_factor=scale_factor, colormap='blue-red',
                          scale_mode='none')
        m.module_manager.scalar_lut_manager.use_default_range = False
        m.module_manager.scalar_lut_manager.data_range = np.array([0, 1])

    mlab.show()

def visualize_pcds_list(pcd_list ):

    import mayavi.mlab as mlab

    fig = mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))

    c_blue = (0. / 255., 0. / 255., 255. / 255.)
    scale_factor = 0.0125

    n_pcd = len(pcd_list)

    clrs = [ (i/n_pcd,i/n_pcd,i/n_pcd) for  i in range(n_pcd) ]

    for i, pcd in  enumerate (pcd_list):
        if type(pcd) == torch.Tensor:
            pcd = pcd.detach().cpu().numpy()
        mlab.points3d(pcd[:, 0], pcd[:, 1], pcd[:, 2], resolution=6, scale_factor=scale_factor, color=clrs[i])

    mlab.show()


def meshgrid(H, W):
    '''
    @param H:
    @param W:
    @return:
    '''
    u = np.arange(0, W).reshape(1,-1)
    v = np.arange(0, H).reshape(-1,1)
    u = np.repeat( u, H, axis=0 )
    v = np.repeat( v, W, axis=1 )
    return u, v

def construct_frame_trimesh(point_image ,pix_mask, mesh_emax = 0.1):
    '''
    @param point_image:
    @param pix_mask:
    @param mesh_emax:
    @return:
    '''
    """
    A---B
    |   |
    D---C
    Two triangles for each pixel square: "ADB" and "DCB"
    right-hand rule for rendering
    :return: indexes of triangles
    """

    _, H, W = point_image.shape

    XYZ = point_image #np.moveaxis(point_image, 0, -1)

    index_x, index_y = meshgrid(H, W )
    index_pix = (index_x + index_y * W)

    A_ind = index_pix[ 1:-1, 1:-1]
    B_ind = index_pix[ 1:-1, 2:]
    C_ind = index_pix[ 2:, 2:]
    D_ind = index_pix[ 2:, 1:-1]

    A_msk = pix_mask[ 1:-1, 1:-1]
    B_msk = pix_mask[ 1:-1, 2:]
    C_msk = pix_mask[ 2:, 2:]
    D_msk = pix_mask[2:, 1:-1]

    A_p3d = XYZ[:, 1:-1, 1:-1]
    B_p3d = XYZ[:, 1:-1, 2:]
    C_p3d = XYZ[:, 2:, 2:]
    D_p3d = XYZ[:, 2:, 1:-1]

    AB_dist = np.linalg.norm(A_p3d - B_p3d, axis=0)
    BC_dist = np.linalg.norm(C_p3d - B_p3d, axis=0)
    CD_dist = np.linalg.norm(C_p3d - D_p3d, axis=0)
    DA_dist = np.linalg.norm(A_p3d - D_p3d, axis=0)
    DB_dist = np.linalg.norm(B_p3d - D_p3d, axis=0)

    AB_mask = (AB_dist < mesh_emax) * A_msk * B_msk
    BC_mask = (BC_dist < mesh_emax) * B_msk * C_msk
    CD_mask = (CD_dist < mesh_emax) * C_msk * D_msk
    DA_mask = (DA_dist < mesh_emax) * D_msk * A_msk
    DB_mask = (DB_dist < mesh_emax) * D_msk * B_msk

    ADB_ind = np.stack([A_ind, D_ind, B_ind]).reshape(3, -1)
    DCB_ind = np.stack([D_ind, C_ind, B_ind]).reshape(3, -1)

    ADB_msk = (AB_mask * DB_mask * DA_mask).reshape(-1)
    DCB_msk = (CD_mask * DB_mask * BC_mask).reshape(-1)

    triangles = np.concatenate([ADB_ind, DCB_ind], axis=1)
    triangles_msk = np.concatenate([ADB_msk, DCB_msk])

    valid_triangles = triangles[:, triangles_msk]

    XYZ = np.moveaxis(XYZ, 0, -1).reshape(-1, 3)
    return  XYZ,   valid_triangles.T


def node_o3d_spheres (node_array, r=0.1, resolution=10, color = [0,1,0]):
    '''
    @param node_array: [N, 3]
    @param r:
    @param resolution:
    @param color:
    @return:
    '''

    N, _ = node_array.shape

    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=resolution)

    vertices  = np.asarray(mesh_sphere.vertices)   # point 3d
    triangles = np.asarray(mesh_sphere.triangles) # index

    num_sphere_vertex, _ = vertices.shape

    vertices = np.expand_dims (vertices, axis=0)
    triangles = np.expand_dims (triangles, axis=0)

    vertices = np.repeat ( vertices , [N], axis=0) # change corr 3D
    triangles = np.repeat ( triangles ,[N], axis=0) # change index

    # reposition vertices
    node_array = np.expand_dims( node_array , axis=1)
    vertices = node_array + vertices
    vertices = vertices.reshape( [-1, 3])

    # change index
    index_offset = np.arange(N).reshape( N, 1, 1) * num_sphere_vertex
    triangles = triangles + index_offset
    triangles = triangles.reshape( [-1, 3])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()

    mesh.paint_uniform_color(color)

    # # o3d.visualization.draw_geometries([ mesh ])
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(mesh)
    # vis.get_render_option().load_from_json("./renderoption.json")
    # vis.run()
    # # vis.capture_screen_image("output/ours_silver-20.jpg")
    # vis.destroy_window()
    return mesh



def save_grayscale_image(filename, image_numpy):
    image_to_save = np.copy(image_numpy)
    image_to_save = (image_to_save * 255).astype(np.uint8)

    if len(image_to_save.shape) == 2:
        io.imsave(filename, image_to_save)
    elif len(image_to_save.shape) == 3:
        assert image_to_save.shape[0] == 1 or image_to_save.shape[-1] == 1
        image_to_save = image_to_save[0]
        io.imsave(filename, image_to_save)





def transform_pointcloud_to_opengl_coords(points_cv):
    assert len(points_cv.shape) == 2 and points_cv.shape[1] == 3

    T_opengl_cv = np.array(
        [[1.0, 0.0, 0.0],
         [0.0, -1.0, 0.0],
         [0.0, 0.0, -1.0]]
    )

    # apply 180deg rotation around 'x' axis to transform the mesh into OpenGL coordinates
    point_opengl = np.matmul(points_cv, T_opengl_cv.transpose())

    return point_opengl


class CustomDrawGeometryWithKeyCallback():
    def __init__(self, geometry_dict, alignment_dict, corresp_set):
        self.added_source_pcd = True
        self.added_source_obj = False
        self.added_target_pcd = False
        self.added_graph = False

        self.added_both = False

        self.added_corresp = False
        self.added_weighted_corresp = False

        self.aligned = False

        self.rotating = False
        self.stop_rotating = False

        self.source_pcd = geometry_dict["source_pcd"]
        self.source_obj = geometry_dict["source_obj"]
        self.target_pcd = geometry_dict["target_pcd"]
        self.graph      = geometry_dict["graph"]

        # align source to target
        self.valid_source_points_cached = alignment_dict["valid_source_points"]
        self.valid_source_colors_cached = copy.deepcopy(self.source_obj.colors)
        self.line_segments_unit_cached  = alignment_dict["line_segments_unit"]
        self.line_lengths_cached        = alignment_dict["line_lengths"]

        self.good_matches_set          = corresp_set["good_matches_set"]
        self.good_weighted_matches_set = corresp_set["good_weighted_matches_set"]
        self.bad_matches_set           = corresp_set["bad_matches_set"]
        self.bad_weighted_matches_set  = corresp_set["bad_weighted_matches_set"]

        # correspondences lines
        self.corresp_set = corresp_set

    def remove_both_pcd_and_object(self, vis, ref):
        if ref == "source":
            if self.added_source_pcd:
                vis.remove_geometry(self.source_pcd)
                self.added_source_pcd = False

            if self.added_source_obj:
                vis.remove_geometry(self.source_obj)
                self.added_source_obj = False
        elif ref == "target":
            if self.added_target_pcd:
                vis.remove_geometry(self.target_pcd)
                self.added_target_pcd = False

    def clear_for(self, vis, ref):
        if self.added_both:
            if ref == "target":
                vis.remove_geometry(self.source_obj)
                self.added_source_obj = False

            self.added_both = False

    def get_name_of_object_to_record(self):
        if self.added_both:
            return "both"

        if self.added_graph:
            if self.added_source_pcd:
                return "source_pcd_with_graph"
            if self.added_source_obj:
                return "source_obj_with_graph"
        else:
            if self.added_source_pcd:
                return "source_pcd"
            if self.added_source_obj:
                return "source_obj"

        if self.added_target_pcd:
            return "target_pcd"

    def custom_draw_geometry_with_key_callback(self):

        def toggle_graph(vis):
            if self.graph is None:
                print("You didn't pass me a graph!")
                return False

            param = vis.get_view_control().convert_to_pinhole_camera_parameters()

            if self.added_graph:
                for g in self.graph:
                    vis.remove_geometry(g)
                self.added_graph = False
            else:
                for g in self.graph:
                    vis.add_geometry(g)
                self.added_graph = True

            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(param)

            return False

        def toggle_obj(vis):
            print("::toggle_obj")

            if self.added_both:
                print("-- will not toggle obj. First, press either S or T, for source or target pcd")
                return False

            param = vis.get_view_control().convert_to_pinhole_camera_parameters()

            # Add source obj
            if self.added_source_pcd:
                vis.add_geometry(self.source_obj)
                vis.remove_geometry(self.source_pcd)
                self.added_source_obj = True
                self.added_source_pcd = False
            # Add source pcd
            elif self.added_source_obj:
                vis.add_geometry(self.source_pcd)
                vis.remove_geometry(self.source_obj)
                self.added_source_pcd = True
                self.added_source_obj = False

            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(param)

            return False

        def view_source(vis):
            print("::view_source")

            self.clear_for(vis, "source")

            param = vis.get_view_control().convert_to_pinhole_camera_parameters()

            if not self.added_source_pcd:
                vis.add_geometry(self.source_pcd)
                self.added_source_pcd = True

            if self.added_source_obj:
                vis.remove_geometry(self.source_obj)
                self.added_source_obj = False

            self.remove_both_pcd_and_object(vis, "target")

            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(param)

            return False

        def view_target(vis):
            print("::view_target")

            self.clear_for(vis, "target")

            param = vis.get_view_control().convert_to_pinhole_camera_parameters()

            if not self.added_target_pcd:
                vis.add_geometry(self.target_pcd)
                self.added_target_pcd = True

            self.remove_both_pcd_and_object(vis, "source")

            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(param)

            return False

        def view_both(vis):
            print("::view_both")

            param = vis.get_view_control().convert_to_pinhole_camera_parameters()

            if self.added_source_pcd:
                vis.add_geometry(self.source_obj)
                vis.remove_geometry(self.source_pcd)
                self.added_source_obj = True
                self.added_source_pcd = False

            if not self.added_source_obj:
                vis.add_geometry(self.source_obj)
                self.added_source_obj = True

            if not self.added_target_pcd:
                vis.add_geometry(self.target_pcd)
                self.added_target_pcd = True

            self.added_both = True

            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(param)

        def rotate(vis):
            print("::rotate")

            self.rotating = True
            self.stop_rotating = False

            total = 2094
            speed = 5.0
            n = int(total / speed)
            for _ in range(n):
                ctr = vis.get_view_control()

                if self.stop_rotating:
                    return False

                ctr.rotate(speed, 0.0)
                vis.poll_events()
                vis.update_renderer()

            ctr.rotate(0.4, 0.0)
            vis.poll_events()
            vis.update_renderer()

            return False

        def rotate_slightly_left_and_right(vis):
            print("::rotate_slightly_left_and_right")

            self.rotating = True
            self.stop_rotating = False

            if self.added_both and not self.aligned:
                moves = ['lo', 'lo', 'pitch_f', 'pitch_b', 'ri', 'ro']
                totals = [(2094/4)/2, (2094/4)/2, 2094/4, 2094/4, 2094/4, 2094/4]
                abs_speed = 5.0
                abs_zoom = 0.15
                abs_pitch = 5.0
                iters_to_move = [range(int(t / abs_speed)) for t in totals]
                stop_at = [True, True, True, True, False, False]
            else:
                moves = ['ro', 'li']
                total = 2094 / 4
                abs_speed = 5.0
                abs_zoom = 0.03
                iters_to_move = [range(int(total / abs_speed)), range(int(total / abs_speed))]
                stop_at = [True, False]

            for move_idx, move in enumerate(moves):
                if move == 'l' or move == 'lo' or move == 'li':
                    h_speed = abs_speed
                elif move == 'r' or move == 'ro' or move == 'ri':
                    h_speed = -abs_speed

                if move == 'lo':
                    zoom_speed = abs_zoom
                elif move == 'ro':
                    zoom_speed = abs_zoom / 4.0
                elif move == 'li' or move == 'ri':
                    zoom_speed = -abs_zoom

                for _ in iters_to_move[move_idx]:
                    ctr = vis.get_view_control()

                    if self.stop_rotating:
                        return False

                    if move == "pitch_f":
                        ctr.rotate(0.0, abs_pitch)
                        ctr.scale(-zoom_speed/2.0)
                    elif move == "pitch_b":
                        ctr.rotate(0.0, -abs_pitch)
                        ctr.scale(zoom_speed/2.0)
                    else:
                        ctr.rotate(h_speed, 0.0)
                        if move == 'lo' or move == 'ro' or move == 'li' or move == 'ri':
                            ctr.scale(zoom_speed)

                    vis.poll_events()
                    vis.update_renderer()

                # Minor halt before changing direction
                if stop_at[move_idx]:
                    time.sleep(0.5)

                    if move_idx == 0:
                        toggle_correspondences(vis)
                        vis.poll_events()
                        vis.update_renderer()

                    if move_idx == 1:
                        toggle_weighted_correspondences(vis)
                        vis.poll_events()
                        vis.update_renderer()

                    time.sleep(0.5)

            ctr.rotate(0.4, 0.0)
            vis.poll_events()
            vis.update_renderer()

            return False

        def align(vis):
            if not self.added_both:
                return False

            moves = ['lo', 'li', 'ro']
            totals = [(2094/4)/2, (2094/4), 2094/4 + (2094/4)/2]
            abs_speed = 5.0
            abs_zoom = 0.15
            abs_pitch = 5.0
            iters_to_move = [range(int(t / abs_speed)) for t in totals]
            stop_at = [True, True, True]

            # Paint source object yellow, to differentiate target and source better
            # self.source_obj.paint_uniform_color([0, 0.8, 0.506])
            self.source_obj.paint_uniform_color([1, 0.706, 0])
            vis.update_geometry(self.source_obj)
            vis.poll_events()
            vis.update_renderer()

            for move_idx, move in enumerate(moves):
                if move == 'l' or move == 'lo' or move == 'li':
                    h_speed = abs_speed
                elif move == 'r' or move == 'ro' or move == 'ri':
                    h_speed = -abs_speed

                if move == 'lo':
                    zoom_speed = abs_zoom
                elif move == 'ro':
                    zoom_speed = abs_zoom/50.0
                elif move == 'li' or move == 'ri':
                    zoom_speed = -abs_zoom/5.0

                for _ in iters_to_move[move_idx]:
                    ctr = vis.get_view_control()

                    if self.stop_rotating:
                        return False

                    if move == "pitch_f":
                        ctr.rotate(0.0, abs_pitch)
                        ctr.scale(-zoom_speed/2.0)
                    elif move == "pitch_b":
                        ctr.rotate(0.0, -abs_pitch)
                        ctr.scale(zoom_speed/2.0)
                    else:
                        ctr.rotate(h_speed, 0.0)
                        if move == 'lo' or move == 'ro' or move == 'li' or move == 'ri':
                            ctr.scale(zoom_speed)

                    vis.poll_events()
                    vis.update_renderer()

                # Minor halt before changing direction
                if stop_at[move_idx]:
                    time.sleep(0.5)

                    if move_idx == 0:
                        n_iter = 125
                        for align_iter in range(n_iter+1):
                            p = float(align_iter) / n_iter
                            self.source_obj.points = o3d.utility.Vector3dVector( self.valid_source_points_cached + self.line_segments_unit_cached * self.line_lengths_cached * p )
                            vis.update_geometry(self.source_obj)
                            vis.poll_events()
                            vis.update_renderer()

                    time.sleep(0.5)

            self.aligned = True

            return False

        def toggle_correspondences(vis):
            if not self.added_both:
                return False

            param = vis.get_view_control().convert_to_pinhole_camera_parameters()

            if self.added_corresp:
                vis.remove_geometry(self.good_matches_set)
                vis.remove_geometry(self.bad_matches_set)
                self.added_corresp = False
            else:
                vis.add_geometry(self.good_matches_set)
                vis.add_geometry(self.bad_matches_set)
                self.added_corresp = True

            # also remove weighted corresp
            if self.added_weighted_corresp:
                vis.remove_geometry(self.good_weighted_matches_set)
                vis.remove_geometry(self.bad_weighted_matches_set)
                self.added_weighted_corresp = False

            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(param)

            return False

        def toggle_weighted_correspondences(vis):
            if not self.added_both:
                return False

            param = vis.get_view_control().convert_to_pinhole_camera_parameters()

            if self.added_weighted_corresp:
                vis.remove_geometry(self.good_weighted_matches_set)
                vis.remove_geometry(self.bad_weighted_matches_set)
                self.added_weighted_corresp = False
            else:
                vis.add_geometry(self.good_weighted_matches_set)
                vis.add_geometry(self.bad_weighted_matches_set)
                self.added_weighted_corresp = True

            # also remove (unweighted) corresp
            if self.added_corresp:
                vis.remove_geometry(self.good_matches_set)
                vis.remove_geometry(self.bad_matches_set)
                self.added_corresp = False

            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(param)

            return False

        def reload_source_object(vis):
            self.source_obj.points = o3d.utility.Vector3dVector(self.valid_source_points_cached)
            self.source_obj.colors = self.valid_source_colors_cached

            if self.added_both:
                vis.update_geometry(self.source_obj)
                vis.poll_events()
                vis.update_renderer()

        key_to_callback = {}
        key_to_callback[ord("G")] = toggle_graph
        key_to_callback[ord("S")] = view_source
        key_to_callback[ord("T")] = view_target
        key_to_callback[ord("O")] = toggle_obj
        key_to_callback[ord("B")] = view_both
        key_to_callback[ord("C")] = toggle_correspondences
        key_to_callback[ord("W")] = toggle_weighted_correspondences
        key_to_callback[ord(",")] = rotate
        key_to_callback[ord(";")] = rotate_slightly_left_and_right
        key_to_callback[ord("A")] = align
        key_to_callback[ord("Z")] = reload_source_object

        o3d.visualization.draw_geometries_with_key_callbacks([self.source_pcd], key_to_callback)


def merge_meshes(meshes):
    # Compute total number of vertices and faces.
    num_vertices = 0
    num_triangles = 0
    num_vertex_colors = 0
    for i in range(len(meshes)):
        num_vertices += np.asarray(meshes[i].vertices).shape[0]
        num_triangles += np.asarray(meshes[i].triangles).shape[0]
        num_vertex_colors += np.asarray(meshes[i].vertex_colors).shape[0]

    # Merge vertices and faces.
    vertices = np.zeros((num_vertices, 3), dtype=np.float64)
    triangles = np.zeros((num_triangles, 3), dtype=np.int32)
    vertex_colors = np.zeros((num_vertex_colors, 3), dtype=np.float64)

    vertex_offset = 0
    triangle_offset = 0
    vertex_color_offset = 0
    for i in range(len(meshes)):
        current_vertices = np.asarray(meshes[i].vertices)
        current_triangles = np.asarray(meshes[i].triangles)
        current_vertex_colors = np.asarray(meshes[i].vertex_colors)

        vertices[vertex_offset:vertex_offset + current_vertices.shape[0]] = current_vertices
        triangles[triangle_offset:triangle_offset + current_triangles.shape[0]] = current_triangles + vertex_offset
        vertex_colors[vertex_color_offset:vertex_color_offset + current_vertex_colors.shape[0]] = current_vertex_colors

        vertex_offset += current_vertices.shape[0]
        triangle_offset += current_triangles.shape[0]
        vertex_color_offset += current_vertex_colors.shape[0]

    # Create a merged mesh object.
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))
    mesh.paint_uniform_color([1, 0, 0])
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    return mesh