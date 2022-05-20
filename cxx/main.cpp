#include <torch/extension.h>

#include "cpu/image_proc.h"
#include "cpu/graph_proc.h"


// Definitions of all methods in the module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {


    m.def("printArray", &image_proc::printArray, "");
    m.def("depth_to_mesh", &image_proc::depthToMesh, "");





    m.def("erode_mesh", &graph_proc::erode_mesh, "Erode mesh");
    m.def("sample_nodes", &graph_proc::sample_nodes, "sample graph nodes");
    m.def("compute_edges_geodesic", &graph_proc::compute_edges_geodesic, "sample graph nodes via geodesic distance");
    m.def("compute_edges_euclidean", &graph_proc::compute_edges_euclidean, "sample graph nodes via euclidean distance");
    m.def("compute_pixel_anchors_geodesic", &graph_proc::compute_pixel_anchors_geodesic, "");
    m.def("node_and_edge_clean_up", &graph_proc::node_and_edge_clean_up, "");
    m.def("update_pixel_anchors", &graph_proc::update_pixel_anchors, "");
    m.def("compute_clusters", &graph_proc::compute_clusters, "");


}