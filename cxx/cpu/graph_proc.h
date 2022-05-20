#pragma once

#include <torch/extension.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#define GRAPH_K 6

namespace graph_proc {

    /**
	 * Erodes mesh.
	 */
    py::array_t<bool> erode_mesh(const py::array_t<float>& vertexPositions, const py::array_t<int>& faceIndices, int nIterations, int minNeighbors);

    /**
	 * Samples nodes that cover all vertex positions with given node coverage.
	 * Nodes are sampled from vertices, resulting node vertex indices are returned.
	 */
    std::tuple< py::array_t<float>, py::array_t<int> >
    sample_nodes(
            const py::array_t<float>& vertexPositions, const py::array_t<bool>& nonErodedVertices,
//            py::array_t<float>& nodePositions, py::array_t<int>& nodeIndices,
            float nodeCoverage,
            const bool useOnlyValidIndices,
            const bool randomShuffle
    );


    /**
     * Computes the graph edges between nodes, connecting nearest nodes using geodesic
     * distances.
     */
    void compute_edges_geodesic(
            const py::array_t<float>& vertexPositions,
            const py::array_t<bool>& validVertices,
            const py::array_t<int>& faceIndices,
            const py::array_t<int>& nodeIndices,
            py::array_t<int>& graphEdges,
            py::array_t<float>& graphEdgesWeights,
            py::array_t<float>& graphEdgesDistances,
            py::array_t<float>& nodeToVertexDistances,
            const int nMaxNeighbors,
            const float nodeCoverage,
            const bool allow_only_valid_vertices,
            const bool enforce_total_num_neighbors
    );


    /**
     * Computes the graph edges between nodes, connecting nearest nodes using Euclidean
     * distances.
     */
    py::array_t<int> compute_edges_euclidean(const py::array_t<float>& nodePositions, int nMaxNeighbors,  float maxInfluence);

    /**
     * Removes invalid nodes (with less than 2 neighbors).
     */
    void node_and_edge_clean_up(const py::array_t<int>& graph_edges, py::array_t<bool>& valid_nodes_mask);

    /**
     * Computes node clusters based on connectivity.
     * @returns Sizes (number of nodes) of each cluster.
     */
    std::vector<int> compute_clusters(
            const py::array_t<int> graph_edges,
            py::array_t<int> graph_clusters
    );


    /**
	 * For each input pixel it computes 4 nearest anchors, following graph edges. 
	 * It also compute skinning weights for every pixel. 
	 */
    void compute_pixel_anchors_geodesic(
            const py::array_t<float> &node_to_vertex_distance,
            const py::array_t<int> &valid_nodes_mask,
            const py::array_t<float> &vertices,
            const py::array_t<int> &vertex_pixels,
            py::array_t<int>& pixel_anchors,
            py::array_t<float>& pixel_weights,
            const int width, const int height,
            const float node_coverage
    );

    /**
     * For each input pixel it computes 4 nearest anchors, using Euclidean distances.
     * It also compute skinning weights for every pixel.
     */
    void compute_pixel_anchors_euclidean(
            const py::array_t<float>& graphNodes,
            const py::array_t<float>& pointImage,
            float nodeCoverage,
            py::array_t<int>& pixelAnchors,
            py::array_t<float>& pixelWeights
    );

    /**
     * Updates pixel anchor after node id change.
     */
    void update_pixel_anchors(
            const std::map<int, int>& node_id_mapping,
            py::array_t<int>& pixel_anchors
    );

    /**
	 * It samples graph regularly from the image, using pixel-wise connectivity
     * (connecting each pixel with at most 8 neighbors).
	 */
    void construct_regular_graph(
            const py::array_t<float>& pointImage,
            int xNodes, int yNodes,
            float edgeThreshold,
            float maxPointToNodeDistance,
            float maxDepth,
            py::array_t<float>& graphNodes,
            py::array_t<int>& graphEdges,
            py::array_t<int>& pixelAnchors,
            py::array_t<float>& pixelWeights
    );

} // namespace graph_proc