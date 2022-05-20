#include "cpu/image_proc.h"

#include <Eigen/Dense>
#include <map>
#include <string>
#include <cmath>
#include <numeric> //std::iota
#include <iostream>

#include <set>
#include <vector>
#include <algorithm>
#include <random>

using namespace std;
using std::vector;



namespace image_proc {

    std::tuple< py::array_t<int>, py::array_t<int> > printArray( ) {

        int n = 10;
        py::array_t<int> A = py::array_t<int>({ n, 1 });
//        memset(A, 0, sizeof(int) * n);

        py::array_t<int> B = py::array_t<int>({ n, 1 });

//        for ( int i =0 ; i < n ; i ++ ){
//            *A.mutable_data(i,0) = i;
//            *B.mutable_data(i,0) = i+100;
//        }
//
//        //"""generate random index"""
//        std::vector<int> C(n);
//        std::iota(std::begin(C), std::end(C), 0);
//        std::default_random_engine re{std::random_device{}()};
//        std::shuffle(std::begin(C), std::end(C), re);
//
//        py::array_t<int> D = py::array_t<int>({ n, 1 });
//        for ( int i =0 ; i < n ; i ++ ){
//            *D.mutable_data(i,0) = C[i];
//        }
//        py::array_t<int> E = py::array_t<int>({ n, 1 });
//
////        *E=&C[0]

//        std::copy(C.begin(), C.end(), E);


        return std::make_tuple(A, B);

    }


    std::tuple< py::array_t<float>, py::array_t<int>, py::array_t<int> > depthToMesh( const py::array_t<float>& pointImage, float maxTriangleEdgeDistance ) {
        int width = pointImage.shape(2);
        int height = pointImage.shape(1);

        // Compute valid pixel vertices and faces.
        // We also need to compute the pixel -> vertex index mapping for
        // computation of faces.
        // We connect neighboring pixels on the square into two triangles.
        // We only select valid triangles, i.e. with all valid vertices and
        // not too far apart.
        // Important: The triangle orientation is set such that the normals
        // point towards the camera.
        std::vector<Eigen::Vector3f> vertices;
        std::vector<Eigen::Vector3i> faces;
        std::vector<Eigen::Vector2i> pixels;

        int vertexIdx = 0;
        std::vector<int> mapPixelToVertexIdx(width * height, -1);

        for (int y = 0; y < height - 1; y++) {
            for (int x = 0; x < width - 1; x++) {
                Eigen::Vector3f obs00(*pointImage.data(0, y, x), *pointImage.data(1, y, x), *pointImage.data(2, y, x));
                Eigen::Vector3f obs01(*pointImage.data(0, y + 1, x), *pointImage.data(1, y + 1, x), *pointImage.data(2, y + 1, x));
                Eigen::Vector3f obs10(*pointImage.data(0, y, x + 1), *pointImage.data(1, y, x + 1), *pointImage.data(2, y, x + 1));
                Eigen::Vector3f obs11(*pointImage.data(0, y + 1, x + 1), *pointImage.data(1, y + 1, x + 1), *pointImage.data(2, y + 1, x + 1));

                int idx00 = y * width + x;
                int idx01 = (y + 1) * width + x;
                int idx10 = y * width + (x + 1);
                int idx11 = (y + 1) * width + (x + 1);

                bool valid00 = obs00.z() > 0;
                bool valid01 = obs01.z() > 0;
                bool valid10 = obs10.z() > 0;
                bool valid11 = obs11.z() > 0;

                if (valid00 && valid01 && valid10) {
                    float d0 = (obs00 - obs01).norm();
                    float d1 = (obs00 - obs10).norm();
                    float d2 = (obs01 - obs10).norm();

                    if (d0 <= maxTriangleEdgeDistance && d1 <= maxTriangleEdgeDistance && d2 <= maxTriangleEdgeDistance) {
                        int vIdx0 = mapPixelToVertexIdx[idx00];
                        int vIdx1 = mapPixelToVertexIdx[idx01];
                        int vIdx2 = mapPixelToVertexIdx[idx10];

                        if (vIdx0 == -1) {
                            vIdx0 = vertexIdx;
                            mapPixelToVertexIdx[idx00] = vertexIdx;
                            vertices.push_back(obs00);
                            pixels.push_back(Eigen::Vector2i(x, y));
                            vertexIdx++;
                        }
                        if (vIdx1 == -1) {
                            vIdx1 = vertexIdx;
                            mapPixelToVertexIdx[idx01] = vertexIdx;
                            vertices.push_back(obs01);
                            pixels.push_back(Eigen::Vector2i(x, y + 1));
                            vertexIdx++;
                        }
                        if (vIdx2 == -1) {
                            vIdx2 = vertexIdx;
                            mapPixelToVertexIdx[idx10] = vertexIdx;
                            vertices.push_back(obs10);
                            pixels.push_back(Eigen::Vector2i(x + 1, y));
                            vertexIdx++;
                        }

                        faces.push_back(Eigen::Vector3i(vIdx0, vIdx1, vIdx2));
                    }
                }

                if (valid01 && valid10 && valid11) {
                    float d0 = (obs10 - obs01).norm();
                    float d1 = (obs10 - obs11).norm();
                    float d2 = (obs01 - obs11).norm();

                    if (d0 <= maxTriangleEdgeDistance && d1 <= maxTriangleEdgeDistance && d2 <= maxTriangleEdgeDistance) {
                        int vIdx0 = mapPixelToVertexIdx[idx11];
                        int vIdx1 = mapPixelToVertexIdx[idx10];
                        int vIdx2 = mapPixelToVertexIdx[idx01];

                        if (vIdx0 == -1) {
                            vIdx0 = vertexIdx;
                            mapPixelToVertexIdx[idx11] = vertexIdx;
                            vertices.push_back(obs11);
                            pixels.push_back(Eigen::Vector2i(x + 1, y + 1));
                            vertexIdx++;
                        }
                        if (vIdx1 == -1) {
                            vIdx1 = vertexIdx;
                            mapPixelToVertexIdx[idx10] = vertexIdx;
                            vertices.push_back(obs10);
                            pixels.push_back(Eigen::Vector2i(x + 1, y));
                            vertexIdx++;
                        }
                        if (vIdx2 == -1) {
                            vIdx2 = vertexIdx;
                            mapPixelToVertexIdx[idx01] = vertexIdx;
                            vertices.push_back(obs01);
                            pixels.push_back(Eigen::Vector2i(x, y + 1));
                            vertexIdx++;
                        }

                        faces.push_back(Eigen::Vector3i(vIdx0, vIdx1, vIdx2));
                    }
                }
            }
        }

        // Convert to numpy array.
        int nVertices = vertices.size();
        int nFaces = faces.size();

        py::array_t<float> vertexPositions = py::array_t<float> ({ nVertices, 3 });
        py::array_t<int> vertexPixels= py::array_t<int> ( { nVertices, 2 } );
        py::array_t<int> faceIndices= py::array_t<int> ( { nFaces, 3 } );


        if (nVertices > 0 && nFaces > 0) {

            for (int i = 0; i < nVertices; i++) {
                *vertexPositions.mutable_data(i, 0) = vertices[i].x();
                *vertexPositions.mutable_data(i, 1) = vertices[i].y();
                *vertexPositions.mutable_data(i, 2) = vertices[i].z();

                *vertexPixels.mutable_data(i, 0) = pixels[i].x();
                *vertexPixels.mutable_data(i, 1) = pixels[i].y();
            }

            for (int i = 0; i < nFaces; i++) {
                *faceIndices.mutable_data(i, 0) = faces[i].x();
                *faceIndices.mutable_data(i, 1) = faces[i].y();
                *faceIndices.mutable_data(i, 2) = faces[i].z();
            }
        }

        return std::make_tuple(vertexPositions, faceIndices, vertexPixels);

    }


} //namespace image_proc