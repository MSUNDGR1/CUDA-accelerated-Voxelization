#include "voxelMap.h"

using namespace std;

voxelMap::voxelMap(std::vector<tri3D> input) {
    for (unsigned int i = 0; i < input.size(); i++) {
        if (input[i].minZ < dMin) dMin = input[i].minZ;
        if (input[i].maxZ > dMax) dMax = input[i].maxZ;

        if (input[i].t_p1.m_x < wMin) { wMin = input[i].t_p1.m_x; }
        else if (input[i].t_p1.m_x > wMax) wMax = input[i].t_p1.m_x;
        if (input[i].t_p2.m_x < wMin) { wMin = input[i].t_p2.m_x; }
        else if (input[i].t_p2.m_x > wMax) wMax = input[i].t_p2.m_x;
        if (input[i].t_p3.m_x < wMin) { wMin = input[i].t_p3.m_x; }
        else if (input[i].t_p3.m_x > wMax) wMax = input[i].t_p3.m_x;

        if (input[i].t_p1.m_y < hMin) { hMin = input[i].t_p1.m_y; }
        else if (input[i].t_p1.m_y > hMax) hMax = input[i].t_p1.m_y;
        if (input[i].t_p2.m_y < hMin) { hMin = input[i].t_p2.m_y; }
        else if (input[i].t_p2.m_y > hMax) hMax = input[i].t_p2.m_y;
        if (input[i].t_p3.m_y < hMin) { hMin = input[i].t_p3.m_y; }
        else if (input[i].t_p3.m_y > hMax) hMax = input[i].t_p3.m_y;
    }
    height = hMax - hMin;
    width = wMax - wMin;
    depth = dMax - dMin;
    fills = new bool** [depth];
    for (int d = 0; d < depth; d++) {
        fills[d] = new bool* [height];
        for (int h = 0; h < height; h++) {
            fills[d][h] = new bool[width]();
        }
    }


    vector<vector<int>> triVecs;
    vector<vector<int>> norms;
    vector<int> minZtris;
    vector<int> maxZtris;
    vector<int> minYtris;
    vector<int> maxYtris;

    vector<int> intermed;

    for (unsigned int i = 0; i < input.size(); i++) {
        intermed.clear();
        intermed.push_back(input[i].t_p1.m_x);
        intermed.push_back(input[i].t_p1.m_y);
        intermed.push_back(input[i].t_p1.m_z);
        triVecs.push_back(intermed);
        intermed.clear();
        intermed.push_back(input[i].t_p2.m_x);
        intermed.push_back(input[i].t_p2.m_y);
        intermed.push_back(input[i].t_p2.m_z);
        triVecs.push_back(intermed);
        intermed.clear();
        intermed.push_back(input[i].t_p3.m_x);
        intermed.push_back(input[i].t_p3.m_y);
        intermed.push_back(input[i].t_p3.m_z);
        triVecs.push_back(intermed);

        intermed.clear();
        intermed.push_back(input[i].N.m_x);
        intermed.push_back(input[i].N.m_y);
        intermed.push_back(input[i].N.m_z);
        norms.push_back(intermed);

        minZtris.push_back(input[i].minZ);
        maxZtris.push_back(input[i].maxZ);
        minYtris.push_back(input[i].minY);
        maxYtris.push_back(input[i].maxY);
    }

   /* voxel::voxelize(triVecs, norms, width, height, depth, minZtris,
        maxZtris, minYtris, maxYtris, fills);*/
    voxel::rayVoxel(triVecs, width, height, depth, minZtris, maxZtris,
        minYtris, maxYtris, fills);

}


bool voxelMap::compareTri3DMinZ(tri3D* first, tri3D* second) {
    return (first->minZ < second->minZ);
}

bool voxelMap::compareTri3DMaxZ(tri3D* first, tri3D* second) {
    return (first->maxZ < second->maxZ);
}

voxelMap::~voxelMap() {}