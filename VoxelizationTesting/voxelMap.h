#pragma once
#include <cstdlib>
#include <vector>
#include "tri3D.h"
#include "vector3D.h"
#include "voxelizer.cuh"


class voxelMap
{
public:
	voxelMap(std::vector<tri3D> inTri);
	~voxelMap();

	bool*** fills;

	int height;
	int width;
	int depth;

	int hMin;
	int wMin;
	int dMin;

	int hMax;
	int wMax;
	int dMax;


	bool compareTri3DMaxZ(tri3D* first, tri3D* second);
	bool compareTri3DMinZ(tri3D* first, tri3D* second);
};

