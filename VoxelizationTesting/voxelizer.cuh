#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <stdlib.h>

namespace voxel {
	void voxelize(
		const std::vector<std::vector<int>> triVecs,
		const std::vector<std::vector<int>> norms,
		const int width, const int height, const int depth,
		const std::vector<int> minZTris,
		const std::vector<int> maxZTris,
		const std::vector<int> minYTris,
		const std::vector<int> maxYTris,
		bool*** fills
	);

	bool*** filler(bool*** shellFills, int xH, int yH, int zH, const std::vector<std::vector<int>> norms, std::vector<int>*** triInts);

	void voxelizeAngle(const std::vector<std::vector<int>> triVecs,
		const std::vector<std::vector<int>> norms,
		const int width, const int height, const int depth,
		const std::vector<int> minZTris,
		const std::vector<int> maxZTris,
		const std::vector<int> minYTris,
		const std::vector<int> maxYTris,
		bool*** fills);
	void rayVoxel(const std::vector<std::vector<int>> triVecs,
		const int width, const int height, const int depth,
		const std::vector<int> minZTris,
		const std::vector<int> maxZTris,
		const std::vector<int> minYTris,
		const std::vector<int> maxYTris,
		bool*** fills, std::vector<int>*** triInts);
}