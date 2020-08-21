#include "voxelizer.cuh"


__global__ void vecSubDArr(int* firX, int* firY, int* firZ,
	int* secX, int* secY, int* secZ,
	int* outX, int* outY, int* outZ) {
	outX[blockIdx.x] = firX[blockIdx.x] - secX[blockIdx.x];
	outY[blockIdx.x] = firY[blockIdx.x] - secY[blockIdx.x];
	outZ[blockIdx.x] = firZ[blockIdx.x] - secZ[blockIdx.x];
}

__global__ void vecSubPoint(int* QY, int* QZ,
	int* VX, int* VY, int* VZ,
	int* outX, int* outY, int* outZ) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	outX[index] = blockIdx.x - VX[threadIdx.x];
	outY[index] = *QY - VY[threadIdx.x];
	outZ[index] = *QZ - VZ[threadIdx.x];
}

__global__ void vecCross(int* firX, int* firY, int* firZ,
	int* secX, int* secY, int* secZ,
	int* outX, int* outY, int* outZ) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	outX[index] = (firY[threadIdx.x] * secZ[index]) - (firZ[threadIdx.x] * secY[index]);
	outY[index] = (firZ[threadIdx.x] * secX[index]) - (firX[threadIdx.x] * secZ[index]);
	outZ[index] = (firX[threadIdx.x] * secY[index]) - (firY[threadIdx.x] * secX[index]);
}


__global__ void normDot(int* firX, int* firY, int* firZ,
	int* secX, int* secY, int* secZ,
	bool* check) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int sum = firX[index] * secX[threadIdx.x];
	sum += firY[index] * secY[threadIdx.x];
	sum += firZ[index] * secZ[threadIdx.x];
	if (sum >= 0) check[index] = true;
}

__global__ void checkSum(bool* c1, bool* c2, bool* c3,
	int* actTris, bool* fills) {
	bool checker = false;
	int index;
	int offset = blockIdx.x * (*actTris);
	for (int i = 0; i < (*actTris); i++) {
		index = offset + i;
		checker = (c1[index] | c2[index]) | c3[index];
		if (checker) fills[blockIdx.x] = true;
	}
}

void actTriFind(std::vector<int> minZTris,
	std::vector<int> maxZTris,
	std::vector<int> minYTris,
	std::vector<int> maxYTris,
	std::vector<int>& actTris,
	const int y, const int z) {
	int len = minZTris.size();
	bool checkZ;
	bool checkY;
	for (int i = 0; i < len; i++) {
		checkY = minYTris[i] <= y && maxYTris[i] >= y;
		checkZ = minZTris[i] <= z && maxZTris[i] >= z;
		if (checkY && checkZ) actTris.push_back(i);
	}
}

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
	) {
		/*fills = new bool** [depth];
		for (int d = 0; d < depth; d++) {
			fills[d] = new bool* [height];
			for (int h = 0; h < height; h++) {
				fills[d][h] = new bool[width];
			}
		}*/
		int numTris = minZTris.size();
		int size;
		int* ax, * ay, * az,
			* bx, * by, * bz,
			* cx, * cy, * cz;

		int* NX, * NY, * NZ;

		int* d_ax, * d_ay, * d_az,
			* d_bx, * d_by, * d_bz,
			* d_cx, * d_cy, * d_cz;

		int* d_BAX, * d_BAY, * d_BAZ,
			* d_CBX, * d_CBY, * d_CBZ,
			* d_ACX, * d_ACY, * d_ACZ;

		int* d_QAX, * d_QAY, * d_QAZ,
			* d_QBX, * d_QBY, * d_QBZ,
			* d_QCX, * d_QCY, * d_QCZ;

		int* d_QY, * d_QZ;

		int* d_BAQAX, * d_BAQAY, * d_BAQAZ,
			* d_CBQBX, * d_CBQBY, * d_CBQBZ,
			* d_ACQCX, * d_ACQCY, * d_ACQCZ;

		int* d_NX, * d_NY, * d_NZ;
		bool* d_C1, * d_C2, * d_C3;
		bool* d_fill; int* d_numTris;
		std::vector<int> activeTris;
		for (int d = 0; d < depth; d++) {
			for (int h = 0; h < height; h++) {
				activeTris.clear();
				actTriFind(minZTris, maxZTris, minYTris, maxYTris, activeTris, h, d);
				int numTris = activeTris.size();
				size = sizeof(int) * numTris;
				ax = (int*)malloc(size); ay = (int*)malloc(size); az = (int*)malloc(size);
				bx = (int*)malloc(size); by = (int*)malloc(size); bz = (int*)malloc(size);
				cx = (int*)malloc(size); cy = (int*)malloc(size); cz = (int*)malloc(size);
				NX = (int*)malloc(size); NY = (int*)malloc(size); NZ = (int*)malloc(size);
				for (int i = 0; i < numTris; i++) {
					std::vector<int> actVecA = triVecs[(activeTris[i] * 3)];
					std::vector<int> actVecB = triVecs[(activeTris[i] * 3) + 1];
					std::vector<int> actVecC = triVecs[(activeTris[i] * 3) + 2];
					ax[i] = actVecA[0]; ay[i] = actVecA[1]; az[i] = actVecA[2];
					bx[i] = actVecB[0]; by[i] = actVecB[1]; bz[i] = actVecB[2];
					cx[i] = actVecC[0]; cy[i] = actVecC[1]; cz[i] = actVecC[2];
					NX[i] = norms[activeTris[i]][0]; NY[i] = norms[activeTris[i]][1]; NZ[i] = norms[activeTris[i]][2];
				}
				cudaMalloc((void**)&d_ax, size); cudaMalloc((void**)&d_ay, size); cudaMalloc((void**)&d_az, size);
				cudaMalloc((void**)&d_bx, size); cudaMalloc((void**)&d_by, size); cudaMalloc((void**)&d_bz, size);
				cudaMalloc((void**)&d_cx, size); cudaMalloc((void**)&d_cy, size); cudaMalloc((void**)&d_cz, size);

				cudaMalloc((void**)&d_BAX, size); cudaMalloc((void**)&d_BAY, size); cudaMalloc((void**)&d_BAZ, size);
				cudaMalloc((void**)&d_CBX, size); cudaMalloc((void**)&d_CBY, size); cudaMalloc((void**)&d_CBZ, size);
				cudaMalloc((void**)&d_ACX, size); cudaMalloc((void**)&d_ACY, size); cudaMalloc((void**)&d_ACZ, size);

				cudaMemcpy(d_ax, ax, size, cudaMemcpyHostToDevice); cudaMemcpy(d_ay, ay, size, cudaMemcpyHostToDevice); cudaMemcpy(d_az, az, size, cudaMemcpyHostToDevice);
				cudaMemcpy(d_bx, bx, size, cudaMemcpyHostToDevice); cudaMemcpy(d_by, by, size, cudaMemcpyHostToDevice); cudaMemcpy(d_bz, bz, size, cudaMemcpyHostToDevice);
				cudaMemcpy(d_cx, cx, size, cudaMemcpyHostToDevice); cudaMemcpy(d_cy, cy, size, cudaMemcpyHostToDevice); cudaMemcpy(d_cz, cz, size, cudaMemcpyHostToDevice);

				free(ax); free(ay); free(az);
				free(bx); free(by); free(bz);
				free(cx); free(cy); free(cz);

				vecSubDArr << <numTris, 1 >> > (d_bx, d_by, d_bz, d_ax, d_ay, d_az, d_BAX, d_BAY, d_BAZ);
				vecSubDArr << <numTris, 1 >> > (d_cx, d_cy, d_cz, d_bx, d_by, d_bz, d_CBX, d_CBY, d_CBZ);
				vecSubDArr << <numTris, 1 >> > (d_ax, d_ay, d_az, d_cx, d_cy, d_cz, d_ACX, d_ACY, d_ACZ);

				int rowActTriSize = sizeof(int) * numTris * width;

				cudaMalloc((void**)&d_QAX, rowActTriSize); cudaMalloc((void**)&d_QAY, rowActTriSize); cudaMalloc((void**)&d_QAZ, rowActTriSize);
				cudaMalloc((void**)&d_QBX, rowActTriSize); cudaMalloc((void**)&d_QBY, rowActTriSize); cudaMalloc((void**)&d_QBZ, rowActTriSize);
				cudaMalloc((void**)&d_QCX, rowActTriSize); cudaMalloc((void**)&d_QCY, rowActTriSize); cudaMalloc((void**)&d_QCZ, rowActTriSize);

				int dvarsize = sizeof(int);
				cudaMalloc((void**)&d_QY, dvarsize); cudaMalloc((void**)&d_QZ, dvarsize);
				cudaMemcpy(d_QY, &h, dvarsize, cudaMemcpyHostToDevice); cudaMemcpy(d_QZ, &d, dvarsize, cudaMemcpyHostToDevice);

				vecSubPoint << <width, numTris >> > (d_QY, d_QZ, d_ax, d_ay, d_az, d_QAX, d_QAY, d_QAZ);
				vecSubPoint << <width, numTris >> > (d_QY, d_QZ, d_bx, d_by, d_bz, d_QBX, d_QBY, d_QBZ);
				vecSubPoint << <width, numTris >> > (d_QY, d_QZ, d_cx, d_cy, d_cz, d_QCX, d_QCY, d_QCZ);

				cudaFree(d_QY); cudaFree(d_QZ);

				cudaFree(d_ax); cudaFree(d_ay); cudaFree(d_az);
				cudaFree(d_bx); cudaFree(d_by); cudaFree(d_bz);
				cudaFree(d_cx); cudaFree(d_cy); cudaFree(d_cz);

				cudaMalloc((void**)&d_BAQAX, rowActTriSize); cudaMalloc((void**)&d_BAQAY, rowActTriSize); cudaMalloc((void**)&d_BAQAZ, rowActTriSize);
				cudaMalloc((void**)&d_CBQBX, rowActTriSize); cudaMalloc((void**)&d_CBQBY, rowActTriSize); cudaMalloc((void**)&d_CBQBZ, rowActTriSize);
				cudaMalloc((void**)&d_ACQCX, rowActTriSize); cudaMalloc((void**)&d_ACQCY, rowActTriSize); cudaMalloc((void**)&d_ACQCZ, rowActTriSize);

				vecCross << <width, numTris >> > (d_BAX, d_BAY, d_BAZ, d_QAX, d_QAY, d_QAZ, d_BAQAX, d_BAQAY, d_BAQAZ);
				vecCross << <width, numTris >> > (d_CBX, d_CBY, d_CBZ, d_QBX, d_QBY, d_QBZ, d_CBQBX, d_CBQBY, d_CBQBZ);
				vecCross << <width, numTris >> > (d_ACX, d_ACY, d_ACZ, d_QCX, d_QCY, d_QCZ, d_ACQCX, d_ACQCY, d_ACQCZ);

				cudaFree(d_BAX); cudaFree(d_BAY); cudaFree(d_BAZ);
				cudaFree(d_CBX); cudaFree(d_CBY); cudaFree(d_CBZ);
				cudaFree(d_ACX); cudaFree(d_ACY); cudaFree(d_ACZ);

				cudaFree(d_QAX); cudaFree(d_QAY); cudaFree(d_QAZ);
				cudaFree(d_QBX); cudaFree(d_QBY); cudaFree(d_QBZ);
				cudaFree(d_QCX); cudaFree(d_QCY); cudaFree(d_QCZ);

				int bvarsize = sizeof(bool) * numTris * width;
				cudaMalloc((void**)&d_C1, bvarsize); cudaMalloc((void**)&d_C2, bvarsize); cudaMalloc((void**)&d_C3, bvarsize);

				cudaMalloc((void**)&d_NX, size); cudaMalloc((void**)&d_NY, size); cudaMalloc((void**)&d_NZ, size);
				cudaMemcpy(d_NX, NX, size, cudaMemcpyHostToDevice); cudaMemcpy(d_NY, NY, size, cudaMemcpyHostToDevice); cudaMemcpy(d_NZ, NZ, size, cudaMemcpyHostToDevice);

				normDot << <width, numTris >> > (d_BAQAX, d_BAQAY, d_BAQAZ, d_NX, d_NY, d_NZ, d_C1);
				normDot << <width, numTris >> > (d_CBQBX, d_CBQBY, d_CBQBZ, d_NX, d_NY, d_NZ, d_C2);
				normDot << <width, numTris >> > (d_ACQCX, d_ACQCY, d_ACQCZ, d_NX, d_NY, d_NZ, d_C3);

				cudaFree(d_BAQAX); cudaFree(d_BAQAY); cudaFree(d_BAQAZ);
				cudaFree(d_CBQBX); cudaFree(d_CBQBY); cudaFree(d_CBQBZ);
				cudaFree(d_ACQCX); cudaFree(d_ACQCY); cudaFree(d_ACQCZ);
				cudaFree(d_NX); cudaFree(d_NY); cudaFree(d_NZ);


				bvarsize = sizeof(bool) * width;
				cudaMalloc((void**)&d_fill, bvarsize);
				size = sizeof(int);
				cudaMalloc((void**)&d_numTris, size);
				cudaMemcpy(d_numTris, &numTris, size, cudaMemcpyHostToDevice);

				checkSum << <width, 1 >> > (d_C1, d_C2, d_C3, d_numTris, d_fill);

				cudaFree(d_C1); cudaFree(d_C2); cudaFree(d_C3); cudaFree(d_numTris);

				size = sizeof(bool) * width;
				cudaMemcpy(fills[d][h], d_fill, size, cudaMemcpyDeviceToHost);
				cudaFree(d_fill);
			}
		}
	}
}