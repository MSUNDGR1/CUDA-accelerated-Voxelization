#include "voxelizer.cuh"
#include <string>
#include <iostream>

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

	//if (blockIdx.x == 0 && threadIdx.x == 0) printf("x: %d tri: %d QVoutX: %d QVoutY: %d QVoutZ: %d  \n", blockIdx.x, threadIdx.x, outX[index], outY[index], outZ[index]);
}

__global__ void vecSubPointOri(int* PY, int* PZ,
	int* VX, int* VY, int* VZ,
	int* outX, int* outY, int* outZ) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	outX[index] = VX[threadIdx.x] - blockIdx.x;
	outY[index] = VY[threadIdx.x] - *PY;
	outZ[index] = VZ[threadIdx.x] - *PZ;
}

__global__ void vecCross(int* firX, int* firY, int* firZ,
	int* secX, int* secY, int* secZ,
	int* outX, int* outY, int* outZ) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	outX[index] = (firY[threadIdx.x] * secZ[index]) - (firZ[threadIdx.x] * secY[index]);
	outY[index] = (firZ[threadIdx.x] * secX[index]) - (firX[threadIdx.x] * secZ[index]);
	outZ[index] = (firX[threadIdx.x] * secY[index]) - (firY[threadIdx.x] * secX[index]);
	//if (blockIdx.x == 0 && threadIdx.x == 0) printf("firX: %d firY: %d firZ: %d outX: %d outY: %d outZ: %d \n", firX[threadIdx.x], firY[threadIdx.x], firZ[threadIdx.x], outX[index], outY[index], outZ[index]);
}


__global__ void normDotDouble(int* firX, int* firY, int*firZ,
	int* secX, int* secY, int* secZ, int* out){
	int sum = firX[blockIdx.x] * secX[blockIdx.x];
	sum += (firY[blockIdx.x] * secY[blockIdx.x]);
	sum += (firZ[blockIdx.x] * secZ[blockIdx.x]);
	out[blockIdx.x] = sum;
}

__global__ void normDot(int* firX, int* firY, int* firZ,
	int* secX, int* secY, int* secZ,
	bool* check) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int sum = firX[index] * secX[threadIdx.x];
	sum += firY[index] * secY[threadIdx.x];
	sum += firZ[index] * secZ[threadIdx.x];
	if (sum >= 0) check[index] = true;
	
		//if (blockIdx.x == 0) printf("x: %d tri: %d crossX: %d crossY: %d crossZ: %d \n", blockIdx.x, threadIdx.x, firX[index], firY[index], firZ[index]);
		//if (blockIdx.x == 0 && threadIdx.x == 0) printf("x: %d tri: %d normX: %d normY: %d normZ: %d \n", blockIdx.x, threadIdx.x, secX[threadIdx.x], secY[threadIdx.x], secZ[threadIdx.x]);
		if(blockIdx.x == 0 && threadIdx.x == 0) printf("x: %d tri: %d dot prod: %d \n", blockIdx.x, threadIdx.x, sum);
	
}

__global__ void DCALC(int* uu, int* uv, int* vv, int* D){
	D[blockIdx.x] = (uv[blockIdx.x] * uv[blockIdx.x]) - (uu[blockIdx.x] * vv[blockIdx.x]);
}

__global__ void normDotW(int* wx, int* wy, int* wz, int* vecx, int* vecy, int* vecz, int* out){
	int indexW = blockIdx.x * blockDim.x + threadIdx.x;
	int sum = wx[indexW] * vecx[threadIdx.x];
	sum += (wy[indexW] * vecy[threadIdx.x]);
	sum += (wz[indexW] * vecz[threadIdx.x]);
	out[indexW] = sum;
}

__global__ void paramTest(int* uu, int* uv, int* vv, int* wu, int* wv, int* D, bool* intersects) {
	float s, t;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	s = (float)(((uv[threadIdx.x] * wv[index]) - (vv[threadIdx.x] * wu[index]))) / D[threadIdx.x];
	t = (float)(((uv[threadIdx.x] * wu[index]) - (uu[threadIdx.x] * wv[index]))) / D[threadIdx.x];
	if (!(s < 0.0 || s > 1.0) && !(t < 0.0 || (s + t) > 1.0)) intersects[index] = true;

	printf("D: %d X: %d s: %f t: %f\n",D[threadIdx.x], blockIdx.x, s, t);

}

__global__ void checkSum(bool* c1, bool* c2, bool* c3,
	int* actTris, bool* fills) {
	bool checker = false;
	int index;
	int offset = blockIdx.x * (*actTris);
	for (int i = 0; i < (*actTris); i++) {
		index = offset + i;
		checker = (c1[index] && c2[index]) && c3[index];
		if (checker) fills[blockIdx.x] = true;
	}
}


__global__ void angleFind(int* firX, int* firY, int* firZ,
	int* secX, int* secY, int* secZ, float* out) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	float top = firX[index] * secX[index];
	top += firY[index] * secY[index];
	top += firZ[index] * secZ[index];

	float normFir = sqrtf((float)(firX[index] * firX[index] + firY[index] * firY[index] + firZ[index] * firZ[index]));
	float normSec = sqrtf((float)(secX[index] * secX[index] + secY[index] * secY[index] + secZ[index] * secZ[index]));

	float input = top / (normFir * normSec);
	out[index] = acosf(input);
}


__global__ void angleSum(float* ang1, float* ang2, float* ang3, bool* planeInt,
	bool* intersect) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = abs(ang1[index]) + abs(ang2[index]) + abs(ang3[index]);
	//printf("X: %d sum: %f \n", blockIdx.x, sum);
	if (abs(sum - 6.28) < 0.02 && planeInt[index]) intersect[index] = true;
}

__global__ void intersectCount(int* numTris, bool* intersects, bool* outIntersect) {
	int indexOffset = blockIdx.x * (*numTris);
	bool out = false;
	int intersectCount = 0;
	for (int i = 0; i < *numTris; i++) {
		
		if (intersects[indexOffset + i]) { out = true; intersectCount++; }
	}
	outIntersect[blockIdx.x] = out;
	printf("X: %d Intersections: %d \n", blockIdx.x, intersectCount);
}

__global__ void planeIntersect(int* A, int* B, int* C, int* D, int* inY, int* inZ, bool* out) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int sum = blockIdx.x * A[threadIdx.x];
	sum += (*inY) * B[threadIdx.x];
	sum += (*inZ) * C[threadIdx.x];
	if (abs(sum - D[threadIdx.x]) < 0.1) out[index] = true;
	if ((*inY) == 9 && (*inZ) == 9) {
		printf("Norm: A: %d B: %d C:%d D: %d X: %d \n", A[threadIdx.x], B[threadIdx.x], C[threadIdx.x], D[threadIdx.x], blockIdx.x);
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
		checkY = (minYTris[i] <= y && maxYTris[i] >= y);
		checkZ = (minZTris[i] <= z && maxZTris[i] >= z);
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
				//if (h == 9 && d == 9) {
					activeTris.clear();
					actTriFind(minZTris, maxZTris, minYTris, maxYTris, activeTris, h, d);
					if (activeTris.size() == 0) {
						for (int w = 0; w < width; w++) {
							fills[d][h][w] = false;
						}
					}
					else {
						printf("Level: %d height: %d Active tris: %d\n", d, h, activeTris.size());
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
							std::vector<int> norm = norms[activeTris[i]];
							NX[i] = norm[0]; NY[i] = norm[1]; NZ[i] = norm[2];
							printf("Tri: %d NormX: %d NormY: %d NormZ: %d\n", i, NX[i], NY[i], NZ[i]);
							if (i == 0) {
								printf("Tri: %d ax: %d ay: %d az: %d \n", i, ax[i], ay[i], az[i]);
								printf("Tri: %d bx: %d by: %d bz: %d \n", i, bx[i], by[i], bz[i]);
								printf("Tri: %d cx: %d cy: %d cz: %d \n", i, cx[i], cy[i], cz[i]);
							}
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

						cudaDeviceSynchronize();

						int rowActTriSize = sizeof(int) * numTris * width;

						cudaMalloc((void**)&d_QAX, rowActTriSize); cudaMalloc((void**)&d_QAY, rowActTriSize); cudaMalloc((void**)&d_QAZ, rowActTriSize);
						cudaMalloc((void**)&d_QBX, rowActTriSize); cudaMalloc((void**)&d_QBY, rowActTriSize); cudaMalloc((void**)&d_QBZ, rowActTriSize);
						cudaMalloc((void**)&d_QCX, rowActTriSize); cudaMalloc((void**)&d_QCY, rowActTriSize); cudaMalloc((void**)&d_QCZ, rowActTriSize);

						int dvarsize = sizeof(int);
						cudaMalloc((void**)&d_QY, dvarsize); cudaMalloc((void**)&d_QZ, dvarsize);
						cudaMemcpy(d_QY, &h, dvarsize, cudaMemcpyHostToDevice); cudaMemcpy(d_QZ, &d, dvarsize, cudaMemcpyHostToDevice);

						vecSubPoint << <width, numTris >> > (d_QY, d_QZ, d_ax, d_ay, d_az, d_QAX, d_QAY, d_QAZ);
						cudaDeviceSynchronize();

						vecSubPoint << <width, numTris >> > (d_QY, d_QZ, d_bx, d_by, d_bz, d_QBX, d_QBY, d_QBZ);
						cudaDeviceSynchronize();
						vecSubPoint << <width, numTris >> > (d_QY, d_QZ, d_cx, d_cy, d_cz, d_QCX, d_QCY, d_QCZ);

						cudaDeviceSynchronize();

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

						cudaDeviceSynchronize();

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

						cudaDeviceSynchronize();

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

						cudaDeviceSynchronize();

						cudaFree(d_C1); cudaFree(d_C2); cudaFree(d_C3); cudaFree(d_numTris);

						size = sizeof(bool) * width;
						cudaMemcpy(fills[d][h], d_fill, size, cudaMemcpyDeviceToHost);
						cudaFree(d_fill);
					}
				/*}
				else {
				for (int i = 0; i < width; i++) {
					fills[d][h][i] = false;
					}
				}*/
			}
		}
	}

	void voxelizeAngle(const std::vector<std::vector<int>> triVecs,
		const std::vector<std::vector<int>> norms,
		const int width, const int height, const int depth,
		const std::vector<int> minZTris,
		const std::vector<int> maxZTris,
		const std::vector<int> minYTris,
		const std::vector<int> maxYTris,
		bool*** fills) {


		int numTris = minZTris.size();
		int size;
		int * ax, * ay, * az,
			* bx, * by, * bz,
			* cx, * cy, * cz;
		int* NX, * NY, * NZ;
		int * d_ax, * d_ay, * d_az,
			* d_bx, * d_by, * d_bz,
			* d_cx, * d_cy, * d_cz;

		int * d_PAX, * d_PAY, * d_PAZ,
			* d_PBX, * d_PBY, * d_PBZ,
			* d_PCX, * d_PCY, * d_PCZ;

		float* d_AB, * d_BC, * d_CA;
		
		int* PLD; bool* d_plInt;
		int* d_PLA, * d_PLB, * d_PLC, * d_PLD;

		bool* d_intersects;
		bool* d_out;
		std::vector<int> activeTris;
		for (int d = 0; d < depth; d++) {
			for (int h = 0; h < height; h++) {
				//if (d == 9) {
					activeTris.clear();
					actTriFind(minZTris, maxZTris, minYTris, maxYTris, activeTris, h, d);
					int numTris = activeTris.size();
					size = sizeof(int) * numTris;
					ax = (int*)malloc(size); ay = (int*)malloc(size); az = (int*)malloc(size);
					bx = (int*)malloc(size); by = (int*)malloc(size); bz = (int*)malloc(size);
					cx = (int*)malloc(size); cy = (int*)malloc(size); cz = (int*)malloc(size);
					NX = (int*)malloc(size); NY = (int*)malloc(size); NZ = (int*)malloc(size);

					PLD = (int*)malloc(size);
					
					for (int i = 0; i < numTris; i++) {

						std::vector<int> actVecA = triVecs[(activeTris[i] * 3)];
						std::vector<int> actVecB = triVecs[(activeTris[i] * 3) + 1];
						std::vector<int> actVecC = triVecs[(activeTris[i] * 3) + 2];
						ax[i] = actVecA[0]; ay[i] = actVecA[1]; az[i] = actVecA[2];
						bx[i] = actVecB[0]; by[i] = actVecB[1]; bz[i] = actVecB[2];
						cx[i] = actVecC[0]; cy[i] = actVecC[1]; cz[i] = actVecC[2];
						std::vector<int> norm = norms[activeTris[i]];
						NX[i] = norm[0]; NY[i] = norm[1]; NZ[i] = norm[2];
						PLD[i] = norm[0] * ax[i] + norm[1] * ay[i] + norm[2] * az[i];
						norm.clear();
					}

					cudaMalloc((void**)&d_ax, size); cudaMalloc((void**)&d_ay, size); cudaMalloc((void**)&d_az, size);
					cudaMalloc((void**)&d_bx, size); cudaMalloc((void**)&d_by, size); cudaMalloc((void**)&d_bz, size);
					cudaMalloc((void**)&d_cx, size); cudaMalloc((void**)&d_cy, size); cudaMalloc((void**)&d_cz, size);

					cudaMemcpy(d_ax, ax, size, cudaMemcpyHostToDevice); cudaMemcpy(d_ay, ay, size, cudaMemcpyHostToDevice); cudaMemcpy(d_az, az, size, cudaMemcpyHostToDevice);
					cudaMemcpy(d_bx, bx, size, cudaMemcpyHostToDevice); cudaMemcpy(d_by, by, size, cudaMemcpyHostToDevice); cudaMemcpy(d_bz, bz, size, cudaMemcpyHostToDevice);
					cudaMemcpy(d_cx, cx, size, cudaMemcpyHostToDevice); cudaMemcpy(d_cy, cy, size, cudaMemcpyHostToDevice); cudaMemcpy(d_cz, cz, size, cudaMemcpyHostToDevice);

					free(ax); free(ay); free(az);
					free(bx); free(by); free(bz);
					free(cx); free(cy); free(cz);

					cudaMalloc((void**)&d_PLA, size); cudaMalloc((void**)&d_PLB, size); cudaMalloc((void**)&d_PLC, size); cudaMalloc((void**)&d_PLD, size); 
					cudaMemcpy(d_PLA, NX, size, cudaMemcpyHostToDevice); cudaMemcpy(d_PLB, NY, size, cudaMemcpyHostToDevice); 
					cudaMemcpy(d_PLC, NZ, size, cudaMemcpyHostToDevice); cudaMemcpy(d_PLD, PLD, size, cudaMemcpyHostToDevice);

					size = sizeof(bool) * width * numTris;
					cudaMalloc((void**)&d_plInt, size);

					int* d_PY, * d_PZ;
					size = sizeof(int);
					cudaMalloc((void**)&d_PY, size); cudaMalloc((void**)&d_PZ, size);
					cudaMemcpy(d_PY, &h, size, cudaMemcpyHostToDevice);
					cudaMemcpy(d_PZ, &d, size, cudaMemcpyHostToDevice);

					planeIntersect << <width, numTris >> > (d_PLA, d_PLB, d_PLC, d_PLD, d_PY, d_PZ, d_plInt);

					cudaDeviceSynchronize();
					cudaFree(d_PLA); cudaFree(d_PLB); cudaFree(d_PLC); cudaFree(d_PLD);

					size = sizeof(int) * numTris * width;
					cudaMalloc((void**)&d_PAX, size); cudaMalloc((void**)&d_PAY, size); cudaMalloc((void**)&d_PAZ, size);
					cudaMalloc((void**)&d_PBX, size); cudaMalloc((void**)&d_PBY, size); cudaMalloc((void**)&d_PBZ, size);
					cudaMalloc((void**)&d_PCX, size); cudaMalloc((void**)&d_PCY, size); cudaMalloc((void**)&d_PCZ, size);

					vecSubPointOri << <width, numTris >> > (d_PY, d_PZ, d_ax, d_ay, d_az, d_PAX, d_PAY, d_PAZ);
					vecSubPointOri << <width, numTris >> > (d_PY, d_PZ, d_bx, d_by, d_bz, d_PBX, d_PBY, d_PBZ);
					vecSubPointOri << <width, numTris >> > (d_PY, d_PZ, d_cx, d_cy, d_cz, d_PCX, d_PCY, d_PCZ);

					cudaDeviceSynchronize();

					cudaFree(d_PY); cudaFree(d_PZ);

					cudaFree(d_ax); cudaFree(d_ay); cudaFree(d_az);
					cudaFree(d_bx); cudaFree(d_by); cudaFree(d_bz);
					cudaFree(d_cx); cudaFree(d_cy); cudaFree(d_cz);

					size = sizeof(float) * numTris * width;
					cudaMalloc((void**)&d_AB, size);
					cudaMalloc((void**)&d_BC, size);
					cudaMalloc((void**)&d_CA, size);

					angleFind << <width, numTris >> > (d_PAX, d_PAY, d_PAZ, d_PBX, d_PBY, d_PBZ, d_AB);
					angleFind << <width, numTris >> > (d_PBX, d_PBY, d_PBZ, d_PCX, d_PCY, d_PCZ, d_BC);
					angleFind << <width, numTris >> > (d_PCX, d_PCY, d_PCZ, d_PAX, d_PAY, d_PAZ, d_CA);

					cudaDeviceSynchronize();

					cudaFree(d_PAX); cudaFree(d_PAY); cudaFree(d_PAZ);
					cudaFree(d_PBX); cudaFree(d_PBY); cudaFree(d_PBZ);
					cudaFree(d_PCX); cudaFree(d_PCY); cudaFree(d_PCZ);

					size = sizeof(bool) * width * numTris;
					cudaMalloc((void**)&d_intersects, size);
					//printf("height: %d\n", h);
					angleSum << <width, numTris >> > (d_AB, d_BC, d_CA, d_plInt, d_intersects);

					cudaDeviceSynchronize();

					cudaFree(d_AB); cudaFree(d_BC); cudaFree(d_CA);

					int* d_numTris;
					size = sizeof(int);
					cudaMalloc((void**)&d_numTris, size);
					cudaMemcpy(d_numTris, &numTris, size, cudaMemcpyHostToDevice);

					size = sizeof(bool) * width;
					cudaMalloc((void**)&d_out, size);

					intersectCount << <width, 1 >> > (d_numTris, d_intersects, d_out);
					cudaFree(d_numTris); cudaFree(d_intersects);

					cudaMemcpy(fills[d][h], d_out, size, cudaMemcpyDeviceToHost);
				//}
			}
		}
	}

	void rayVoxel(const std::vector<std::vector<int>> triVecs,
		const int width, const int height, const int depth,
		const std::vector<int> minZTris,
		const std::vector<int> maxZTris,
		const std::vector<int> minYTris,
		const std::vector<int> maxYTris,
		bool*** fills) {
		int size;
		int* ax, * ay, * az;
		int* d_ax, * d_ay, * d_az;

		int* NX, * NY, * NZ;
		int* d_NX, * d_NY, * d_NZ;

		int* ux, * uy, * uz,
			* vx, * vy, * vz;

		int* d_ux, * d_uy, * d_uz,
			* d_vx, * d_vy, * d_vz;

		int* d_wx, * d_wy, * d_wz;

		int* d_uu, * d_uv, * d_vv,
			* d_wu, * d_wv, * d_D;

		bool* d_intersects;
		bool* d_out;
		std::vector<int> activeTris;
		for (int d = 0; d < depth; d++) {
			for (int h = 0; h < height; h++) {
				if (d == 10) {
					if (h == 5) {
						activeTris.clear();
						actTriFind(minZTris, maxZTris, minYTris, maxYTris, activeTris, h, d);
						int numTris = activeTris.size();
						size = sizeof(int) * numTris;
						ax = (int*)malloc(size); ay = (int*)malloc(size); az = (int*)malloc(size);


						ux = (int*)malloc(size); uy = (int*)malloc(size); uz = (int*)malloc(size);
						vx = (int*)malloc(size); vy = (int*)malloc(size); vz = (int*)malloc(size);
						NX = (int*)malloc(size); NY = (int*)malloc(size); NZ = (int*)malloc(size);


						for (int i = 0; i < numTris; i++) {

							std::vector<int> actVecA = triVecs[(activeTris[i] * 3)];
							std::vector<int> actVecB = triVecs[(activeTris[i] * 3) + 1];
							std::vector<int> actVecC = triVecs[(activeTris[i] * 3) + 2];
							ax[i] = actVecA[0]; ay[i] = actVecA[1]; az[i] = actVecA[2];


							ux[i] = actVecB[0] - ax[i]; uy[i] = actVecB[1] - ay[i]; uz[i] = actVecB[2] - az[i];
							vx[i] = actVecC[0] - ax[i]; vy[i] = actVecC[1]; -ay[i]; vz[i] = actVecC[2] - az[i];
							NX[i] = (uy[i] * vz[i]) - (uz[i] * vy[i]);
							NY[i] = (uz[i] * vx[i]) - (ux[i] * vz[i]);
							NZ[i] = (ux[i] * vy[i]) - (uy[i] * vx[i]);
						}

						int* d_Z, * d_Y;
						size = sizeof(int); cudaMalloc((void**)&d_Z, size); cudaMalloc((void**)&d_Y, size);
						cudaMemcpy(d_Z, &d, size, cudaMemcpyHostToDevice); cudaMemcpy(d_Y, &h, size, cudaMemcpyHostToDevice);
						size = sizeof(int) * numTris * width; cudaMalloc((void**)&d_wx, size); cudaMalloc((void**)&d_wy, size); cudaMalloc((void**)&d_wz, size);
						size = sizeof(int) * numTris; cudaMalloc((void**)&d_ux, size); cudaMalloc((void**)&d_uy, size); cudaMalloc((void**)&d_uz, size);
						cudaMalloc((void**)&d_vx, size); cudaMalloc((void**)&d_vy, size); cudaMalloc((void**)&d_vz, size);
						cudaMemcpy(d_ux, ux, size, cudaMemcpyHostToDevice); cudaMemcpy(d_uy, uy, size, cudaMemcpyHostToDevice); cudaMemcpy(d_uz, uz, size, cudaMemcpyHostToDevice);
						cudaMemcpy(d_vx, vx, size, cudaMemcpyHostToDevice); cudaMemcpy(d_vy, vy, size, cudaMemcpyHostToDevice); cudaMemcpy(d_vz, vz, size, cudaMemcpyHostToDevice);
						cudaMalloc((void**)&d_ax, size); cudaMalloc((void**)&d_ay, size); cudaMalloc((void**)&d_az, size);
						cudaMemcpy(d_ax, ax, size, cudaMemcpyHostToDevice); cudaMemcpy(d_ay, ay, size, cudaMemcpyHostToDevice); cudaMemcpy(d_az, az, size, cudaMemcpyHostToDevice);
						free(ax); free(ay); free(az);
						vecSubPoint << <width, numTris >> > (d_Y, d_Z, d_ax, d_ay, d_az, d_wx, d_wy, d_wz);

						cudaFree(d_ax); cudaFree(d_ay); cudaFree(d_az);
						cudaFree(d_Y); cudaFree(d_Z);

						cudaMalloc((void**)&d_uu, size); cudaMalloc((void**)&d_uv, size); cudaMalloc((void**)&d_vv, size);
						normDotDouble << <numTris, 1 >> > (d_ux, d_uy, d_uz, d_ux, d_uy, d_uz, d_uu);
						normDotDouble << <numTris, 1 >> > (d_ux, d_uy, d_uz, d_vx, d_vy, d_vz, d_uv);
						normDotDouble << <numTris, 1 >> > (d_vx, d_vy, d_vz, d_vx, d_vy, d_vz, d_vv);

						cudaMalloc((void**)&d_D, size);
						DCALC << <numTris, 1 >> > (d_uu, d_uv, d_vv, d_D);

						size = sizeof(int) * numTris * width;
						cudaMalloc((void**)&d_wu, size); cudaMalloc((void**)&d_wv, size);
						normDotW << <width, numTris >> > (d_wx, d_wy, d_wz, d_ux, d_uy, d_uz, d_wu);
						normDotW << <width, numTris >> > (d_wx, d_wy, d_wz, d_vx, d_vy, d_vz, d_wv);

						cudaFree(d_wx); cudaFree(d_wy); cudaFree(d_wz);
						cudaFree(d_ux); cudaFree(d_uy); cudaFree(d_uz);
						cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);

						size = sizeof(bool) * numTris * width;
						cudaMalloc((void**)&d_intersects, size);

						paramTest << <width, numTris >> > (d_uu, d_uv, d_vv, d_wu, d_wv, d_D, d_intersects);

						cudaFree(d_uu); cudaFree(d_uv); cudaFree(d_vv);
						cudaFree(d_wu); cudaFree(d_wv); cudaFree(d_D);

						size = sizeof(bool) * width; cudaMalloc((void**)&d_out, size);

						size = sizeof(int); int* d_numTris; cudaMalloc((void**)&d_numTris, size); cudaMemcpy(d_numTris, &numTris, size, cudaMemcpyHostToDevice);

						intersectCount << <width, 1 >> > (d_numTris, d_intersects, d_out);
						cudaFree(d_numTris); cudaFree(d_intersects);
						cudaMemcpy(fills[d][h], d_out, size, cudaMemcpyDeviceToHost);
						cudaFree(d_out);
					}
				}
			}
		}
	}
}