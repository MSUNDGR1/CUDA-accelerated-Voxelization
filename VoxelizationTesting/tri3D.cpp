#include "tri3D.h"
#include <algorithm>


tri3D::tri3D() {
	t_p1 = vector3D();
	t_p2 = vector3D();
	t_p3 = vector3D();
	flat = false;
	BA = vector3D();
	CB = vector3D();
	AC = vector3D();
}

tri3D::tri3D(vector3D p1, vector3D p2, vector3D p3, vector3D normal) {
	t_p1 = p1;
	t_p2 = p2;
	t_p3 = p3;
	int z1 = p1.m_z;
	int z2 = p2.m_z;
	int z3 = p3.m_z;
	if (z1 == z2 && z1 == z3) {
		flat = true;
		minZ = z1;
		maxZ = z1;
	}
	else {
		flat = false;
		int min1 = std::min(z1, z2);
		minZ = std::min(min1, z3);
		int max1 = std::max(z1, z2);
		maxZ = std::max(max1, z3);
	}
	int y1 = p1.m_y;
	int y2 = p2.m_y;
	int y3 = p3.m_y;
	if (y1 == y2 && y1 == y3) {
		flat = true;
		minY = y1;
		maxY = y1;
	}
	else {
		flat = false;
		int min1 = std::min(y1, y2);
		minY = std::min(min1, y3);
		int max1 = std::max(y1, y2);
		maxY = std::max(max1, y3);
	}
	int bax = t_p2.m_x - t_p1.m_x;
	int bay = t_p2.m_y - t_p1.m_y;
	int baz = t_p2.m_z - t_p1.m_z;
	BA = vector3D(bax, bay, baz);
	int cbx = t_p3.m_x - t_p2.m_x;
	int cby = t_p3.m_y - t_p2.m_y;
	int cbz = t_p3.m_z - t_p2.m_z;
	CB = vector3D(cbx, cby, cbz);
	int acx = t_p1.m_x - t_p3.m_x;
	int acy = t_p1.m_y - t_p3.m_y;
	int acz = t_p1.m_z - t_p3.m_z;
	AC = vector3D(acx, acy, acz);
	N = normal;
}

tri3D::~tri3D() {

}

void tri3D::cross(vector3D first, vector3D second, vector3D output) {
	int x = (first.m_y * second.m_z) - (first.m_z * second.m_y);
	int y = (first.m_z * second.m_x) - (first.m_x * second.m_z);
	int z = (first.m_x * second.m_y) - (first.m_y * second.m_x);
	output.m_x = x;
	output.m_y = y;
	output.m_z = z;
}

int tri3D::dot(vector3D first, vector3D second) {
	int x = first.m_x * second.m_x;
	int y = first.m_y * second.m_y;
	int z = first.m_z * second.m_z;
	int output = x + y + z;
	return output;
}

bool tri3D::coordIntersects(int x, int y, int z) {
	int qax = x - t_p1.m_x;
	int qay = y - t_p1.m_y;
	int qaz = z - t_p1.m_z;
	vector3D QA(qax, qay, qaz);

	int qbx = x - t_p2.m_x;
	int qby = y - t_p2.m_y;
	int qbz = z - t_p2.m_z;
	vector3D QB(qbx, qby, qbz);

	int qcx = x - t_p3.m_x;
	int qcy = y - t_p3.m_y;
	int qcz = z - t_p3.m_z;
	vector3D QC(qcx, qcy, qcz);

	vector3D vecCheck1;
	vector3D vecCheck2;
	vector3D vecCheck3;

	bool check1, check2, check3;

	cross(BA, QA, vecCheck1);
	cross(CB, QB, vecCheck2);
	cross(AC, QC, vecCheck3);
	check1 = dot(vecCheck1, N) >= 0;
	check2 = dot(vecCheck2, N) >= 0;
	check3 = dot(vecCheck3, N) >= 0;

	return (check1 && check2 && check3);

}