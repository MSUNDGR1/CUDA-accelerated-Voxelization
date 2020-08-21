#include "vector3D.h"

vector3D::vector3D() {
	m_x = 0;
	m_y = 0;
	m_z = 0;
}

vector3D::vector3D(double x, double y, double z) {
	m_x = (int)x;
	m_y = (int)y;
	m_z = (int)z;
}

vector3D::vector3D(char* binary) {
	char xC[4] = { binary[0], binary[1], binary[2], binary[3] };
	char yC[4] = { binary[4], binary[5], binary[6], binary[7] };
	char zC[4] = { binary[8], binary[9], binary[10], binary[11] };

	m_x = (int)(double(*((float*)xC)));
	m_y = (int)(double(*((float*)yC)));
	m_z = (int)(double(*((float*)zC)));
}

vector3D::vector3D(int x, int y, int z) {
	m_x = x;
	m_y = y;
	m_z = z;
}

vector3D::~vector3D() {

}