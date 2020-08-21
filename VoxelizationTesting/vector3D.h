#pragma once
class vector3D
{
public:
	vector3D();
	vector3D(char* binary);
	vector3D(double x, double y, double z);
	vector3D(int x, int y, int z);
	~vector3D();

	int m_x, m_y, m_z;
};

