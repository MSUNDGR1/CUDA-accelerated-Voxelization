#pragma once
#include "vector3D.h"

class tri3D
{
public:

	tri3D();
	tri3D(vector3D p1, vector3D p2, vector3D p3, vector3D normal);
	~tri3D();

	int minZ;
	int maxZ;
	int minY;
	int maxY;
	vector3D t_p1, t_p2, t_p3;
	bool flat;
	vector3D N;
	bool coordIntersects(int x, int y, int z);
private:
	void cross(vector3D first, vector3D second, vector3D output);
	int dot(vector3D first, vector3D second);
	vector3D BA, AC, CB;

};

