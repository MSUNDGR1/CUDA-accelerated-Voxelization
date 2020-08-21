// VoxelizationTesting.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include <fstream>

#include "tri3D.h"
#include "vector3D.h"
#include "voxelMap.h"


using namespace std;

void stl_in(string fname, vector<tri3D>& out) {

    //!!
    //don't forget ios::binary
    //!!
    ifstream myFile(fname.c_str(), ios::in | ios::binary);

    char header_info[80] = "";
    char nTri[4];
    unsigned long nTriLong;

    //read 80 byte header
    if (myFile) {
        myFile.read(header_info, 80);
        cout << "header: " << header_info << endl;
    }
    else {
        cout << "error" << endl;
    }

    //read 4-byte ulong
    if (myFile) {
        myFile.read(nTri, 4);
        nTriLong = *((unsigned long*)nTri);
        cout << "n Tri: " << nTriLong << endl;
    }
    else {
        cout << "error" << endl;
    }

    for (unsigned int i = 0; i < nTriLong; i++) {
        char facet[50];

        if (myFile) {
            myFile.read(facet, 50);
            vector3D normal(facet);
            vector3D point1(facet + 12);
            vector3D point2(facet + 24);
            vector3D point3(facet + 36);

            out.push_back(tri3D(point1, point2, point3, normal));

        }
    }

    return;
}


int main()
{
    vector<tri3D> tris;
    stl_in("Cube.stl", tris);
    voxelMap weenor(tris);


}
    
