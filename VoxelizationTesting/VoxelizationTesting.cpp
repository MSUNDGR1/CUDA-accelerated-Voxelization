// VoxelizationTesting.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include <fstream>
#include <stdlib.h>
#include "tri3D.h"
#include "vector3D.h"
#include "voxelMap.h"
#include "gif.h"

using namespace std;

//reads in STL file and converts it into component triangles.
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
            vector3D normal(facet, true);
            vector3D point1(facet + 12);
            vector3D point2(facet + 24);
            vector3D point3(facet + 36);

            out.push_back(tri3D(point1, point2, point3, normal));

        }
    }

    return;
}

//level printer for output voxelmap
void print(bool*** fills, int width, int height, int printLev) {
    system("CLS");
    bool** currLev = fills[printLev];
    string printLine;
    for (int h = 0; h < height; h++) {
        printLine = "";
        for (int w = 0; w < width; w++) {
            if (currLev[h][w] == true) printLine += "*";
            else printLine += "_";
        }
        cout << printLine << endl;
    }
}


//converts voxelmap to gif file, 1 filled voxel becomes 1 white pixel in that gif frame
void gifWrite(bool*** fills, int width, int height, int depth) {

    GifWriter out;
    int delay = 10;
    GifBegin(&out, "stampOut.gif", width, height, delay);

    for (int d = 0; d < depth; d++) {
        vector<uint8_t> currFrame;

        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                if (fills[d][h][w]) {
                    currFrame.push_back(255);
                    currFrame.push_back(255);
                    currFrame.push_back(255);
                    currFrame.push_back(255);
                }
                else {
                    currFrame.push_back(0);
                    currFrame.push_back(0);
                    currFrame.push_back(0);
                    currFrame.push_back(0);
                }
            }
        }

        GifWriteFrame(&out, currFrame.data(), width, height, delay);
        currFrame.clear();
    }
    GifEnd(&out);
}

int main()
{
    vector<tri3D> tris;
    //stl_in("encoderMount.stl", tris);
    stl_in("stampTop.stl", tris);
    voxelMap cubeMap(tris);
    int levPrint = 0;
    bool gifOut = false;
    while (levPrint != 1000) {
        cout << "enter level to print:";
        cin >> levPrint;
        if (levPrint == -1){
            gifOut = true;
            break;
        }
        print(cubeMap.fills, cubeMap.width, cubeMap.height, levPrint);
    }
    if (gifOut) {
        gifWrite(cubeMap.fills, cubeMap.width, cubeMap.height, cubeMap.depth);
    }

}
    
