# CUDA-accelerated-Voxelization
Voxelization method that takes a triangle mesh (in this case from an stl file), and converts it into a 3d grid of voxels.

![nozzle gif](https://i.imgur.com/fpVrY0p.gif) 
![nozzle pic](rocketnozzlePic.png)

CUDA acceleration is used to perform all the annoying trigonometry. When an stl file is voxelized, we consider individual lines across the 3d grid at a time. This is actually unoptimal, since most GPUs have enough power to consider entire layers at a time, however I designed this with large STL files in mind, on the order of meters in a dimension. I wrote the majority of this when I had very little CS experience, so sue me. The triangles that are known to intersect that line are then found, and then each point along that line is tested to see if it intersects with any of these triangles.

![press gif](stampOut.gif) 
![press pic](presstopPic.png)

I do a lot of CAD work with fusion360, and I wanted to do a project that would allow for cool visualizations of some of the models I've made, as well as an excuse to dip my feet into CUDA programming.


