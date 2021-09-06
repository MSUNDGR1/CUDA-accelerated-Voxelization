# CUDA-accelerated-Voxelization

**Voxelization method that takes a triangle mesh (in this case from an stl file), and converts it into a 3d grid of voxels.**

<p float="center">
  <img src="https://i.imgur.com/fpVrY0p.gif" width="376" />
  <img src="rocketnozzlePic.PNG" width="300" />
  <br>
    <em>A converging-diverging nozzle I designed to burn kerosene and high-test peroxide </em>
 </p>

CUDA acceleration is used to perform all the annoying trigonometry. When an stl file is voxelized, we consider individual lines across the 3d grid at a time. *This is actually unoptimal, since most GPUs have enough power to consider entire layers at a time, however I designed this with large STL files in mind, on the order of meters in a dimension. I wrote the majority of this when I had very little CS experience, so sue me.* The triangles that are known to intersect that line are then found, and then each point along that line is tested to see if it intersects with any of these triangles.

<p float="center">
  <img src="stampOut.gif" width="150" />
  <img src="presstopPic.PNG" width="200" />
  <br>
    <em>The rotating stamp head for a rotary press intended to form primer cups out of brass sheet.</em>
</p>
  
I do a lot of CAD work with fusion360, and I wanted to do a project that would allow for cool visualizations and simulations with models I make, as well as an excuse to dip my feet into CUDA programming. My only parallelization experience before this was with POSIX threads in C, and that was minimal at best. The original motivation was to create a voxelization tool that would allow for interesting simulations, particularly of neutron interactions with certain metals, to be run much more easily. Voxels are actually quite useful for this purpose as they are a particularly easy base unit to deal with. I had no idea what I was getting into, and the precursor project became a main project.

<p float="center">
  <img src="encoderOut.gif" width="250" />
  <img src="encodermountPic.PNG" width="250" />
  <br>
    <em>The mount for holding the rotary encoder of my pendulum balancing project.</em>
</p>
