Black Box Geometric Computing with Python
=======

This course is presented at [Eurographics 2020](https://conferences.eg.org/egev20/program/) as a full day tutorial. A [shorter version](https://geometryprocessing.github.io/geometric-computing-python/) of this course was presented at [Siggraph 2019](https://s2019.siggraph.org/).

In the course, we present a set of real-world examples from Geometry Processing, Physical Simulation, and Geometric Deep Learning. Each example is prototypical of a common task in research or industry and is implemented in a few lines of code. By the end of the course, attendees will have exposure to a swiss-army-knife of simple, composable, and high-performance tools for Geometric Computing.


## Course Material
<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/videoseries?list=PL0wZK75RSlnkbxgukKIgI1FiOoT8nxhqw" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

The full course is available as videos [here](https://www.youtube.com/playlist?list=PL0wZK75RSlnkbxgukKIgI1FiOoT8nxhqw). It covers the theory and practice in an hands-on approach with many examples. We also publish the *rendered* [Jupyter](https://jupyter.org) notebooks on this website and *interactive and editable* versions on [Binder](https://mybinder.org/).

The course is divided into the following parts:

- [Introduction](https://www.icloud.com/keynote/0S7N5YO_5dhSTfYacBIGRxLvA#01_-_Introduction)
- [FEM Introduction](fem-intro) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/geometryprocessing/blackbox-computing-python.git/doc?filepath=nb%2Ffem-intro.ipynb)
- [FEM Higher Order](fem-intro-high-order) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/geometryprocessing/blackbox-computing-python.git/doc?filepath=nb%2Ffem-intro-high-order.ipynb)
- [Geometry Processing](geo_viz) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/geometryprocessing/blackbox-computing-python.git/doc?filepath=nb%2Fgeo_viz.ipynb)
- [Black Box Simulation](https://www.icloud.com/keynote/0iQMvxSA-M-FJGH8QcKpcf8Mw#04_-_Black_Box)
- [CAD Deep Learning](cad_ml)
- [Meshing and Simulation](polyfem2d) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/geometryprocessing/blackbox-computing-python.git/doc?filepath=nb%2Fpolyfem2d.ipynb)
- [Ultimate Example](ultimate) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/geometryprocessing/blackbox-computing-python.git/doc?filepath=nb%2Fultimate.ipynb)
- [Conclusion](https://www.icloud.com/keynote/0LqBqqPGg0qnz9sJyZjMpDZUg#07_-_Conclusions)



## Installation
The easiest way to install the libraries used in the course is trough the [conda](https://anaconda.org/) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) python package manager.

All libraries are part of the channel [conda forge](https://conda-forge.org/), which we advise to add to your conda channels by:
```bash
conda config --add channels conda-forge
```

This step allows to install any conda forge package simply with `conda install <package>`.

To install all our packages just run:
```bash
conda install meshplot igl wildmeshing polyfempy
```
In addition, [Jupyter](https://jupyter.org) can be easily installed to run the interactive notebooks:

```bash
conda install jupyter
```

**Note**: If the problem persists, you can post an issue on the github bugtracker of each library or [here](https://geometryprocessing.github.io/blackbox-computing-python/issues).


### Packages Description

More information about the libraries used in the course can be found on their websites:

- [Meshplot](https://skoch9.github.io/meshplot/): fast 2d and 3d mesh viewer based on `pythreejs`.
- [Wildmeshing](https://wildmeshing.github.io/): robust 2d and 3d meshing package ([python documentation](https://wildmeshing.github.io/wildmeshing-notebook/))
- [igl](https://libigl.github.io/): swiss-army-knife for Geometric Processing functions ([python documentation](https://libigl.github.io/libigl-python-bindings/))
- [polyfempy](https://polyfem.github.io/): simple but powerful finite element library ([python documentation](https://polyfem.github.io/python/))
- [ABC CAD dataset](https://deep-geometry.github.io/abc-dataset/): 1 million meshed CAD models with feature descriptions


## Motivation
Many disciplines of Computer Science have access to high level libraries allowing researchers and engineers to quickly produce prototypes. For instance, in Machine Learning, one can construct complex, state-of-the-art models which run on the GPU in just a few lines of Python.

In the field of Geometric Computing, however such high-level libraries are sparse. As a result, writing prototypes for geometry processing is time consuming and difficult even for advanced users.

In this course, we present a set of easy-to-use Python packages for applications in Geometric Computing. We have designed these libraries to have a shallow learning curve, while also enabling programmers to easily accomplish a wide variety of complex tasks. Furthermore, the libraries we present share `NumPy` arrays as a common interface, making them highly composable with each-other as well as existing scientific computing packages. Finally, our libraries are blazing fast, doing most of the heavy computations in C++ with a minimal constant-overhead interface to Python.


## Contact
This course is a group endeavor by Sebastian Koch, Teseo Schneider, Francis Williams, Chengchen Li, and Daniele Panozzo. Please contact us if you have questions or comments. For troubleshooting, please post an issue on github. We are grateful to the authors of all open source C++ libraries we are using. In particular, libigl, tetwild, polyfem, pybind11, and Jupyter.
