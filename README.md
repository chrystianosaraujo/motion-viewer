# Motion Viewer
This project is co-authored with https://github.com/sparkon
## Overview
Motion Viewer is an open source implementation of the [2002 SIGGRAPH paper "Motion Graphs"](https://dl.acm.org/citation.cfm?id=566605). 
It features:  

- Motion Capture data (.BVH) importer and visualizer  
- Motion Graph construction from imported MoCap data  
- Motion Graph interactive visualization
- Motion Graph walking (currently fairly naive)

The codebase here is still in an unfinished state, in particular the following features are temporarily missing:

- Interface callbacks: Although most of the interface is there, part of it has still to be wired to the main code.
- Interactive control: As suggested towards the end of the paper, we will add the possibility of interactively controlling the character by precomputing the best paths for each sampled direction and storing them in each node.
- Code quality and performance improvements

## How to run
`python motion_viewer_app.py`  
Will launch the main application, which will start constructing a motion graph from the input files specified in `motion_viewer_app.py` `MotionViewerApp::run()`. 
Importing individual motions can be done from File > Import .BVH and then play/stop to control the playback.
**Note**: After the Motion Graph is constructed, a `cache.p` file will be created. This file will contain the cached Motion Graph, delete it if you want to reconstruct it in subsequent runs

## Requirements
The code supports Python 3.5+ and depends on some libraries, all of which available from the Python Package Index (pip) with the following names:
```
numpy
pyglm
pyquaternion
pyqt5
pyopengl
graphviz (Will be made optional)
```
