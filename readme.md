# 3D ShapeNets for 2.5D Object Recognition and Next-Best-View Prediction
------------------------------------------------------------------------

## Overview

- The code implements Convolution Deep Belief Network in 3D and apply it for depth image object recognition, NBV selection.
- To run the code, a CUDA supported GPU should be installed.
- For some unknown reason, in order to use OpenGL rendering, not all Matlab releases are working. We have only tested R2013a.
- A model after training is save as ShapeNet.mat.

## Code

To train the model, simply run the main.m file. Training involves a pretraining phase and a fine-tuning phase. 
The architecture parameters and training parameters could both be set in the run_pretraining.m and run_finetuning.m file.

Here is how the code is organized:

	1. The . folder primarily contains all the code for training, and some for testing.
	2. The 3D folder involves 3D computations like TSDF and Rendering.
	3. The voxelization folder is a toolbox to convert the mesh model to a volume representation.
	4. The util folder provides some functions for visualization and preparing the data. 

After training, the model could be powerful to do various tasks:

	1. Extract feature of a 3D mesh model. (feature could be used to train a discriminative classifier afterward).
	2. Classify a 3D mesh model.
	3. Given a surface a 3D mesh model, infer/complete the invisible part (missing part) and classify it at the same time.
	4. Compute the recognition uncertainties for the current view and decide the Next-Best-View to move the camera.

## Data

- All of the data is in the data folder. For each category, the off subfolder contains the original 3D mesh data.
- If the resolution is low(30), one can convert all offs to volumetric representations and load it in memory. However, if the resolution is high(50), it would be impossible. That's why I pre-computed volumetric representations of size 48 in a seperate folder and one can load a mini-batch for each stochastic gradient update.
- If you are interested in other resolutions, you may run write_input_data.m to precompute the volumes.
