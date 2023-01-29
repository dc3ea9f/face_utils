# FaceUtils

## Overview
This project is built to easy/end-to-end extract face coeffs and rendering.

## Features
- Extractor
	- DeepFace3DRecon
		- extract face box / 68 keypoints (CPU / CUDA)^[only test with single face images]
		- align with `lm3d_std` and re-crop
		- extract 257-dimensional coefficients
		- get raw face mesh with type `trimesh.Trimesh`
		- visualize face mesh by overlay on original video
- Renderer
	- PIRender
		- render video by given identity image and coefficients 
- Utils
	- read any video to 4D numpy array
	- write 4D uint8 numpy array to any video 

## Usage
Refer to [example.py](./example.py).

## Todos
- [ ] more extractors and renders
- [ ] more tests
- [ ] more documents

## Note
This repo mainly borrows code from [sicxu/Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch) and [RenYurui/PIRender](https://github.com/RenYurui/PIRender)
