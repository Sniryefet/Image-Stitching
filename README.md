# Image Stitching - Final Project

Introduction:
---
Image stitching or photo stitching is the process of combining multiple photographic images with overlapping fields of view to produce a segmented panorama or high-resolution image.

Requirments:
---
* this programs uses opencv and numpy libraries.
* Make sure the loaded pictures are in the right order meaning for each consecutive two there is overlap field of view between them.

Algorithm
---
1. Read images
2. Match feature points (we used SIFT)
3. Find homography between two consecutive pictures
4. Warp each consecutive two pictures into one current panorama picture of the two
5. Repeat sections 3 and 4 until there is only one picture left on the list  

How To Use:
---
* Put the source code and a folder of images under the same directory.
* run main.py /path/to/image_folder

### Input example

```bash
	python main.py testPictures/backyard/
```

* This will create a panoramaTools object that will take care of the rest (building and saving the panorama)

image | image
:-----:|:------:
![](https://github.com/FerasTr/Image-Stitching/blob/master/pictures/road1.jpg) | ![](https://github.com/FerasTr/Image-Stitching/blob/master/pictures/road2.jpg)

### The output
---
#### Result of panorama

![](https://github.com/FerasTr/Image-Stitching/blob/master/pictures/roadPanorama.PNG)

#### Exapmle for matching


![](https://github.com/FerasTr/Image-Stitching/blob/master/pictures/roadinliners.jpg)


**REMARK** \
The output photos are also being saved in the project file 
