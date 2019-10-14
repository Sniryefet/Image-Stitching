import sys
import os
from panoramaTools import panoramaTools as pant


if __name__ == "__main__":
    '''
    Input: folder that contains photos
    Output: resultPanorama.jpg and inliners.jpg
    '''
    # Handle file dir
    path_to_imgs = sys.argv[1]
    path_to_imgs.strip()
    if os.path.exists(path_to_imgs):
        imgs_list = sorted([os.path.join(path_to_imgs, imgs_list)
                            for imgs_list in os.listdir(path_to_imgs)], key=os.path.getctime)
        # Build the panorama if the url is correct
        pt = pant(imgs_list)
        pt.buildPanorama()

    else:
        print("failed to load dir")
