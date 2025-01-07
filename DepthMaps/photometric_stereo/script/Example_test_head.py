#%%
from photostereo import photometry
import cv2 as cv
import time
import numpy as np
import glob
import os


root_fold = "/Users/dianamarin/Documents/Cheminova/DethMaps/photometric_stereo/photostereo_py/samples/head"
light_manual = False
list_files = glob.glob(f"{root_fold}/*")

list_images = [f for f in list_files if "mask" not in f and "LightMatrix" not in f]
mask_path = [f for f in list_files if "mask" in f][0]
matrix = [f for f in list_files if "LightMatrix" in f][0]
IMAGES = len(list_images)

out = os.path.join(os.path.dirname(root_fold),"out",os.path.basename(root_fold))
if not os.path.exists(out):
    os.makedirs(out)


#Load input image array
image_array = []
for id in list_images:
    try:
        im = cv.imread(id, cv.IMREAD_GRAYSCALE)
        image_array.append(im)
    except cv.error as err:
        print(err)

myps = photometry(IMAGES, False)

if light_manual:
    # SETTING LIGHTS MANUALLY
    #tilts = [136.571, 52.4733, -40.6776, -132.559]
    #slants = [52.6705, 53.2075, 47.3992, 48.8037]
    #slants = [37.3295, 36.7925, 42.6008, 41.1963]

    #tilts = [139.358, 50.7158, -42.5016, -132.627]
    #slants = [74.3072, 70.0977, 69.9063, 69.4498]
    #tilts = [0, 270, 180, 90]
    #slants = [45, 45, 45, 45]

    slants = [71.4281, 66.8673, 67.3586, 67.7405]
    tilts = [140.847, 47.2986, -42.1108, -132.558]

    slants = [42.9871, 49.5684, 45.9698, 43.4908]
    tilts = [-137.258, 140.542, 44.8952, -48.3291]

    myps.setlmfromts(tilts, slants)
    print(myps.settsfromlm())
else:
    # LOADING LIGHTS FROM FILE
    fs = cv.FileStorage(matrix, cv.FILE_STORAGE_READ)
    fn = fs.getNode("Lights")
    light_mat = fn.mat()
    myps.setlightmat(light_mat)
    #print(myps.settsfromlm())

tic = time.process_time()
mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
normal_map = myps.runphotometry(image_array, np.asarray(mask, dtype=np.uint8))
normal_map = cv.normalize(normal_map, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC3)
albedo = myps.getalbedo()
albedo = cv.normalize(albedo, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
#gauss = myps.computegaussian()
#med = myps.computemedian()

cv.imwrite(os.path.join(out,'normal_map.png'),normal_map)
cv.imwrite(os.path.join(out,'albedo.png'),albedo)
#cv.imwrite('gauss.png',gauss)
#cv.imwrite('med.png',med)

toc = time.process_time()
print("Process duration: " + str(toc - tic))

# TEST: 3d reconstruction

# Depth Map V1
depth_map = myps.computedepthmap()
cv.imwrite(os.path.join(out,'depth_map.png'), depth_map)

# Depth  map v2

# depth_map2
depth_map2 = myps.computedepth2()
depth_map2_normalized = cv.normalize(depth_map2, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
cv.imwrite(os.path.join(out,'depth_map2.png'), depth_map2_normalized)

## Custom deep map 
depth_map_c = myps.computedepthmap_custom()
cv.imwrite(os.path.join(out, 'depth_map_poisson.png'), depth_map_c)


# Color depth map v2
depth_map2_colored = cv.applyColorMap(depth_map2_normalized, cv.COLORMAP_JET)
cv.imwrite(os.path.join(out, 'depth_map2_colored.png'), depth_map2_colored)


myps.display3dobj(out)
#cv.imshow("normal", normal_map)
#cv.imshow("mean", med)
#cv.imshow("gauss", gauss)
cv.waitKey(0)
cv.destroyAllWindows()