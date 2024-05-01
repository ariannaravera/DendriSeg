from skimage.io import imread
import pyclesperanto_prototype as cle  # version 0.24.1
import napari_segment_blobs_and_things_with_membranes as nsbatwm  # version 0.3.4
import napari_simpleitk_image_processing as nsitk
import napari_pyclesperanto_assistant as ncle
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import tifffile
from cellpose import utils
from skimage import morphology

data_path = "/Users/aravera/Documents/DNF_Bagni/Giorgia/data"
res_path = "/Users/aravera/Documents/DNF_Bagni/Giorgia/results"

def workflowneu2(image):
    # median filter
    image_M = nsitk.median_filter(image, 1, 1, 0)
    # threshold otsu
    image_T1 = nsitk.threshold_otsu(image_M)
    # laplacian of gaussian
    image_L = ncle._napari_cle_functions.laplacian_of_gaussian(image_M, 0.0, 0.0, 0.0)
    # gaussian blur
    image_gb = cle.gaussian_blur(image_L, None, 1.0, 1.0, 0.0)
    # subtract gaussian background
    image_sgb = cle.subtract_gaussian_background(image_gb, None, 10.0, 10.0, 0.0)
    # curvature flow denoise
    image_C = nsitk.curvature_flow_denoise(image_sgb, 1.0, 2)
    # threshold huang
    image_T2 = nsitk.threshold_huang(image_C)

    image_T1M = cv2.medianBlur(image_T1,3)
    image_T1MC = cv2.morphologyEx(image_T1M, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
    image_T1MCO = morphology.area_opening(image_T1MC, area_threshold=5, connectivity=3)
    image_T1MCOO = morphology.area_opening(image_T1MCO, area_threshold=15, connectivity=6)

    
    image_T2C = cv2.morphologyEx(image_T2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
    image_T2CM = cv2.medianBlur(image_T2C,3)
    image_T2MCO = morphology.area_opening(image_T2CM, area_threshold=5, connectivity=3)
    image_T2MCOO = morphology.area_opening(image_T2MCO, area_threshold=15, connectivity=6)

    image_th = image_T1MCOO+image_T2MCO
    image_th[image_th>0] = 255
    median = cv2.medianBlur(image_th,3)

    return median

def segment(image):
    try:
        # Gaussian blur
        blur = cle.gaussian_blur(image, None, 1.0, 1.0, 1.0)
        # Percentile filter
        filtered = nsbatwm.percentile_filter(blur, 0.0, 0.5)

        # Find neuron center
        center_filtered = filtered.copy().astype('uint8')
        center_filtered[center_filtered < 250] = 0
        center_filtered[center_filtered != 0] = 1
        cnts = cv2.findContours(center_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        max_area = 0
        max_x = 0
        max_y = 0
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            mask = np.zeros(center_filtered.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [c], 1)
            pixels = cv2.countNonZero(mask)
            if pixels > max_area:
                max_area = pixels
                max_x = np.average([x, x+w])
                max_y = np.average([y, y+h])

        # Contrast when needed
        if np.average(filtered) < 20:
            filtered = cv2.convertScaleAbs(filtered, alpha=2, beta=0) #alpha=contrast
            filtered[filtered > 255] = 255
        filtered = filtered.astype('uint8')
        mask = workflowneu2(filtered)
        return mask
    except:
        return None

def crop(image, image_mask, roi_ids, roi_image, output_path):
    cropped_areas = []
    for roi_id in roi_ids:
        roi_mask = np.zeros(image.shape, dtype=np.uint8)
        roi_mask[roi_image == roi_id] = 1
        masked_img = cv2.bitwise_and(image, image, mask = roi_mask)
        masked_image_mask = cv2.bitwise_and(image_mask, image_mask, mask = roi_mask)
        # Crop the black border of the mask and the masked image
        y_nonzero, x_nonzero = np.nonzero(roi_mask)
        cropped_image = masked_img[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]
        cropped_image_mask = masked_image_mask[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]
        cropped_roi_mask = roi_mask[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]
        cropped_area = np.zeros((3, cropped_image.shape[0], cropped_image.shape[1]))
        cropped_area[0] = cropped_image
        cropped_area[1] = cropped_image_mask
        cropped_area[2] = cropped_roi_mask
        cropped_areas.append(cropped_area)
        tifffile.imwrite(output_path+'_image_crop'+str(roi_id)+'.tif', cropped_area)
    return

def open_file(file_path):
    if not os.path.exists(file_path):
        print("ERROR! File not found.")
        return
    if '.tif' not in file_path:
        print("ERROR! File mush have TIF extention.")
        return
    try:
        name = os.path.basename(file_path).split('.')[0]
        image = tifffile.imread(file_path)
        return name, image[0], image[1]
    except Exception as e:
        print("ERROR!\n"+str(e))
        return

def sholl_analysis(output_path, name, center, image, mask):
    if len(list(np.unique(center))) != 2:
        print('WARNING! Select only one center')
        return
    center[center != 0] = 255
    center = center.astype('uint8')
    mass_x, mass_y = np.where(center == 255)
    cent_x = int(np.average(mass_x))
    cent_y = int(np.average(mass_y))
    
    radius = np.arange(1, int(image.shape[0]), 50)
    # Add the sholl circles
    for rad in radius:
        cv2.circle(image, center=(cent_y, cent_x), radius=rad, color=(255,215,0), thickness=1)
    
    plt.figure(frameon=False)
    plt.imshow(image, cmap='gray')#, aspect='auto')
    # Add the center
    plt.plot(cent_y, cent_x, marker="o", mec='mediumblue', markersize=3, color="deepskyblue")
    plt.axis('off')
    plt.savefig(os.path.join(os.path.dirname(str(output_path)), "sholl_"+name.replace(".tif", ".png")), dpi=300, bbox_inches='tight', pad_inches=0)
    #plt.show()
    plt.close()