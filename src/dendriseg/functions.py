import pyclesperanto_prototype as cle  # version 0.24.1
import napari_segment_blobs_and_things_with_membranes as nsbatwm  # version 0.3.4
import napari_simpleitk_image_processing as nsitk
import napari_pyclesperanto_assistant as ncle
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import tifffile
from skimage import morphology
from readlif.reader import LifFile
from PyQt5.QtWidgets import QMessageBox
from mycolorpy import colorlist as mcp
from PIL import ImageColor
import csv

data_path = "/Users/aravera/Documents/DNF_Bagni/Giorgia/data/NeuroniDIV19_40xtiles.tif"
res_path = "/Users/aravera/Documents/DNF_Bagni/Giorgia/results23052024"


def open_lif(file_path, saving_dir):
    if not os.path.isfile(file_path):
        print(f"File '{file_path}' not found.")
    else:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        lif_file = LifFile(file_path)
        for i, img in enumerate(lif_file.get_iter_image()):
            scale = img.info['scale'][0]
            for c in range(img.info['channels']):
                image_z = np.zeros((img.info['dims'].x, img.info['dims'].y, img.info['dims'].z))
                for z in range(img.info['dims'].z):
                    image_z[:,:,z] = img.get_frame(z=int(z), t=0, c=int(c))
                tifffile.imwrite(os.path.join(saving_dir, file_name+'_img'+str(i+1)+'_ch'+str(c)+'_scale='+str(round(scale,2)).replace('.','+')+'.tif'), np.max(image_z, axis=2).astype('uint16'), imagej=True)
        print('Files saved in '+saving_dir)
        msg = QMessageBox() 
        msg.setIcon(QMessageBox.Information)
        msg.setText("Finished!")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg.exec_() 


def workflowneu2(image):
    # median filter
    image_M = nsitk.median_filter(image, 1, 1, 0)
    # threshold otsu
    image_T1 = nsitk.threshold_otsu(image_M)
    #_, th1 = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
    th = cv2.erode(image_T1, np.ones((5, 5), np.uint8), iterations=1) 
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

    """image_T1M = cv2.medianBlur(image_T1,3)
    image_T1MC = cv2.morphologyEx(image_T1M, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
    image_T1MCO = morphology.area_opening(image_T1MC, area_threshold=5, connectivity=3)
    image_T1MCOO = morphology.area_opening(image_T1MCO, area_threshold=15, connectivity=6)"""

    image_T2C = cv2.morphologyEx(image_T2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
    image_T2CM = cv2.medianBlur(image_T2C,3)
    image_T2MCO = morphology.area_opening(image_T2CM, area_threshold=5, connectivity=3)

    image_th = th+image_T2MCO
    image_th[image_th>0] = 1
    median = cv2.medianBlur(image_th,3)

    return median


def segment(image):
    try:
        # Gaussian blur
        blur = cle.gaussian_blur(image, None, 1.0, 1.0, 1.0)
        # Percentile filter
        filtered = nsbatwm.percentile_filter(blur, 0.0, 0.5)

        """# Find neuron center
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
                max_y = np.average([y, y+h])"""

        # Contrast when needed
        if np.average(filtered) < 20:
            filtered = cv2.convertScaleAbs(filtered, alpha=2, beta=0) #alpha=contrast
            filtered[filtered > 255] = 255
        filtered = filtered.astype('uint8')
        
        mask = workflowneu2(filtered)
        msg = QMessageBox() 
        msg.setIcon(QMessageBox.Information)
        msg.setText("Mask created!")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg.exec_() 

        return mask
    except:
        return None


def crop(image, image_mask, roi_ids, roi_image, output_path, image_name, scale):
    cropped_areas = []
    print("Cropping the "+str(len(roi_ids))+" ROIs drawn in "+output_path)
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
        tifffile.imwrite(os.path.join(output_path, image_name+'_ROI'+str(roi_id)+'.tif'), cropped_area)
    msg = QMessageBox() 
    msg.setIcon(QMessageBox.Information)
    msg.setText("ROIs cropped and saved!")
    msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    msg.exec_() 
    return


def open_file(file_path):
    if not os.path.exists(file_path):
        print("ERROR! File not found.")
        return
    if '.tif' not in file_path:
        print("ERROR! File must have TIF extention.")
        return
    try:
        name = os.path.basename(file_path).split('.')[0]
        image = tifffile.imread(file_path)
        return name, image[0], image[1]
    except Exception as e:
        print("ERROR!\n"+str(e))
        return


def sholl_analysis(output_path, name, center, image, mask, scale):
    if len(list(np.unique(center))) != 2:
        print('WARNING! Select only one center')
        return
    center[center != 0] = 255
    center = center.astype('uint8')
    mass_x, mass_y = np.where(center == 255)
    cent_x = int(np.average(mass_x))
    cent_y = int(np.average(mass_y))

    # radius = 10um x 10 times -> we need to scale um to pixel and then we define the range: from 10 to 200 (it is enough for all the images) with step 10
    radius = np.arange(int(scale*10), int(scale*100), int(scale*10))
    image3D = np.zeros((image.shape[0], image.shape[1],3))
    image3D[:,:,0] = image
    image3D[:,:,1] = image
    image3D[:,:,2] = image

    circles_mask = np.zeros(mask.shape)

    colors = mcp.gen_color(cmap="rainbow",n=10)
    # Add the sholl circles
    for i, rad in enumerate(radius):
        cv2.circle(image3D, center=(cent_y, cent_x), radius=rad, color=ImageColor.getcolor(colors[i], "RGB"), thickness=1)
        cv2.circle(circles_mask, center=(cent_y, cent_x), radius=rad, color=(i+1)*2, thickness=1)
    
    cv2.circle(image3D, center=(cent_y, cent_x), radius=1, color=(0,100,255), thickness=1)
    cv2.imwrite(os.path.join(os.path.dirname(str(output_path)), "sholl_"+name+".jpg"), image3D)
    
    with open(os.path.join(os.path.dirname(str(output_path)), "sholl_"+name+".csv"), "w") as file:
        writer = csv.writer(file)
        writer.writerow(['circle', 'num intersections'])
    
    intersections = mask + circles_mask
    intersections[intersections == 1] = 0
    for i in range(0,100,2):
        intersections[intersections == i] = 0
    for i, col in enumerate(np.unique(intersections)):
        if col != 0:
            i_mask = np.zeros(intersections.shape, dtype=np.uint8)
            i_mask[intersections==col] = 1
            contours, _ = cv2.findContours(i_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            with open(os.path.join(os.path.dirname(str(output_path)), "sholl_"+name+".csv"), "a") as file:
                writer = csv.writer(file)
                writer.writerow([i, len(contours)])

    print("Analysis saved in "+str(output_path))
    msg = QMessageBox() 
    msg.setIcon(QMessageBox.Information)
    msg.setText("Analysis saved!")
    msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    msg.exec_()



if __name__ == "__main__":
    """img = tifffile.imread(data_path)
    mask = segment(img)
    tifffile.imwrite(os.path.join(res_path, 'NeuroniDIV19_40xtiles_mask.tif'), mask)"""
    open_lif('/Users/aravera/Documents/DNF_Bagni/Giorgia/data/new_data/40x WT GFP DIV8 210524.lif', '')