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
import xlsxwriter
from scipy import ndimage

res_path = "/Users/aravera/Documents/PROJECTS/DNF_Bagni/Giorgia/results/new_results"


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
    image_T2C = cv2.morphologyEx(image_T2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
    image_T2CM = cv2.medianBlur(image_T2C,3)
    image_T2MCO = morphology.area_opening(image_T2CM, area_threshold=5, connectivity=3)

    if np.sum(image_T2MCO) > int(np.sum(image)/50):
        image_M1 = ndimage.median_filter(image, size=10)
        # laplacian of gaussian
        image_L = ncle._napari_cle_functions.laplacian_of_gaussian(image_M1, 0.0, 0.0, 0.0)
        # gaussian blur
        image_gb = cle.gaussian_blur(image_L, None, 1.0, 1.0, 0.0)
        # subtract gaussian background
        image_sgb = cle.subtract_gaussian_background(image_gb, None, 10.0, 10.0, 0.0)
        # curvature flow denoise
        image_C = nsitk.curvature_flow_denoise(image_sgb, 1.0, 2)
        # threshold huang
        image_T2 = nsitk.threshold_huang(image_C)
        image_T2C = cv2.morphologyEx(image_T2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
        image_T2CM = cv2.medianBlur(image_T2C,3)
        image_T2MCO = morphology.area_opening(image_T2CM, area_threshold=5, connectivity=3)


    #if np.sum(image_T2MCO) < int(np.sum(image)/50):
    image_th = th+image_T2MCO
    #else:
    #    image_th = th
    image_th[image_th>0] = 1
    median = cv2.medianBlur(image_th,3)

    return median


def segment(image):
    try:
        # Gaussian blur
        blur = cle.gaussian_blur(image, None, 1.0, 1.0, 1.0)
        # Percentile filter
        filtered = nsbatwm.percentile_filter(blur, 0.0, 0.5)
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


def crop(image, image_mask, roi_ids, roi_image, output_path, image_name):
    cropped_areas = []
    print("Cropping the "+str(len(roi_ids))+" ROIs drawn in "+output_path)
    for roi_id in roi_ids:
        rointersection_mask = np.zeros(image.shape, dtype=np.uint8)
        rointersection_mask[roi_image == roi_id] = 1
        masked_img = cv2.bitwise_and(image, image, mask = rointersection_mask)
        masked_image_mask = cv2.bitwise_and(image_mask, image_mask, mask = rointersection_mask)
        # Crop the black border of the mask and the masked image
        y_nonzero, x_nonzero = np.nonzero(rointersection_mask)
        cropped_image = masked_img[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]
        cropped_image_mask = masked_image_mask[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]
        cropped_rointersection_mask = rointersection_mask[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]
        cropped_area = np.zeros((3, cropped_image.shape[0], cropped_image.shape[1]), dtype=np.uint8)
        cropped_area[0] = cropped_image
        cropped_area[1] = cropped_image_mask
        cropped_area[2] = cropped_rointersection_mask
        cropped_areas.append(cropped_area)
        tifffile.imwrite(os.path.join(output_path, image_name+'_ROI'+str(roi_id)+'.tif'), cropped_area, metadata={'axes': 'CYX'}, imagej=True)
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


def sholl_analysis(output_path, name, center, image, mask, scale, rad):
    if len(list(np.unique(center))) != 2:
        print('WARNING! Select only one center')
        return
    center[center != 0] = 255
    center = center.astype('uint8')
    mass_x, mass_y = np.where(center == 255)
    cent_y = int(np.average(mass_x))
    cent_x = int(np.average(mass_y))

    # radius = 5um x 10 times -> we need to scale um to pixel and then we define the range: from 5 to 100 (it is enough for all the images) with step 20
    if rad == 15:
        radius = np.arange(int(scale*15), int(scale*110), int(scale*5))
    else:
        radius = np.arange(int(scale*5), int(scale*100), int(scale*5))

    # Default for cv2 is BGR not RGB !
    image3D = np.zeros((image.shape[0], image.shape[1],3))
    image3D[:,:,0] = image
    image3D[:,:,1] = image
    image3D[:,:,2] = image
    image3D[:,:,2] += mask*50
    image3D[image3D[:,:,2] > 255 ,2] =255

    circles_mask = np.zeros(mask.shape)

    colors = mcp.gen_color(cmap="rainbow", n=20)
    # Add the sholl circles
    for i, rad in enumerate(radius):
        cv2.circle(image3D, center=(cent_x, cent_y), radius=rad, color=ImageColor.getcolor(colors[i], "RGB"), thickness=1)
        cv2.circle(circles_mask, center=(cent_x, cent_y), radius=rad, color=(i+1)*2, thickness=1)
    # Center
    cv2.circle(image3D, center=(cent_x, cent_y), radius=1, color=(0, 204, 0), thickness=1)
    
    workbook = xlsxwriter.Workbook(os.path.join(os.path.dirname(str(output_path)), "sholl_"+name+".xlsx"))
    worksheet = workbook.add_worksheet()
    worksheet.write(0, 0, 'circle id')
    worksheet.write(0, 1, 'intersections')
    row = 1
    
    intersections = mask + circles_mask
    intersections[intersections == 1] = 0
    for i in range(0,100,2):
        intersections[intersections == i] = 0
    for i, col in enumerate(np.unique(intersections)):
        if col != 0:
            intersection_mask = np.zeros(intersections.shape, dtype=np.uint8)
            intersection_mask[intersections==col] = 1
            contours, _ = cv2.findContours(intersection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            worksheet.write(row, 0, i)
            worksheet.write(row, 1, len(contours))
            row += 1
            
            # Intersections' spots
            for c in contours:
                meanx = np.median([i[0][0] for i in c])
                meany = np.median([i[0][1] for i in c])
                cv2.circle(image3D, (int(meanx),int(meany)), 2, ImageColor.getcolor(colors[i-1], "RGB"), -1)

    cv2.imwrite(os.path.join(os.path.dirname(str(output_path)), "sholl_"+name+".jpg"), image3D)
    workbook.close()

    print("Analysis saved in "+str(output_path))
    msg = QMessageBox() 
    msg.setIcon(QMessageBox.Information)
    msg.setText("Analysis saved!")
    msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    msg.exec_()


if __name__ == "__main__":
    pass
    """img = tifffile.imread(data_path)
    mask = segment(img)
    tifffile.imwrite(os.path.join(res_path, 'NeuroniDIV19_40xtiles_mask.tif'), mask)"""

    #img = tifffile.imread('/Users/aravera/Downloads/MAX_40x KO GFP DIV5 200524.lif - KO GFP.tif')
    #'/Users/aravera/Downloads/MAX_40x KO GFP DIV5 200524.lif - KO GFP.tif')
    #'/Users/aravera/Documents/PROJECTS/DNF_Bagni/Giorgia/data/new_data/extracted/40x WT GFP DIV8 210524_img1_ch0_scale=3+52.tif')
    #segment(img)

    #img1 = tifffile.imread('/Users/aravera/Documents/PROJECTS/DNF_Bagni/Giorgia/data/new_data/extracted/40x WT GFP DIV8 210524_img1_ch0_scale=3+52.tif')
    #segment(img1)
    
    #sholl_analysis_test('/Users/aravera/Documents/PROJECTS/DNF_Bagni/Giorgia/results/new_results/40x WT GFP DIV8 210524_img1_ch2_scale=3+52_ROI1.tif')