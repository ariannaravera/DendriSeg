"""
This module contains four napari widgets declared in
different ways:

- a pure Python function flagged with `autogenerate: true`
    in the plugin manifest. Type annotations are used by
    magicgui to generate widgets for each parameter. Best
    suited for simple processing tasks - usually taking
    in and/or returning a layer.
- a `magic_factory` decorated function. The `magic_factory`
    decorator allows us to customize aspects of the resulting
    GUI, including the widgets associated with each parameter.
    Best used when you have a very simple processing task,
    but want some control over the autogenerated widgets. If you
    find yourself needing to define lots of nested functions to achieve
    your functionality, maybe look at the `Container` widget!
- a `magicgui.widgets.Container` subclass. This provides lots
    of flexibility and customization options while still supporting
    `magicgui` widgets and convenience methods for creating widgets
    from type annotations. If you want to customize your widgets and
    connect callbacks, this is the best widget option for you.
- a `QWidget` subclass. This provides maximal flexibility but requires
    full specification of widget layouts, callbacks, events, etc.

References:
- Widget specification: https://napari.org/stable/plugins/guides.html?#widgets
- magicgui docs: https://pyapp-kit.github.io/magicgui/

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from magicgui.widgets import Container, create_widget, PushButton, Label, FileEdit, LineEdit
from qtpy.QtWidgets import QFileDialog
import os
import numpy as np
from .functions import open_lif, segment, crop, open_file, sholl_analysis
import tifffile
import cv2

if TYPE_CHECKING:
    import napari


class LifReader(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        # Browse segmentation mask
        self.file_path = LineEdit(label="Input file")
        self.browse_btn = PushButton(value=True, text='Browse')
        self.browse_btn.clicked.connect(self._add_file)
        # Define output directory
        self.saving_dir = LineEdit(label="Output folder")
        self.browse1_btn = PushButton(value=True, text='Browse')
        self.browse1_btn.clicked.connect(self._add_folder)
        # Button to read lif image
        self.read_btn = PushButton(value=True, text='Read lif file')
        self.read_btn.clicked.connect(self._on_click_read)
       
        # Append into/extend the container with your widgets
        self.extend(
            [
                self.file_path,
                self.browse_btn,
                self.saving_dir,
                self.browse1_btn,
                self.read_btn
            ]
        )

    def _add_file(self):
        # Add the selected mask as input
        self.file_path.value, _ = QFileDialog.getOpenFileName(filter='*.lif')
    
    def _add_folder(self):
        # Add the selected mask as input
        self.saving_dir.value = QFileDialog.getExistingDirectory()

    def _on_click_read(self):
        file_path = self.file_path.value
        saving_dir  = self.saving_dir.value
        if file_path is None:
            print("Select a valid lif file")
            return
        open_lif(file_path, saving_dir)


class ImageSegmentation(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        self.label1 = Label(value='Step 1: Open your tif image')
        self.label2 = Label(value='Step 2: Define the output folder')
        # Define output directory
        self.saving_dir = LineEdit(label="Output folder")
        self.browse_btn = PushButton(value=True, text='Browse')
        self.browse_btn.clicked.connect(self._add_folder)

        # Button to segment image layer selected
        self.label3 = Label(value='Step 3: Perform segmentation')
        # Select Shape and image layers
        self._image_layer_combo = create_widget(
            label="Image", annotation="napari.layers.Image"
        )
        self.segment_btn = PushButton(value=True, text='Segment')
        self.segment_btn.clicked.connect(self._on_click_segment)
        # Button to save mask layer created
        self.save_btn = PushButton(value=True, text='Save mask')
        self.save_btn.clicked.connect(self._on_click_save)

        # Button to create Shape layer
        self.label4 = Label(value='Step 4: Create your ROIs')
        self.btn_create = PushButton(value=True, text='Create ROIs layer')
        self.btn_create.clicked.connect(self._on_click_create)
        # Button to add ellipse to Shape layer
        self.btn_add = PushButton(value=True, text='Add ROI')
        self.btn_add.clicked.connect(self._on_click_add)

        self.label5 = Label(value='Step 5: Crop and save the ROIs areas')

        # Select Shape and image layers
        self._image_layer_combo1 = create_widget(
            label="Image", annotation="napari.layers.Image"
        )
        self._mask_layer_combo = create_widget(
            label="Mask", annotation="napari.layers.Labels"
        )
        self._shape_layer_combo = create_widget(
            label="ROI", annotation="napari.layers.Shapes"
        )
        # Button to crop rois drawn
        self.btn_crop = PushButton(value=True, text='Crop and Save')
        self.btn_crop.clicked.connect(self._on_click_crop)

        # Append into/extend the container with your widgets
        self.extend(
            [
                self.label1,
                self.label2,
                self.saving_dir,
                self.browse_btn,
                self.label3,
                self._image_layer_combo,
                self.segment_btn,
                self.save_btn,
                self.label4,
                self.btn_create,
                self.btn_add,
                self.label5,
                self._image_layer_combo1,
                self._mask_layer_combo,
                self._shape_layer_combo,
                self.btn_crop
            ]
        )
    
    def _on_click_segment(self):
        image_layer = self._image_layer_combo.value
        if image_layer is None:
            print("Select image layer before segmenting")
            return

        image = np.asarray(image_layer.data)
        name = image_layer.name + "_mask"
        self.mask = segment(image)
        if self.mask is not None:
            if name in self._viewer.layers:
                self._viewer.layers[name].data = self.mask
            else:
                self._viewer.add_labels(self.mask, name=name)
    
    def _on_click_save(self):
        image_layer = self._image_layer_combo.value
        if image_layer is None:
            print("Select image layer before segmenting")
            return
        if self.mask is None:
            print("Mask not found")
            return
        if self.saving_dir.value != '':
            output_path = self.saving_dir.value
        else:
            output_path = os.path.dirname(image_layer.source.path)
        tifffile.imwrite(os.path.join(output_path, os.path.splitext(os.path.basename(image_layer.source.path))[0]+"_mask.tif"), self.mask)

    def _add_folder(self):
        # Add the selected mask as input
        self.saving_dir.value = QFileDialog.getExistingDirectory()
        
    def _on_click_create(self):
        self.shapes_layer = self._viewer.add_shapes(name="ROIs")
        self._on_click_add()
    
    def _on_click_add(self):
        ellipse = np.array([[100, 100], [100, 100]])
        self.shapes_layer.add_ellipses(ellipse, edge_width=5,edge_color='coral', face_color='royalblue')
    
    def _on_click_crop(self):
        roi_layer = self._shape_layer_combo.value
        mask_layer = self._mask_layer_combo.value
        image_layer = self._image_layer_combo1.value
        if image_layer is None or mask_layer is None or roi_layer is None:
            print("Select roi, mask and image layers before cropping")
            return
        image = np.asarray(image_layer.data)
        mask = np.asarray(mask_layer.data)
        roi_image = roi_layer.to_labels(image.shape)
        roi_ids = list(np.unique(roi_image))
        roi_ids.remove(0)

        image_path = image_layer.source.path
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        if self.saving_dir.value != '':
            output_path = self.saving_dir.value
        else:
            output_path = os.path.dirname(image_path)
        
        crop(image, mask, roi_ids, roi_image, output_path, image_name)


class ShollAnalysis(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        self.scaledefaultvalue = 1

        self.label1 = Label(value='Step 1: Open the cropped image')
        # Select file
        self.file_layer = FileEdit(value='')
        # Button to convert file into image and mask layer
        self.btn_openfile = PushButton(value=True, text='Open')
        self.btn_openfile.clicked.connect(self._on_click_openfile)

        self.label2 = Label(value='Step 2: Manually clean the mask')

        self.label3 = Label(value='Step 3: Set scale [px/um]')
        # Select file
        self.scale = LineEdit(value=self.scaledefaultvalue)

        self.label4 = Label(value='Step 4: Create shape layer with a dot')
        # Button to create Shape layer
        self.btn_center = PushButton(value=True, text='Create "Neuron center" layer')
        self.btn_center.clicked.connect(self._on_click_centerlayer)

        self.label5 = Label(value='Step 5: Move the dot over the neuron center')

        self.label6 = Label(value='Step 6: perform Sholl Analysis')
        # Button to create Shape layer
        self.btn_sholl = PushButton(value=True, text='Perform analysis')
        self.btn_sholl.clicked.connect(self._on_click_sholl)

        """# Select Shape and image layers
        self._image_layer_combo = create_widget(
            label="Image", annotation="napari.layers.Image"
        )
        self._mask_layer_combo = create_widget(
            label="Mask", annotation="napari.layers.Labels"
        )
        # Button to crop rois drawn
        self.btn_crop = PushButton(value=True, text='Crop')
        self.btn_crop.clicked.connect(self._on_click_crop)"""

        # append into/extend the container with your widgets
        self.extend(
            [
                self.label1,
                self.file_layer,
                self.btn_openfile,
                self.label2,
                self.label3,
                self.scale,
                self.label4,
                self.btn_center,
                self.label5,
                self.label6,
                self.btn_sholl
            ]
        )

    def _on_click_openfile(self):
        self.file_path = self.file_layer.value
        name, image, mask = open_file(str(self.file_path))
        self.image_name = name.split('.tif')[0]
        mask_name = self.image_name+'_mask'
        self._viewer.add_image(image, name=self.image_name)
        self._viewer.add_labels(mask.astype('uint8'), name=mask_name)
        self.image_layer = self._viewer.layers[self.image_name]
        self.mask_layer = self._viewer.layers[mask_name]
        if 'scale' in self.image_name:
            scale = str(self.image_name.split('_ROI')[0].split('scale=')[1]).replace('+','.')
            self.scaledefaultvalue = scale
            self.scale.value = self.scaledefaultvalue
    
    def _on_click_centerlayer(self):
        self.center_layer = self._viewer.add_shapes(name="Neuron center")
        ellipse = np.array([[2, 2], [2, 2]])
        self.center_layer.add_ellipses(ellipse, edge_width=1, edge_color='royalblue', face_color='blue')
        
    def _on_click_sholl(self):
        center_layer = self.center_layer
        image_layer = self.image_layer
        mask_layer = self.mask_layer
        scale = self.scale.value
        if image_layer is None or mask_layer is None or center_layer is None or scale is None:
            print("Set correctly all the layers before performing the analysis")
            return
        center = center_layer.to_labels(image_layer.data.shape)
        sholl_analysis(self.file_path, self.image_name, center, image_layer.data, mask_layer.data, float(scale))
