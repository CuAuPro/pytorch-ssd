import numpy as np
import logging
import pathlib
import xml.etree.ElementTree as ET
import cv2
import os


class VOCDataset:

    def __init__(self, root, transform=None, target_transform=None, is_test=False, keep_difficult=False, label_file=None, dataset_type="train"):
        """Dataset for VOC data.
        Args:
            root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        if dataset_type == "train":
            self.data_dir = self.root / "train"
        elif dataset_type == "val":
            self.data_dir = self.root / "val"
        elif dataset_type == "test":
            self.data_dir = self.root / "test"
            
        self.ids = self._read_image_ids()
        self.keep_difficult = keep_difficult

        # if the labels file exists, read in the class names
        label_file_name = self.root / "labels.txt"

        if os.path.isfile(label_file_name):
            classes = []

            # classes should be a line-separated list
            with open(label_file_name, 'r') as infile:
                for line in infile:
                    classes.append(line.rstrip())

            # prepend BACKGROUND as first class
            classes.insert(0, 'BACKGROUND')
            #classes  = [ elem.replace(" ", "") for elem in classes]
            self.class_names = tuple(classes)
            logging.info("VOC Labels read from file: " + str(self.class_names))

        else:
            logging.info("No labels file, using default VOC classes.")
            self.class_names = ('BACKGROUND',
            'plate')


        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_id)
        
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
            
        #print('__getitem__  image_id=' + str(image_id) + ' \nboxes=' + str(boxes) + ' \nlabels=' + str(labels))
            
        image = self._read_image(image_id)
        
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
            
        return image, boxes, labels

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    def _read_image_ids(self):
        ids = []
        for filename in self.data_dir.glob("*.xml"):
            image_id = filename.stem
            if self._get_num_annotations(filename) > 0:
                ids.append(image_id)
            else:
                print('warning - image {:s} has no box/labels annotations, ignoring from dataset'.format(filename))
        return ids

    def _get_num_annotations(self, filename):
        objects = ET.parse(filename).findall("object")
        return len(objects)
        
    def _get_annotation(self, image_id):
        annotation_file = self.data_dir / f"{image_id}.xml"
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for object in objects:
            class_name = object.find('name').text.strip() #.lower().strip()
            # we're only concerned with clases in our list
            if class_name in self.class_dict:
                bbox = object.find('bndbox')

                # VOC dataset format follows Matlab, in which indexes start from 0
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1
                boxes.append([x1, y1, x2, y2])

                labels.append(self.class_dict[class_name])
                is_difficult_str = object.find('difficult').text
                is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def _read_image(self, image_id):
        # Check if the file exists
        image_path = self.data_dir / f"{image_id}.jpg"
        if image_path.with_suffix(".jpg").exists():
            image_path = image_path.with_suffix(".jpg")
        elif image_path.with_suffix(".png").exists():
            image_path = image_path.with_suffix(".png")
        else:
            raise FileNotFoundError(f"Image file {image_path} not found.")
        
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image



