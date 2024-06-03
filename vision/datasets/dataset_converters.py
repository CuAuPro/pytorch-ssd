import os
import json
import cv2
import tqdm
from PIL import Image
import numpy as np
from PIL import ImageOps
from collections import Counter
from .image_processing import generate_image_rotation_variants


def rotate_image_by_exif(image):
    """
    Rotate photo

    Parameters
    ----------
    image
    """
    try:
        orientation = 274  # key of orientation ExifTags
        if image._getexif() is not None:
            exif = dict(image._getexif().items())
            if orientation in exif.keys():
                if exif[orientation] == 3:
                    image = image.rotate(180, expand=True)
                    image = ImageOps.mirror(image)
                elif exif[orientation] == 6:
                    image = image.rotate(270, expand=True)
                    image = ImageOps.mirror(image)
                elif exif[orientation] == 8:
                    image = image.rotate(90, expand=True)
                    image = ImageOps.mirror(image)
    except AttributeError:
        pass
    return image


def save_in_yolo_format(image,
                        target_boxes,
                        path_to_res_ann,
                        path_to_res_images,
                        image_id, 
                        labels,
                        debug=True,
                        suffix=""):
    height, width, c = image.shape
    to_txt_data = []
    is_corrupted = 0
    for bbox, label_id in zip(target_boxes, labels):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        mx = bbox[0]+w/2
        my = bbox[1]+h/2

        # class x_center y_center width height
        yolo_bbox = [label_id, mx/width, my/height, w/width, h/height]
        if yolo_bbox[1] >= 1 \
                or yolo_bbox[2] >= 1 \
                or yolo_bbox[3] >= 1 \
                or yolo_bbox[4] >= 1:
            print("[corrupted]", os.path.join(path_to_res_images, image_id), width, height)
            print("bbox", bbox)
            print("yolo_bbox", yolo_bbox)
            is_corrupted = 1
        yolo_bbox = " ".join([str(item) for item in yolo_bbox])
        to_txt_data.append(yolo_bbox)
        if debug or is_corrupted:
            cv2.rectangle(image, 
                          (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])),
                          (0, 120, 255),
                          3)
    res_path = f'{path_to_res_ann}/{".".join(image_id.split(".")[:-1])}{suffix}.txt'
    if debug or is_corrupted:
        import matplotlib.pyplot as plt
        
        print(res_path)
        print("\n".join(to_txt_data))
        print("______________________")
        plt.imshow(image)
        plt.show()
        pass
    else:
        with open(res_path, "w") as wFile:
            wFile.write("\n".join(to_txt_data))
        cv2.imwrite(os.path.join(path_to_res_images, 
                                 f"{'.'.join(image_id.split('.')[:-1])}{suffix}.{image_id.split('.')[-1]}"), 
                    image[..., ::-1])

        
def rotation_augumentation(image,
                           target_boxes,
                           path_to_res_ann,
                           path_to_res_images,
                           image_id, 
                           labels,
                           angles=None,
                           debug=False):
    if angles is None:
        angles = [90, 180, 270]
    variant_images, variants_bboxes = generate_image_rotation_variants(image, 
                                                                       target_boxes, 
                                                                       angles=angles)
    angles = [0, *angles]
    for image, target_boxes, angle in zip(variant_images, variants_bboxes, angles):
        save_in_yolo_format(image,
                            target_boxes,
                            path_to_res_ann,
                            path_to_res_images,
                            image_id, 
                            labels,
                            suffix=f"_{angle}",
                            debug=debug)
        
        
def rotation_augumentation_voc(image,
                           target_boxes,
                           path_to_res_ann,
                           path_to_res_images,
                           image_id, 
                           labels,
                           angles=None,
                           debug=False):
    if angles is None:
        angles = [90, 180, 270]
    variant_images, variants_bboxes = generate_image_rotation_variants(image, 
                                                                       target_boxes, 
                                                                       angles=angles)
    
    angles = [0, *angles]
    for image, target_boxes, angle in zip(variant_images, variants_bboxes, angles):
        save_in_voc_format(image,
                            target_boxes,
                            path_to_res_ann,
                            path_to_res_images,
                            image_id, 
                            labels,
                            suffix=f"_{angle}",
                            debug=debug)
        
def convert_dataset_to_yolo_format(path_to_res_ann, 
                                   path_to_res_images, 
                                   path_to_images, 
                                   path_to_json, 
                                   classes=None,
                                   debug=True,
                                   is_generate_image_rotation_variants=False):
    if classes is None:
        classes = ['numberplate']
    with open(path_to_json, encoding="utf8") as ann:
        ann_data = json.load(ann)
    cat2label = {k: i for i, k in enumerate(classes)}
    image_list = ann_data
    all_labels = {}
    c = Counter()

    for _id in tqdm.tqdm(image_list["_via_img_metadata"]):
        image_id = image_list["_via_img_metadata"][_id]["filename"]
        filename = f'{path_to_images}/{image_id}'
        pil_image = Image.open(filename)
        pil_image = rotate_image_by_exif(pil_image)
        image = np.array(pil_image)
        target_boxes = []
        labels = []
        for region in image_list["_via_img_metadata"][_id]["regions"]:
            label = classes[0]
            if region.get("region_attributes", None) is not None:
                if region["region_attributes"].get("label", None) is not None:
                    if isinstance(region["region_attributes"]["label"], str):
                        label = region["region_attributes"]["label"]
                    if isinstance(region["region_attributes"]["label"], int):
                        if region["region_attributes"]["label"] < len(classes):
                            label = classes[region["region_attributes"]["label"]]

            label_id = cat2label.get(label, -1)
            all_labels[label] = 1
            if region["shape_attributes"].get("name", None) is None or label_id == -1:
                c["skip_attributes"] += 1
                continue
            name = region["shape_attributes"]["name"]
            if name == "polygon":
                if region["shape_attributes"].get("all_points_x", None) is None:
                    c["skip_attributes"] += 1
                    continue
                if region["shape_attributes"].get("all_points_y", None) is None:
                    c["skip_attributes"] += 1
                    continue
                bbox = [
                    min(region["shape_attributes"]["all_points_x"]),
                    min(region["shape_attributes"]["all_points_y"]),
                    max(region["shape_attributes"]["all_points_x"]),
                    max(region["shape_attributes"]["all_points_y"]),
                ]
            elif name == "rect":
                bbox = [
                    region["shape_attributes"]["x"],
                    region["shape_attributes"]["y"],
                    region["shape_attributes"]["x"]+region["shape_attributes"]["width"],
                    region["shape_attributes"]["y"]+region["shape_attributes"]["height"],
                ]
            else:
                c["skip_attributes"] += 1
                continue
            c["count_attributes"] += 1
            target_boxes.append(bbox)
            labels.append(label_id)

        if is_generate_image_rotation_variants:
            print(image_id)
            rotation_augumentation(image,
                                   target_boxes,
                                   path_to_res_ann, 
                                   path_to_res_images,
                                   image_id, 
                                   labels,
                                   debug=debug)
        else:
            save_in_yolo_format(image,
                                target_boxes,
                                path_to_res_ann, 
                                path_to_res_images,
                                image_id, 
                                labels,
                                debug=debug)
    print(f"[INFO] find labels {list(all_labels.keys())}")
    print(f"[INFO] use labels {classes}")
    print(f"[INFO] count attributes {c['count_attributes']}")
    print(f"[INFO] skip attributes {c['skip_attributes']}")
    
    
    
    
    
    

    
def convert_dataset_to_voc_format(path_to_res_ann, 
                                   path_to_res_images, 
                                   path_to_images, 
                                   path_to_json, 
                                   classes=None,
                                   debug=True,
                                   is_generate_image_rotation_variants=False):
    if classes is None:
        classes = ['numberplate']
    with open(path_to_json, encoding="utf8") as ann:
        ann_data = json.load(ann)
    cat2label = {k: i for i, k in enumerate(classes)}
    image_list = ann_data
    all_labels = {}
    c = Counter()

    for _id in tqdm.tqdm(image_list["_via_img_metadata"]):
        image_id = image_list["_via_img_metadata"][_id]["filename"]
        filename = f'{path_to_images}/{image_id}'
        pil_image = Image.open(filename)
        pil_image = rotate_image_by_exif(pil_image)
        image = np.array(pil_image)
        target_boxes = []
        labels = []
        for region in image_list["_via_img_metadata"][_id]["regions"]:
            label = classes[0]
            if region.get("region_attributes", None) is not None:
                if region["region_attributes"].get("label", None) is not None:
                    if isinstance(region["region_attributes"]["label"], str):
                        label = region["region_attributes"]["label"]
                    if isinstance(region["region_attributes"]["label"], int):
                        if region["region_attributes"]["label"] < len(classes):
                            label = classes[region["region_attributes"]["label"]]

            label_id = cat2label.get(label, -1)
            all_labels[label] = 1
            if region["shape_attributes"].get("name", None) is None or label_id == -1:
                c["skip_attributes"] += 1
                continue
            name = region["shape_attributes"]["name"]
            if name == "polygon":
                if region["shape_attributes"].get("all_points_x", None) is None:
                    c["skip_attributes"] += 1
                    continue
                if region["shape_attributes"].get("all_points_y", None) is None:
                    c["skip_attributes"] += 1
                    continue
                bbox = [
                    min(region["shape_attributes"]["all_points_x"]),
                    min(region["shape_attributes"]["all_points_y"]),
                    max(region["shape_attributes"]["all_points_x"]),
                    max(region["shape_attributes"]["all_points_y"]),
                ]
            elif name == "rect":
                bbox = [
                    region["shape_attributes"]["x"],
                    region["shape_attributes"]["y"],
                    region["shape_attributes"]["x"]+region["shape_attributes"]["width"],
                    region["shape_attributes"]["y"]+region["shape_attributes"]["height"],
                ]
            else:
                c["skip_attributes"] += 1
                continue
            c["count_attributes"] += 1
            target_boxes.append(bbox)
            labels.append(classes[label_id])

        if is_generate_image_rotation_variants:
            #print(image_id)
            rotation_augumentation_voc(image,
                                   target_boxes,
                                   path_to_res_ann, 
                                   path_to_res_images,
                                   image_id, 
                                   labels,
                                   debug=debug)
        else:
            save_in_voc_format(image,
                                target_boxes,
                                path_to_res_ann, 
                                path_to_res_images,
                                image_id, 
                                labels,
                                debug=debug)
    print(f"[INFO] find labels {list(all_labels.keys())}")
    print(f"[INFO] use labels {classes}")
    print(f"[INFO] count attributes {c['count_attributes']}")
    print(f"[INFO] skip attributes {c['skip_attributes']}")



def save_in_voc_format(image,
                        target_boxes,
                        path_to_res_ann,
                        path_to_res_images,
                        image_id, 
                        labels,
                        debug=True,
                        suffix=""):
    height, width, c = image.shape
    to_txt_data = []
    is_corrupted = 0
    for bbox, label_id in zip(target_boxes, labels):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        mx = bbox[0]+w/2
        my = bbox[1]+h/2

        # class x_center y_center width height
        yolo_bbox = [label_id, mx/width, my/height, w/width, h/height]
        if yolo_bbox[1] >= 1 \
                or yolo_bbox[2] >= 1 \
                or yolo_bbox[3] >= 1 \
                or yolo_bbox[4] >= 1:
            print("[corrupted]", os.path.join(path_to_res_images, image_id), width, height)
            print("bbox", bbox)
            print("yolo_bbox", yolo_bbox)
            is_corrupted = 1
            
    res_path = f'{path_to_res_ann}/{".".join(image_id.split(".")[:-1])}{suffix}.xml'
    
    vocFileData = convert_to_voc_format(target_boxes, labels, image_id, height, width)
    with open(res_path, "w") as wFile:
        wFile.write(vocFileData)
    cv2.imwrite(os.path.join(path_to_res_images, 
                                f"{'.'.join(image_id.split('.')[:-1])}{suffix}.{image_id.split('.')[-1]}"), 
                image[..., ::-1])
                
def convert_to_voc_format(target_boxes, labels, image_filename, im_h, im_w):
    xml_str = "<annotation>\n"
    xml_str += "\t<folder></folder>\n"
    xml_str += f"\t<filename>{image_filename}</filename>\n"
    xml_str += f"\t<path>{image_filename}</path>\n"
    xml_str += "\t<source>\n"
    xml_str += "\t\t<database></database>\n"
    xml_str += "\t</source>\n"
    xml_str += "\t<size>\n"
    xml_str += f"\t\t<width>{im_w}</width>\n"
    xml_str += f"\t\t<height>{im_h}</height>\n"
    xml_str += "\t\t<depth>3</depth>\n"
    xml_str += "\t</size>\n"
    xml_str += "\t<segmented>0</segmented>\n"

    for i, target_box in enumerate(target_boxes):
        xml_str += "\t<object>\n"
        xml_str += f"\t\t<name>{labels[i]}</name>\n"
        xml_str += "\t\t<pose>Unspecified</pose>\n"
        xml_str += "\t\t<truncated>0</truncated>\n"
        xml_str += "\t\t<difficult>0</difficult>\n"
        xml_str += "\t\t<occluded>0</occluded>\n"
        xml_str += "\t\t<bndbox>\n"
        xml_str += f"\t\t\t<xmin>{target_box[0]}</xmin>\n"
        xml_str += f"\t\t\t<xmax>{target_box[2]}</xmax>\n"
        xml_str += f"\t\t\t<ymin>{target_box[1]}</ymin>\n"
        xml_str += f"\t\t\t<ymax>{target_box[3]}</ymax>\n"
        xml_str += "\t\t</bndbox>\n"
        xml_str += "\t</object>\n"

    xml_str += "\t<metadata>\n"
    xml_str += "\t\t<tags>\n"
    xml_str += "\t\t\t<tag>Object-detection</tag>\n"
    xml_str += "\t\t</tags>\n"
    xml_str += "\t</metadata>\n"
    xml_str += "</annotation>"

    return xml_str