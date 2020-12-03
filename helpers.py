import os
import cv2
import tensorflow as tf
import xml.etree.ElementTree as ET
import numpy as np

## CV2 IMAGE LOADING HELPERS
# loads images directly
def load_img(path):
    return cv2.imread(path)

# convert cv2 np array image to grayscale from BGR
def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
# convert cv2 np array image to RGB from BGR
def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
# this assumes each image has a 3 character extension, i.e .png, .jpg
# NOTE: it will fail for extensions like .jpeg
def load_dir(path):
    """ load all images in a directory
    returns a dict with image filenames (no extension) as keys
    np images are values
    """
    imgs = {}
    for img in os.listdir(path):
        # use image name as key
        imgs[img[:-4]] = load_img(path + '/' + img)
    return imgs


## DATASET MANIPULATION
# targeted at kaggle_863 dataset only
# loads all images, then maps each image to its associated annotation
# returns a dict with image names (no extension) as key
# values are tuples of (np array image (BGR color), annotation (XML tree root))
def load_kaggle_863(path):
    imgs = load_dir(path + '/images')
    for img_name, img in imgs.items():
        xml_path = img_name + '.xml'
        tree = ET.parse(path + '/annotations/' + xml_path)
        imgs[img_name] = {
            'img': img, 
            'raw_annotation': tree.getroot()
        }
    return imgs
    
def convert_kaggle_863_for_metrics(imgs_with_labels):
    from od_metrics.BoundingBox import BoundingBox
    from od_metrics.BoundingBoxes import BoundingBoxes
    from od_metrics.utils import BBFormat, BBType
    
    bboxes = BoundingBoxes()
    for img_name, img_data in imgs_with_labels.items():
        boxes, labels = parse_voc_bndboxes(img_data['raw_annotation'])
        for (xmin, ymin, xmax, ymax) in boxes:
            bbox = BoundingBox(
                imageName=img_name,
                classId='face',
                x = xmin,
                y = ymin,
                w = xmax,
                h = ymax,
                bbType=BBType.GroundTruth,
                format=BBFormat.XYX2Y2,
            )
            bboxes.addBoundingBox(bbox)
            
    return {
        'raw_data': imgs_with_labels,
        'bboxes': bboxes
    }

def convert_kaggle_863_for_metrics_class_labels(imgs_with_labels):
    from od_metrics.BoundingBox import BoundingBox
    from od_metrics.BoundingBoxes import BoundingBoxes
    from od_metrics.utils import BBFormat, BBType
    
    bboxes = BoundingBoxes()
    for img_name, img_data in imgs_with_labels.items():
        boxes, labels = parse_voc_bndboxes(img_data['raw_annotation'])
        # for (xmin, ymin, xmax, ymax) in boxes:
        for i in range(len(boxes)):
            (xmin, ymin, xmax, ymax) = boxes[i]
            bbox = BoundingBox(
                imageName=img_name,
                classId=labels[i],
                x = xmin,
                y = ymin,
                w = xmax,
                h = ymax,
                bbType=BBType.GroundTruth,
                format=BBFormat.XYX2Y2,
            )
            bboxes.addBoundingBox(bbox)
            
    return {
        'raw_data': imgs_with_labels,
        'bboxes': bboxes
    }
    
# parse the boudning boxes from the given XML tree root
def parse_voc_bndboxes(xml_root):
    all_boxes = []
    all_labels = []
    
    for boxes in xml_root.iter('object'):

        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        # coco detection tools want boxes in this format
        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        
        all_boxes.append(list_with_single_boxes)
        if boxes.find("name").text == 'mask_weared_incorrect':
            all_labels.append('without_mask')
        else:
            all_labels.append(boxes.find("name").text)

    return all_boxes, all_labels


## EVALUATION
def compute_precision(true_positives, false_positives):
    return true_positives / (true_positives + false_positives)

def compute_recall(true_positives, total_positive):
    return true_positives / total_positive


## CURRENTLY UNUSED
def scale_to_size(imgs: dict, size: tuple):
    for img_name, img in imgs.items():
        imgs[img_name] = tf.image.resize(img, size)
    
    return imgs
    
# def convert_kaggle_863_for_coco(imgs_with_labels):
#     for img_name, img_data in imgs_with_labels.items():
#         boxes = parse_voc_bndboxes(img_data['raw_annotation'])
#         img_data['boxes'] = boxes
#     return imgs_with_labels

# for img_name, img in imgs_with_labels.items()[:5]:
#     bboxes = helpers.parse_voc_bndboxes(img[1])
#     img_data = img[0]

#     for (x1,y1,x2,y2) in bboxes:
#         img_data = cv2.rectangle(img_data,(x1,y1),(x2,y2),(255,0,0),2)

#     cv2.imshow(img_name,img_data, delat)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# cnt = 0
# for img_name, img_data in imgs_with_labels.items():
#     img = img_data['img']
#     gray_img = helpers.to_gray(img)
#     # note to self: it appears decreasing minNeighbors improves performance by ~5% (going from 5 to 1)
#     faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)
    
#     matches = 0
#     matched_boxes = []
#     for labeled_box in labeled_faces:
#     for (x, y, w, h) in faces:
#         box = (x, y, x+w, y+h)
#         # for each face, check its IOU against every face label in this image
#         for labeled_box in labeled_faces:
#             if helpers.bb_intersection_over_union(box, labeled_box) >= iou_threshold:
#                 # consider this a match
#                 matches += 1
#                 matched_boxes.append(box)
# uncomment this block to display each image comparison    
#     for (x1,y1,x2,y2) in labeled_faces:
#         img = cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
#     for (x1, y1, x2, y2) in matched_boxes:
#         img = cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

#     cv2.imshow(img_name,img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

    # disallow negatives - i.e if we generated two matches per face that is ok
#     missed = max(len(labeled_faces) - matches, 0)
#     missed_pct = missed/len(labeled_faces)
#     missed_pct_tracker.append(missed_pct)
#     if missed_pct >= poor_img_threshold:
#         poorest_images[img_name] = img_data
# uncomment this to only iterate over 20 images (useful when displaying each image)
#     cnt += 1
#     if (cnt > 20): break