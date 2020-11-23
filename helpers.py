import os
import cv2
import tensorflow as tf
import xml.etree.ElementTree as ET

# loads images directly
def load_img(path):
    return cv2.imread(path)

# convert cv2 np array image to grayscale from BGR
def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
# convert cv2 np array image to RGB from BGR
def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
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

# targeted at kaggle_863 dataset only
# loads all images, then maps each image to its associated annotation
# returns a dict with image names (no extension) as key
# values are tuples of (np array image (BGR color), annotation (XML tree root))
def load_kaggle_863(path):
    imgs = load_dir(path + '/images')
    for img_name, img in imgs.items():
        xml_path = img_name + '.xml'
        tree = ET.parse(path + '/annotations/' + xml_path)
        imgs[img_name] = (img, tree.getroot())
    return imgs
    
# parse the boudning boxes from the given XML tree root
def parse_voc_bndboxes(xml_root):
    list_with_all_boxes = []
    
    for boxes in xml_root.iter('object'):

        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return list_with_all_boxes

# found at: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou
	
# CURRENTLY UNUSED
def scale_to_size(imgs: dict, size: tuple):
    for img_name, img in imgs.items():
        imgs[img_name] = tf.image.resize(img, size)
    
    return imgs
# for img_name, img in imgs_with_labels.items()[:5]:
#     bboxes = helpers.parse_voc_bndboxes(img[1])
#     img_data = img[0]

#     for (x1,y1,x2,y2) in bboxes:
#         img_data = cv2.rectangle(img_data,(x1,y1),(x2,y2),(255,0,0),2)

#     cv2.imshow(img_name,img_data, delat)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()