import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import tensorflow as tf
import time
import warnings
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from PIL import Image
from pathlib import Path
from shapely.geometry import Polygon


def dir_img_files(path_dir, img_extensions=["png", "jpg", "jpeg"]):
    dir_files = os.listdir(path_dir)

    for file_path in dir_files[:]:  # dir_files[:] makes a copy of dir_files.
        if file_path.split(".")[-1] not in img_extensions:
            dir_files.remove(file_path)
    # print(dir_files)
    return dir_files


def display_image(image, name):

    cv2.imshow("Image", image)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()


def save_image(image, dir_out, image_path):

    path_out = str(dir_out / image_path.stem) + ".png"
    cv2.imwrite(path_out, image)
    print("\nSaved image to ", path_out)


def read_image(image_path, metadata=False):

    image, width, height = None, -1, -1
    image = cv2.imread(image_path)

    if metadata:
        height, width = image.shape[:2]

    return image, width, height


def load_image_into_numpy_array(path):

    return np.array(Image.open(path))


def load_detection_model(path_to_model_dir, path_to_labels):

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logging (1)
    tf.get_logger().setLevel("ERROR")  # Suppress TensorFlow logging (2)

    # Enable GPU dynamic memory allocation
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # LOAD THE MODEL

    PATH_TO_SAVED_MODEL = path_to_model_dir / "saved_model"

    print("Loading model...", end="")
    start_time = time.time()

    # LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
    tf_model = tf.saved_model.load(str(PATH_TO_SAVED_MODEL))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Done! Took {} seconds".format(round(elapsed_time, 3)))

    # LOAD LABEL MAP DATA FOR PLOTTING

    category_index = label_map_util.create_category_index_from_labelmap(
        str(path_to_labels), use_display_name=True
    )

    warnings.filterwarnings("ignore")  # Suppress Matplotlib warnings

    return tf_model, category_index


def read_img_to_tensor(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print("Image read error. Check image path!")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image_expanded = np.expand_dims(image_rgb, axis=0)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    image_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    image_tensor = image_tensor[tf.newaxis, ...]

    return image_tensor, image


def intersects(first, other):
    """ TODO: Adapt to xmin xmax, ymin, ymax """
    return not (
        first.top_right.x < other.bottom_left.x
        or first.bottom_left.x > other.top_right.x
        or first.top_right.y < other.bottom_left.y
        or first.bottom_left.y > other.top_right.y
    )


def threshold_detections(detections, min_confidence):

    accepted_detections = []
    for i, det_score in enumerate(detections["detection_scores"]):
        if det_score > min_confidence:
            accepted_detections.append(i)

    return accepted_detections


def reduce_detections(detections, detection_indices, img_w, img_h):
    """ 
        Removes overlapping detections
        Important note: Tensorflow detections are already sorted by detection score!
            (optional)
        TODO: Could tweak this paramter based on num of faces (2 ears, 1 of each class, per person max)
        TODO: Improvement - merge with face detector?
        TODO: improve selection accuracy by questioning first selection:
            example case:   High confidence for left ear, but all further detections pick right ear in same spot
                            (with similar score)
    """

    unique_classes, unique_indices, unique_boxes = [], [], []
    for i in detection_indices:

        # det_class = detections["detection_classes"][i]
        new_box = get_polygon_bbox(get_pixel_bbox(detections["detection_boxes"][i], img_w, img_h))

        if not any([intersect_boxes(new_box, box_x) for box_x in unique_boxes]):
            # unique_classes.append(det_class)
            unique_indices.append(i)
            unique_boxes.append(
                get_polygon_bbox(get_pixel_bbox(detections["detection_boxes"][i], img_w, img_h))
            )

    return unique_indices


def get_pixel_bbox(tf_bbox, img_w, img_h):

    x_min = int(max(1, (tf_bbox[1] * img_w)))
    x_max = int(min(img_w, (tf_bbox[3] * img_w)))
    y_min = int(max(1, (tf_bbox[0] * img_h)))
    y_max = int(min(img_h, (tf_bbox[2] * img_h)))

    return [x_min, y_min, x_max, y_max]


def get_polygon_bbox(pixel_box):

    [x_min, y_min, x_max, y_max] = pixel_box

    return [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]


def get_iou(box_1, box_2):

    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area

    return iou


def intersect_boxes(box_1, box_2):

    if get_iou(box_1, box_2) > 0:
        return True
    return False


def visualize_detections(image, detections, min_conf, category_index, dir_out, img_path, save=True):

    img_h, img_w, _ = image.shape

    num_detections = int(detections.pop("num_detections"))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections["num_detections"] = num_detections

    # detection_classes should be ints.
    detections["detection_classes"] = detections["detection_classes"].astype(np.int64)

    scores = detections["detection_scores"]
    boxes = detections["detection_boxes"]
    classes = detections["detection_classes"]

    detection_indices_th = threshold_detections(detections, min_conf)
    reduced_detection_indices = reduce_detections(detections, detection_indices_th, img_w, img_h)

    count = 0
    # for i in range(len(scores)):
    for i in reduced_detection_indices:
        if (scores[i] > min_conf) and (scores[i] <= 1.0):
            # increase count
            count += 1
            # Get bounding box coordinates and draw box
            [x_min, y_min, x_max, y_max] = get_pixel_bbox(boxes[i], img_w, img_h)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (10, 255, 0), 2)

            object_name = category_index[int(classes[i])]["name"]
            label = "%d%%:%s" % (int(scores[i] * 100), object_name)  # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_y_min = max(y_min, labelSize[1] + 10)
            # Realign if required (not to draw over image's edges)
            over_edge = (
                ((x_min + labelSize[0]) - img_w) + 5 if ((x_min + labelSize[0]) - img_w) > 0 else 0
            )
            # Print label with bounding rectangle (for visiblity)
            cv2.rectangle(
                image,
                (x_min - over_edge, label_y_min - labelSize[1] - 10),
                (x_min + labelSize[0] - over_edge, label_y_min + baseLine - 10),
                (255, 255, 255),
                cv2.FILLED,
            )
            cv2.putText(
                image,
                label,
                (x_min - over_edge, label_y_min - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

    cv2.putText(
        image,
        "Number of detections : " + str(count),
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (10, 255, 0),
        2,
        cv2.LINE_AA,
    )

    if save:
        save_image(image, dir_out, img_path)


def main():

    PATH_TO_MODEL_DIR = Path("training/exported_models/ear_detect_same_ssd_mobilenet_v2_fpnlite")
    PATH_TO_LABELS = Path("training/annotations/label_map.pbtxt")
    MIN_CONF_THRESH = float(0.05)
    DIR_IMAGES_TEST = Path("test/images")
    DIR_IMAGES_OUT = Path("test/output")

    list_dir_imgs = dir_img_files(DIR_IMAGES_TEST)

    detection_model, category_index = load_detection_model(PATH_TO_MODEL_DIR, PATH_TO_LABELS)

    for img_name in list_dir_imgs:

        print("Running detections on {}... ".format(img_name), end="")
        img_path = DIR_IMAGES_TEST / img_name

        image_tensor, image = read_img_to_tensor(str(img_path))

        # image_tensor = np.expand_dims(image_np, 0)
        detections = detection_model(image_tensor)

        visualize_detections(
            image, detections, MIN_CONF_THRESH, category_index, DIR_IMAGES_OUT, img_path,
        )


if __name__ == "__main__":
    main()
