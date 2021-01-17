import os
from typing import MutableMapping
import cv2
import pandas as pd
from pathlib import Path
from pascal_voc_writer import Writer


def dir_png_files(path_dir):

    dir_files = os.listdir(path_dir)

    for file_path in dir_files[:]:  # dir_files[:] makes a copy of dir_files.
        if not (file_path.endswith(".png")):
            dir_files.remove(file_path)
    # print(dir_files)
    return dir_files


def display_image(image):

    cv2.imshow("Image", image)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()


def read_image(image_path, metadata=False):

    image, width, height = None, -1, -1
    image = cv2.imread(image_path)

    if metadata:
        height, width = image.shape[:2]

    return image, width, height


def show_contour_box(image, cnt, label):

    # draw on a copy
    img_cnt = image.copy()
    # visualize bbox
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img_cnt, (x, y), (x + w, y + h), (0, 255, 0), 5)

    # display class label
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        img_cnt, label, (x + int(w / 2), y + int(h / 2)), font, 1, (255, 255, 0), 2, cv2.LINE_AA
    )

    display_image(img_cnt)


def detect_contours(image, class_override=None, dual=False, debug=False):

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 2:
        print("Found image with more than 2 contours.")

    img_w = img_gray.shape[1]

    contours_data = []
    centers = []

    for cnt in contours:
        # x,y => left, top
        x, y, w, h = cv2.boundingRect(cnt)

        cnt_class = "human-ear" if (class_override is None) else class_override

        if dual:
            # distinguish between left and right ear detection
            # dataset specific - max 2 ears in an images
            dist_left_edge = x
            dist_right_edge = img_w - (x + w)
            ear_type_ext = "right" if dist_left_edge <= dist_right_edge else "left"
            cnt_class = "-".join([cnt_class, ear_type_ext])
            centers.append(x + int(w / 2))

        if debug:
            show_contour_box(image, cnt, cnt_class)

        contours_data.append([cnt_class, [x, y, w, h]])

    # TODO: validate - do not accept two of same class (when dual)
    if len(contours) == 2 and dual:
        if centers[0] <= centers[1]:
            contours_data[0][0] = contours_data[0][0].replace("-left", "-right")
            contours_data[1][0] = contours_data[1][0].replace("-right", "-left")
        else:
            contours_data[0][0] = contours_data[0][0].replace("-right", "-left")
            contours_data[1][0] = contours_data[1][0].replace("-left", "-right")

    return contours_data


def convert_bbox_opencv_to_pascal(bbox):
    """
    Bounding box format(s)
    - Opencv contour: x-top left, y-topleft, width, height
    - Pascal VOC : x-top left, y-top left,x-bottom right, y-bottom right
    """
    [x, y, w, h] = bbox

    return [x, y, x + w, y + h]


def create_pascal_voc_xml(path_image, path_xml, img_w, img_h, objects):

    # Writer(path, width, height)
    writer = Writer(path_image, img_w, img_h)

    for (class_label, cnt_bbox) in objects:

        (class_label, [xmin, ymin, xmax, ymax]) = (
            class_label,
            convert_bbox_opencv_to_pascal(cnt_bbox),
        )

        writer.addObject(class_label, xmin, ymin, xmax, ymax)

    writer.save(path_xml)

    return True


def run_annotation_conversion(dir_imgs, dir_masks, dir_xmls, ids=None, dual=False):

    if not os.path.exists(dir_xmls):
        os.makedirs(dir_xmls)

    png_files = dir_png_files(dir_imgs)

    for img_path in png_files:

        image, img_w, img_h = read_image(str(dir_masks / img_path), metadata=True)

        class_override = None
        if ids is not None:
            parent_dir = os.path.split(dir_imgs)[-1]
            class_override = "PERSON-" + str(ids[parent_dir + "/" + img_path])

        contours = detect_contours(image, class_override=class_override, dual=dual)

        succ = create_pascal_voc_xml(
            str(dir_imgs / img_path),
            str((dir_xmls / img_path).with_suffix(".xml")),
            img_w,
            img_h,
            contours,
        )

        if len(contours) == 0 or not succ:
            print("Problems with image: ", str(dir_masks / img_path))

    return True


def read_annotation_ids(path_csv_file, export=False, multi_class=None):

    dict_df_ids = pd.read_csv(
        path_csv_file, usecols=["AWE-Full image path", "Subject ID"], index_col=0, squeeze=True,
    ).to_dict()

    if export:
        class_ids = list(set(list(dict_df_ids.values())))
        with open(str(path_csv_file.parents[0] / "label_map.pbtxt"), "w") as pbtxt_file:

            for i, id in enumerate(class_ids):
                if multi_class is not None:
                    for cls_i, str_class in enumerate(multi_class):
                        class_indx = (i * len(multi_class)) + 1 + cls_i
                        pbtxt_file.write("item {\n")
                        pbtxt_file.write("    id: {}\n".format(int(class_indx)))
                        pbtxt_file.write(
                            "    name: '{0}'\n".format("-".join(["PERSON", str(id), str_class]))
                        )
                        pbtxt_file.write(
                            "    display_name: '{0}'\n".format(
                                "-".join(["PERSON", str(id), str_class])
                            )
                        )
                        pbtxt_file.write("}\n\n")
                else:
                    pbtxt_file.write("item {\n")
                    pbtxt_file.write("    id: {}\n".format(int(i + 1)))
                    pbtxt_file.write("    name: '{0}'\n".format("PERSON-" + str(id)))
                    pbtxt_file.write("    display_name: '{0}'\n".format("PERSON-" + str(id)))
                    pbtxt_file.write("}\n\n")

        if multi_class is not None:
            print("NUMBER OF ALL CLASSES: ", len(class_ids * len(multi_class)))
        else:
            print("NUMBER OF ALL CLASSES: ", len(class_ids))

    return dict_df_ids


def main():

    TEST_IMGS = Path("training/images/test/")
    TEST_MASKS = Path("training/images/test_masks_rect/")
    TEST_XMLS = TEST_IMGS

    TRAIN_IMGS = Path("training/images/train/")
    TRAIN_MASKS = Path("training/images/train_masks_rect/")
    TRAIN_XMLS = TRAIN_IMGS

    MODE = "simple"
    # MODE = "link_ids"
    MODE = "link_ids_dual"

    if MODE == "link_ids" or MODE == "link_ids_dual":
        FILE_IDS = Path("training/annotations/awe-translation.csv")
        file_id_map = read_annotation_ids(FILE_IDS, export=True, multi_class=["left", "right"])
    else:
        file_id_map = None

    # if run_annotation_conversion(TEST_IMGS, TEST_MASKS, TEST_XMLS, file_id_map, dual=True):
    if run_annotation_conversion(TRAIN_IMGS, TRAIN_MASKS, TRAIN_XMLS, file_id_map, dual=True):
        print("Conversion process was successful!")
    else:
        print("Problem(s) encountered during conversion!")


if __name__ == "__main__":
    main()
