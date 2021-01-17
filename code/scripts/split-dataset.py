import splitfolders  # or import split_folders


def main():

    DIR_DATASET = "dataset/AWEDataset/awe"
    DIR_DATASET_OUT = "dataset/AWEDataset/awe-train-test-val"

    RATIO_TEST = 0.1
    RATIO_VAL = 0.1
    RATIO_TRAIN = 1.0 - RATIO_TEST - RATIO_VAL

    splitfolders.ratio(
        DIR_DATASET,
        output=DIR_DATASET_OUT,
        seed=128,
        ratio=(RATIO_TRAIN, RATIO_VAL, RATIO_TEST),
        group_prefix=None,
    )


if __name__ == "__main__":
    main()
