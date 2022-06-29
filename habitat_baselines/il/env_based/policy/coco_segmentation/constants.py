# To replace the segmentation model trained on simulation data
# with a segmentation model trained on real-world data
coco_categories_to_goal21_categories = {
    # Goal categories
    56: 1,  # chair (MP3D 3)
    57: 6,  # couch (MP3D 10)
    58: 9,  # plant (MP3D 14)
    59: 7,  # bed (MP3D 11)
    61: 11,  # toilet (MP3D 18)
    62: 14,  # tv (MP3D 22)
}
