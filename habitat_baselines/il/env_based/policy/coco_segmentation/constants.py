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


coco_categories_to_goal21_categories_expanded = {
    # Goal categories
    56: 1,  # chair (MP3D 3)
    57: 6,  # couch (MP3D 10)
    58: 9,  # plant (MP3D 14)
    59: 7,  # bed (MP3D 11)
    61: 11,  # toilet (MP3D 18)
    62: 14,  # tv (MP3D 22)
    # Others (mapped to any free numbers)
    60: 15,  # dining table
    69: 16,  # oven
    71: 17,  # sink
    72: 18,  # refrigerator
    73: 19,  # book
    74: 20,  # clock
    75: 21,  # vase
    41: 22,  # cup
    39: 23,  # bottle
}
