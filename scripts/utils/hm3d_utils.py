import numpy as np

# shapeconv_hm3d_to_rednet_mapping = {
#     1: 1, # 'chair', # 1
#     2: 7, #'table', # 2
#     3: 8, #'picture', # 3
#     4: 9, #'cabinet', # 4
#     5: 10, #'cushion', # 5
#     6: 6, #'sofa', # 6
#     7: 2, #'bed', # 7
#     8: 11, #'chest_of_drawers', # 8
#     9: 3, #'plant', # 9
#     10: 12, #'sink', # 10
#     11: 4, #'toilet', # 11
#     12: 13, #'stool', # 12
#     13: 14, #'towel', # 13
#     14: 5, #'tv_monitor', # 14
#     15: 15, #'shower', # 15
#     16: 16, #'bathtub', # 16
#     17: 17, #'counter', # 17
#     18: 18, #'fireplace', # 18
#     #20: 20, #'shelving', # 19
#     20: 20, #'seating', # 20
#     # 'furniture', # 21
#     # 'appliances', # 22
#     23: 21, #'clothes', # 23
# }

shapeconv_hm3d_to_rednet_mapping = {
    19: 0, #'shelving', # 19
    21: 0, # furniture
    22: 0, # appliances
}

mask_shapeconv_new_cats = np.vectorize(lambda x:
                                        shapeconv_hm3d_to_rednet_mapping[x] if
                                        x in shapeconv_hm3d_to_rednet_mapping
                                        else x)