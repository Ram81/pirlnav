import json
# The new config inherits a base config to highlight the necessary modification
_base_ = '../mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
# classes = [
#     "background", "tiled wall", "DOOR FRAME", "door/window frame", "door  hinge", " doorframe", "clothing stand", "doorframe", "pc tower", "WALL", "wall hanging decoration", "windowsill", "toy", "doors", "desk", "wall top", "wall hanger", "door ", "piano", "window frame", "wall detail", "sauna bowl", "window ", "bookshelf", "stack", "wall decoration", "shower hose", "sunbed", "backrest", "railing", "armchair", "wall cubby", "window shutter", "shower door", "bulletin board", "shower bar", "window fram", "wall controller", "doorway", "floor ", "patio floor", "ceiiling", "door drame", "mailbox", "kitchen counter", "window shade", "wall beam", "floor vent", "tiled floor", "basket of something", "ceiling lamp", "mirror", "electrical controller", "monitor", "statue", "window /outside", "separator", "beam", "bathroom window", "closet floor", "drywall boards", "bench", "door rame", "windown frame", "table lamp", "doorpost", "door slide", "bathroom cabinet", "window sill", "unknown picture/window", "exercise ladder", "vase", "fire extinguisher", " fireplace wall", "window panel", "window curtain", "door window", "doorstep", "unknown/ probably wall lamp", "bathroom floor", "ceiling/ west wall", "bridge", "closet mirror wall", "door hinge", "stage", "sauna floor", "floor stand", " wall panel", "dumbbell", "ceiling/west wall", "tv stand door", "door/window", "wall board", "coffee maker", "garage door frame", "sofa", "calendar", "garage door", "books", "wall sign", "door framr", "umbrella", "fireplace", "windows frame", "fireplace wall", "window glass", "kitchen wall", "garage door railing", "wall hanging organizer", "gate", "sink", "mat", "rocks", "wall post", "shower door frame", "wall panel frame", "bath wall", "post", "ceiling under stairs", "door knob ", "window  shade", "speaker", "cabinet", "floor mat", "window  frame", "bar cabinet", "curtain", "piano bench", "frame door", "keyboard piano", "yoga mat", "countertop", "appliance", "trash bin", " window frame", "wall light", "pool table", "ceiling window", "rack", "panel wall", "figure", "box", "brochure", "Floor", "garage door opener", "shower floor", "wall  clock", "doorframe ", "sauna heater", "window shades", "lamp table", "Unknown", "sauna wall", "fireplace tool set", "ceiling/ wall west", "door", " wall lamp", "door stopper", "shower wall", "table", "floor lamp", "kitchen cabinet", "screen", "garage door opener motor", "chair", "garden swing", "floor /outside", "wall panel", "window blinds", "stairs railing", "windowsil", "cabinet door", "wall corridor", "window", "wall clock", "wall cabinet", "sliding door", "wall vent", "doorf rame", "sconce", "elevator", "cup", "attic door", "door mat", "wine rack", "display cabinet", "small table", "bar chair", "shelf", "door handle", "window /otherroom", "sink cabinet", "unknown wall", "bathroom cabinet door", "exit sign", "wall soap shelf", "hanger", "lounge chair", "table plant", "window rame", "exercise machine", "painting", "recessed wall", "solarium door", "wall electronics", "kitchen walll", "computer desk", "grass", "drywall board", "garage door motor", "office chair", "roof", "stairs wall", "sauna heat rocks", "ball pool", "clock", "column", "keyboard", "stones", "wall toilet paper", "wall lamp", "sliding glass door", "terrace door", "counter door", "plant", "parapet", "ceiling", "stair wall", "seat", "floor", "unknown", "balustrade", "bathroom wall", "pile of magazines", "shower ceiling", "wall outside", "door knob", " door knob", "doormat", "pot", "door frame", "stairs", "exercise ball", "stove door", "candle", "decoration", "flower vase", "curb", "bucket", "wardrobe sliding door", "walll", "shelving", "garage door opener bar", "glass", "ceiling door", "massage bed", "wall ", "ceiling wall", "bedside cabinet door", "recessed cubby", " wall", "bar", "wall", "coffee table", "shower wall cubby", "closet door", "shower cabin", "night table", "wall tv"
# ]
classes = ["background"] + sorted(json.loads(open("../../spring_2022/AnalyzeSemanticDataset/data/hm3d_semantic/hm3d_objects_train.json").read()))

# print("bef: {}".format(len(classes)))

# print("after: {}".format(len(classes)))

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=len(classes)),
        mask_head=dict(num_classes=len(classes))))

# Modify dataset related settings
dataset_type = 'COCODataset'

data = dict(
    samples_per_gpu=12,  # Batch size of a single GPU
    workers_per_gpu=8,  # Worker to pre-fetch data for each single GPU
    train=dict(
        img_prefix='data/hm3d_semantic/segmentation_coco_150k/train/',
        classes=classes,
        ann_file='data/hm3d_semantic/segmentation_coco_150k/annotations/hm3d_train_deduped.json'),
    val=dict(
        img_prefix='data/hm3d_semantic/segmentation_coco_150k/val/',
        classes=classes,
        ann_file='data/hm3d_semantic/segmentation_coco_150k/annotations/hm3d_val_deduped.json'),
    test=dict(
        img_prefix='data/hm3d_semantic/segmentation_coco_150k/val/',
        classes=classes,
        ann_file='data/hm3d_semantic/segmentation_coco_150k/annotations/hm3d_val_deduped.json'))

optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'