from scripts.utils.utils import load_json_dataset, write_json

if __name__ == "__main__":
    all_objects = load_json_dataset("configs/detector/all_objects.json")
    
    exclude = ["wall", "floor", "ceiling", "railing", "shelving", "window", "door", "curtain", "stairs", "blinds"]
    filtered_objs = []
    for i, obj in enumerate(all_objects):
        obj = obj.lower()
        is_excluded = False
        for excluded_obj in exclude:
            if excluded_obj in obj:
                is_excluded = True
                break
        if not is_excluded:
            filtered_objs.append(i)
    
    print("Total objs: {}, Filtered objs: {}".format(len(all_objects), len(filtered_objs)))
    
    write_json(filtered_objs, "configs/detector/filtered_objects_mmdet.json")
