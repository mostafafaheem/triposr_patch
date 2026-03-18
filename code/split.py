def split(image_path_source, image_path_target, train_percent):
    import os
    import random
    import shutil

    print('split ...')
    sub_sets = ['train','valid']
    for sub_set in sub_sets:
        sub_path = os.path.join(image_path_target, sub_set)
        shutil.rmtree(sub_path, ignore_errors=True)
        os.makedirs(sub_path, exist_ok=True)
    random.seed(122333)
    unit = 10000
    for one_file in os.listdir(image_path_source):
        if os.path.isfile(os.path.join(image_path_source, one_file)): 
            sub_path = sub_sets[0] if random.randrange(0,unit)<0.8*unit else sub_sets[1]      
            shutil.copy(os.path.join(image_path_source, one_file), os.path.join(image_path_target, sub_path, one_file))

def main():
    import os
    image_root = './data/image/'
    
    if not os.path.exists(image_root):
        print(f"Image root directory {image_root} not found.")
        return

    for source in os.listdir(image_root):
        source_path = os.path.join(image_root, source)
        if not os.path.isdir(source_path):
            continue
            
        for category in os.listdir(source_path):
            category_path = os.path.join(source_path, category)
            if not os.path.isdir(category_path):
                continue
                
            for obj_name in os.listdir(category_path):
                obj_path = os.path.join(category_path, obj_name)
                if not os.path.isdir(obj_path):
                    continue
                    
                focus_dir = os.path.join(obj_path, 'images_focus/')
                split_dir = os.path.join(obj_path, 'images_split/')
                
                if os.path.exists(focus_dir):
                    print(f"Processing split for: {source}/{category}/{obj_name}")
                    split(image_path_source=focus_dir, image_path_target=split_dir, train_percent=0.8)

if __name__ == '__main__':
    main()

