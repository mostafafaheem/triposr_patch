import torch

class Focus:
    def __init__(self, device):
        from carvekit.api.high import HiInterface  #pip install carvekit --extra-index-url https://download.pytorch.org/whl/cpu  #cu117
        self.hiInterface = HiInterface(object_type=['object','hairs-like'][0],  batch_size_seg=1, batch_size_matting=1, seg_mask_size=640, matting_mask_size=2048, trimap_prob_threshold=231, trimap_dilation=30, trimap_erosion_iters=5, fp16=False, device=device)

    @torch.no_grad()
    def __call__(self, image):
        image = self.hiInterface([image])[0]
        return image

def focus(image_source_path, image_target_path):
    import os
    #if os.path.exists(image_target_path): return False
    super_focus = Focus(device=['cpu','cuda'][torch.cuda.is_available()])
    image_todo = sorted(os.listdir(image_source_path))
    for index, image_file in enumerate(image_todo):
        image_save = os.path.join(image_target_path, image_file[0:-len(image_file.split('.')[-1])-1]+'.png')
        if not os.path.exists(image_save):
            image_full = os.path.join(image_source_path, image_file)
            image_rgba = super_focus(image_full)  # [H, W, 4]
            os.makedirs(os.path.dirname(image_save), exist_ok=True)
            image_rgba.save(image_save)
        print('focus:', '%08d/%08d'%(index, len(image_todo)), image_save)
    return True

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
                    
                mesh_dir = os.path.join(obj_path, 'images_mesh/')
                focus_dir = os.path.join(obj_path, 'images_focus/')
                
                if os.path.exists(mesh_dir):
                    print(f"Processing focus for: {source}/{category}/{obj_name}")
                    focus(image_source_path=mesh_dir, image_target_path=focus_dir)

if __name__ == '__main__':
    main()

