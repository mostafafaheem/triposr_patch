import torch
import numpy as np
import os

import math
class VisionDataset(torch.utils.data.Dataset):
    class Coordinate:
        def view_to_world(distance, azimuth, elevation, is_degree):  #pytorch3d.renderer.look_at_view_transform(dist=distance, elev=elevation, azim=azimuth)  #horizontal plane y=0
            if is_degree:
                azimuth_radian = azimuth /180.0*math.pi
                elevation_radian = elevation /180.0*math.pi
            else:
                azimuth_radian = azimuth
                elevation_radian = elevation

            x = distance * torch.cos(elevation_radian) * torch.sin(azimuth_radian)
            y = distance * torch.sin(elevation_radian)
            z = distance * torch.cos(elevation_radian) * torch.cos(azimuth_radian)
            camera_position = torch.stack([x, y, z], dim=1)  #world

            at = torch.tensor(((0, 0, 0),), dtype=torch.float32)
            up = torch.tensor(((0, 1, 0),), dtype=torch.float32)
            z_axis = torch.nn.functional.normalize(at - camera_position, eps=1e-5)  #first
            x_axis = torch.nn.functional.normalize(torch.cross(up, z_axis, dim=1), eps=1e-5)
            y_axis = torch.nn.functional.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5)

            R = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)
            t = -torch.bmm(R, camera_position[:, :, None])[:, :, 0]
            R = R.transpose(1, 2)
            return R, t

        def world_to_view(R, t, to_degree):
            camera_position = -torch.bmm(R, t[:, :, None])[:, :, 0]

            distance = torch.norm(camera_position)

            azimuth_radian = torch.atan2(camera_position[:, 0], camera_position[:, 2])
            elevation_radian = torch.atan2(camera_position[:, 1], torch.sqrt(camera_position[:, 0]**2 + camera_position[:, 2]**2))

            if to_degree:
                azimuth = azimuth_radian /math.pi*180.0
                elevation = elevation_radian /math.pi*180.0

            return distance, azimuth, elevation

        def Rt_to_matrix(R, t):
            matrix = torch.eye(4)
            matrix[:3, :3] = R
            matrix[:3, 3] = t
            return matrix

    def __init__(self, is_train, data_path, image_size):
        def pick_pose_from_file(data_file):
            item = data_file[len('image__'):-len('.png')].split('__')  #image__distance_2_0__elevation_330__azimuth_330.png
            distance = float(item[0][len('distance_'):].replace('_', '.'))
            elevation = int(item[1][len('elevation_'):])
            azimuth = int(item[2][len('azimuth_'):])
            matrix = self.Coordinate.Rt_to_matrix(*self.Coordinate.view_to_world(torch.tensor([distance]), torch.tensor([azimuth]), torch.tensor([elevation]), is_degree=True))
            return torch.Tensor(matrix)

        import torchvision
        if is_train:
            transforms = torchvision.transforms.Compose([torchvision.transforms.Resize([image_size,image_size]), torchvision.transforms.ToTensor()])  #augment
        else:
            transforms = torchvision.transforms.Compose([torchvision.transforms.Resize([image_size,image_size]), torchvision.transforms.ToTensor()])

        import os
        import PIL
        self.data_all = []
        import glob
        # Search for images recursively in data_path (e.g. /data/image/[source]/[category]/[id]/images_split/train/)
        # Pattern: image__distance_*.png
        search_pattern = os.path.join(data_path, "**", "train", "image__*.png") if is_train else os.path.join(data_path, "**", "valid", "image__*.png")
        image_files = sorted(glob.glob(search_pattern, recursive=True))
        
        if not image_files:
            # Fallback to direct listdir if no recursive images found
            image_files = [os.path.join(data_path, f) for f in sorted(os.listdir(data_path)) if f.startswith('image__') and f.endswith('.png')]

        for image_full_path in image_files:
            data_file = os.path.basename(image_full_path)
            with open(image_full_path, "rb") as handler:
                rgba = PIL.Image.open(handler).convert("RGBA")
                rgba = transforms(rgba)
                image = rgba[0:3,:,:].permute(1,2,0)
                mask = rgba[3:4,:,:].permute(1,2,0)
                pose = pick_pose_from_file(data_file)
                self.data_all.append((image,mask,pose))
    
    def __len__(self):
        return len(self.data_all)

    def __getitem__(self, index):
        item = self.data_all[index]
        return item

def infer(image_size, image_path, output_file, remove_bg, foreground_ratio, render_video, device):
    print('superv ...')
    if remove_bg:
        import rembg  #pip install rembg
        import PIL
        def remove_background(image, rembg_session, **rembg_kwargs):
            if image.mode != "RGBA" or image.getextrema()[3][0] == 255:
                image = rembg.remove(image, session=rembg_session, **rembg_kwargs)
            return image

        def resize_foreground(image, ratio):
            alpha = np.where(image[..., 3] > 0)
            y1, y2, x1, x2 = (alpha[0].min(), alpha[0].max(), alpha[1].min(), alpha[1].max())
            fg = image[y1:y2, x1:x2]  #crop the foreground
            
            size = max(fg.shape[0], fg.shape[1])
            ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
            ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
            new_image = np.pad(fg, ((ph0, ph1), (pw0, pw1), (0, 0)), mode="constant", constant_values=((0, 0), (0, 0), (0, 0)))  #pad to square
            
            new_size = int(new_image.shape[0] / ratio)  #compute padding according to the ratio
            ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
            ph1, pw1 = new_size - size - ph0, new_size - size - pw0
            new_image = np.pad(new_image, ((ph0, ph1), (pw0, pw1), (0, 0)), mode="constant", constant_values=((0, 0), (0, 0), (0, 0)))  #pad to size, double side
            return PIL.Image.fromarray(new_image)

        rembg_session = rembg.new_session()
        image = remove_background(PIL.Image.open(image_path), rembg_session)
        image = resize_foreground(np.array(image), foreground_ratio)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = PIL.Image.fromarray((image * 255.0).astype(np.uint8))
        image = np.array(image).astype(np.float32) / 255.0
    else:
        image = np.array(PIL.Image.open(image_path).convert("RGB"))
 
    image = torch.stack([torch.from_numpy(image)], dim=0)  #batched  
    print('image', image.shape)
    image = torch.nn.functional.interpolate(image.permute(0, 3, 1, 2), (image_size, image_size), mode="bilinear", align_corners=False, antialias=True).permute(0, 2, 3, 1)
    print('image', image.shape)
    image = image[:, None].to(device)
    print('image', image.shape)

    print('superv >>> image ok, network to') 
    from network import TSR
    model = TSR(img_size=image_size, depth=16, embed_dim=768, num_channels=1024, num_layers=16, cross_attention_dim=768, radius=3, valid_thresh=0.001, num_samples_per_ray=128, n_hidden_layers=9, official=True)
    model.load_state_dict(torch.load('/data/ckpt/model.ckpt', map_location='cpu'))
    model.to(device)

    print('superv >>> network ok, infer to')
    with torch.no_grad():
        print('image', image.shape)
        scene_codes = model(image)

    print('superv >>> infer ok, mesh to')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    mesh = model.extract_mesh(scene_codes)[0]
    mesh.export(output_file)  #.ply

    print('superv >>> mesh ok, video to')
    if render_video:
        video_file = output_file[:-len(output_file.split('.')[-1])]+'mp4'
        os.makedirs(os.path.dirname(video_file), exist_ok=True)
        render_images = model.render_images(scene_codes, n_views=16, return_type="pil")[0]
        import imageio  #pip install imageio[ffmpeg]
        with imageio.get_writer(video_file, fps=30) as writer:
            for frame in render_images: 
                writer.append_data(np.array(frame))
            writer.close()
    print('superv !!!')

def train(image_size, batch_size, epochs, checkpoint_path, best_checkpoint_file=None, device=None):    
    def get_ray_bundle(height, width, focal_length, tform_cam2world):
        def meshgrid_xy(tensor1, tensor2):
            ii, jj = torch.meshgrid(tensor1, tensor2, indexing='ij')
            return ii.transpose(-1, -2), jj.transpose(-1, -2)
        ii, jj = meshgrid_xy(torch.arange(width).to(tform_cam2world), torch.arange(height).to(tform_cam2world))
        directions = torch.stack([(ii - width * .5) / focal_length, -(jj - height * .5) / focal_length, -torch.ones_like(ii)], dim=-1)
        ray_directions = torch.sum(directions[..., None, :] * tform_cam2world[:3, :3], dim=-1)
        ray_origins = tform_cam2world[:3, -1].expand(ray_directions.shape)
        return ray_origins, ray_directions

    is_train = 1
    dataset_train = VisionDataset(is_train=is_train, data_path='/data/image/', image_size=image_size)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=bool(is_train), num_workers=1, drop_last=bool(is_train), collate_fn=None, pin_memory=False)

    #images,masks,poses = next(iter(dataloader_train))
    #print('images', images.shape)    #[-1, 32, 32, 3]
    #print('masks',  masks.shape)     #[-1, 32, 32, 1]
    #print('poses',  poses)           #[4, 4]
 
    #data = np.load('./data/nerf/tiny_nerf_data.npz')  #TODO  render and load objavase for training
    #images, poses, focal = data["images"],data["poses"], data["focal"]
    #print('images', images.shape)  #(106, 100, 100, 3)
    #print('poses', poses.shape)    #(106, 4, 4)
    #print('focal', focal)          #138.88887889922103

    #height, width = images.shape[1:3]   #(-1, 224, 224, 3)
    #near_thresh, far_thresh = 1, 1000   #2.0, 6.0

    #images = torch.from_numpy(images[..., :3]).to(device)
    #poses = torch.from_numpy(poses).to(device)
    # PyTorch3d FoVPerspectiveCameras defaults to fov=60 degrees
    # focal_length = image_size / (2 * tan(fov / 2))
    # For fov = 60 degrees, tan(30 deg) ≈ 0.57735
    fov_degrees = 60.0
    focal_length = (image_size / 2.0) / math.tan(fov_degrees * math.pi / 360.0)

    from network import TSR
    # Use official TripoSR parameters for full checkpoint training
    model = TSR(img_size=image_size, depth=16, embed_dim=768, num_channels=1024, num_layers=16, cross_attention_dim=768, radius=3, valid_thresh=0.001, num_samples_per_ray=128, n_hidden_layers=9, official=True)
    
    checkpoint_file = '/data/ckpt/model.ckpt'
    if os.path.exists(checkpoint_file):
        print(f"Loading checkpoint from {checkpoint_file}")
        model.load_state_dict(torch.load(checkpoint_file, map_location='cpu'))
    
    model.to(device)
    model.train()
    
    print("parameters: %d M"%(sum(p.numel() for p in model.parameters())/1024/1024))

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001) # Lower learning rate for fine-tuning
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(range(0, epochs, epochs//10 if epochs>10 else 1)), gamma=0.95)

    try:
        import lpips
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    except ImportError:
        loss_fn_vgg = None
        print("Please install lpips (pip install lpips) for TripoSR LPIPS loss. Falling back to MSE.")

    def save_image(image_I, image_O, alpha_I, alpha_O, flag, epoch, index):
        temp_path = './outs/image/'+flag+'/'
        os.makedirs(temp_path, exist_ok=True)
        image_i = image_I.detach().cpu().numpy()
        image_o = image_O.detach().cpu().numpy()
        alpha_i = alpha_I.detach().cpu().numpy().repeat(repeats=3,axis=2)
        alpha_o = alpha_O.detach().cpu().numpy().repeat(repeats=3,axis=2)
        import PIL
        PIL.Image.fromarray((image_i*255).astype('uint8')).save(temp_path+'/image__epoch_{:04d}__index_{:04d}__{:s}__image_1i.png'.format(epoch,index,flag))
        PIL.Image.fromarray((image_o*255).astype('uint8')).save(temp_path+'/image__epoch_{:04d}__index_{:04d}__{:s}__image_2o.png'.format(epoch,index,flag))
        PIL.Image.fromarray((alpha_i*255).astype('uint8')).save(temp_path+'/image__epoch_{:04d}__index_{:04d}__{:s}__alpha_1i.png'.format(epoch,index,flag))
        PIL.Image.fromarray((alpha_o*255).astype('uint8')).save(temp_path+'/image__epoch_{:04d}__index_{:04d}__{:s}__alpha_2o.png'.format(epoch,index,flag))

    import tqdm

    for epoch in range(0, epochs):
        losses = []
        pbar_train = tqdm.tqdm(dataloader_train, desc=f"Epoch {epoch}/{epochs} [Train]")
        for index, (images,masks,poses) in enumerate(pbar_train):
            target_img = images.to(device)
            target_msk = masks.to(device)
            target_pos = poses.to(device)

            target_img = target_img[:, None]  #[-1, 32, 32, 3]  ->  [-1, 1, 32, 32, 3]
            target_msk = target_msk[:, None]  #[-1, 32, 32, 1]  ->  [-1, 1, 32, 32, 1]
            print('target_img', target_img.shape)
            print('target_msk', target_msk.shape)

            scene_codes = model(target_img)
            #print('scene_codes', scene_codes.shape)  #[-1, 3, 40, 64, 64]

            images_all = []
            masks_all = []
            for idx, scene_code in enumerate(scene_codes):
                images_one = []
                masks_one = []
                for i in range(1):
                    rays_o, rays_d = get_ray_bundle(height=image_size, width=image_size, focal_length=focal_length, tform_cam2world=target_pos[idx])  #[32, 32, 3]  [32, 32, 3]
                    image, alpha = model.renderer(model.decoder, scene_code, rays_o, rays_d)
                    images_one.append(image)
                    masks_one.append(alpha)
                images_all.append(torch.stack(images_one, dim=0))
                masks_all.append(torch.stack(masks_one, dim=0))
            image_pred = torch.stack(images_all, dim=0)
            mask_pred = torch.stack(masks_all, dim=0)
            print('image_pred', image_pred.shape)  #[-1, 1, 32, 32, 3]
            print('mask_pred', mask_pred.shape)  #[-1, 1, 32, 32, 1]

            loss_mse = torch.nn.functional.mse_loss(image_pred, target_img)
            loss_bce = torch.nn.functional.binary_cross_entropy(mask_pred.clamp(1e-5, 1-1e-5), target_msk)
            
            if loss_fn_vgg is not None:
                image_pred_reshaped = image_pred.view(-1, image_size, image_size, 3).permute(0, 3, 1, 2)
                target_img_reshaped = target_img.view(-1, image_size, image_size, 3).permute(0, 3, 1, 2)
                loss_lpips = loss_fn_vgg((image_pred_reshaped * 2) - 1, (target_img_reshaped * 2) - 1).mean()
                # Loss = (1/n other views mse + current view lpips + current view mask bce)
                loss = loss_mse + loss_lpips + loss_bce
                print('loss', loss.item(), 'mse', loss_mse.item(), 'lpips', loss_lpips.item(), 'bce', loss_bce.item())
            else:
                loss = loss_mse + loss_bce
                print('loss', loss.item(), 'mse', loss_mse.item(), 'bce', loss_bce.item())


            losses.append(loss.item())
            pbar_train.set_postfix(loss=f"{loss.item():.4f}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()   

            if index % 200 == 0:
                save_image(target_img[0][0], image_pred[0][0], target_msk[0][0], mask_pred[0][0], flag='train', epoch=epoch, index=index)  

                mesh_path = './outs/image/'+'train'+'/'
                os.makedirs(mesh_path, exist_ok=True)
                mesh = model.extract_mesh(scene_codes)[0]
                mesh.export(os.path.join(mesh_path, "mesh__epoch_{:04d}__index_{:04d}.obj".format(epoch, index)))  #.ply

        LOSS_train = sum(losses)/len(losses)
        print('epoch=%06d  train_loss=%.6f'%(epoch, LOSS_train))

        try:
            dataset_valid = VisionDataset(is_train=0, data_path='/data/image/', image_size=image_size)
            if len(dataset_valid) > 0:
                dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=1)
                model.eval()
                valid_losses = []
                pbar_valid = tqdm.tqdm(dataloader_valid, desc=f"Epoch {epoch}/{epochs} [Valid]")
                with torch.no_grad():
                    for val_idx, (v_images, v_masks, v_poses) in enumerate(pbar_valid):
                        v_img = v_images.to(device)[:, None]
                        v_msk = v_masks.to(device)[:, None]
                        v_pos = v_poses.to(device)

                        v_scene_codes = model(v_img)
                        v_images_all = []
                        v_masks_all = []
                        for idx, scene_code in enumerate(v_scene_codes):
                            rays_o, rays_d = get_ray_bundle(height=image_size, width=image_size, focal_length=focal_length, tform_cam2world=v_pos[idx])
                            img_out, alpha_out = model.renderer(model.decoder, scene_code, rays_o, rays_d)
                            v_images_all.append(img_out.unsqueeze(0))
                            v_masks_all.append(alpha_out.unsqueeze(0))
                        
                        v_image_pred = torch.stack(v_images_all, dim=0)
                        v_mask_pred = torch.stack(v_masks_all, dim=0)

                        v_loss_mse = torch.nn.functional.mse_loss(v_image_pred, v_img)
                        v_loss_bce = torch.nn.functional.binary_cross_entropy(v_mask_pred.clamp(1e-5, 1-1e-5), v_msk)
                        
                        if loss_fn_vgg is not None:
                            v_img_pred_reshaped = v_image_pred.view(-1, image_size, image_size, 3).permute(0, 3, 1, 2)
                            v_img_targ_reshaped = v_img.view(-1, image_size, image_size, 3).permute(0, 3, 1, 2)
                            v_loss_lpips = loss_fn_vgg((v_img_pred_reshaped * 2) - 1, (v_img_targ_reshaped * 2) - 1).mean()
                            v_loss = v_loss_mse + v_loss_lpips + v_loss_bce
                        else:
                            v_loss = v_loss_mse + v_loss_bce
                        valid_losses.append(v_loss.item())
                        pbar_valid.set_postfix(loss=f"{v_loss.item():.4f}")
                
                LOSS_valid = sum(valid_losses) / len(valid_losses)
                print('epoch=%06d  valid_loss=%.6f' % (epoch, LOSS_valid))
                model.train()
        except Exception as e:
            print(f"Validation skipped or failed: {e}")

        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scheduler.step()

        # Save model checkpoint
        os.makedirs(checkpoint_path, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_path, f"model_epoch_{epoch:04d}.pth")
        torch.save(model.state_dict(), checkpoint_file)
        print(f"Saved checkpoint to {checkpoint_file}")

        #torch.cuda.empty_cache()

def main(device=['cpu','cuda'][torch.cuda.is_available()]):
    #infer(image_size=512, image_path=['/data/image/test/images/Lion.png'][0], output_file='./outs/stereo/test/Lion.obj', remove_bg=True, foreground_ratio=0.85, render_video=True, device=device)
    train(image_size=512, batch_size=16, epochs=10, checkpoint_path='/data/ckpt/', best_checkpoint_file='/data/ckpt/checkpoint.pth', device=device)

if __name__ == '__main__':  #cls; python -Bu superv.py
    main()

# wget https://huggingface.co/facebook/dino-vitb16/tree/main/*  ./ckpt/dino-vitb16/*
# wget https://huggingface.co/stabilityai/TripoSR/blob/main/* ./ckpt/TripoSR/*

