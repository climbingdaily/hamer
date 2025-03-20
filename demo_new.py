from pathlib import Path
import torch
import argparse
import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

from vitpose_model import ViTPoseModel

import json
from typing import Dict, Optional

def bbox_containment_ratio(hand_bbox, arm_bbox):
    x1_hand, y1_hand, x2_hand, y2_hand = hand_bbox
    x1_arm, y1_arm, x2_arm, y2_arm = arm_bbox

    # 计算交集 bbox
    x1_inter = max(x1_hand, x1_arm)
    y1_inter = max(y1_hand, y1_arm)
    x2_inter = min(x2_hand, x2_arm)
    y2_inter = min(y2_hand, y2_arm)

    # 计算交集面积
    if x2_inter > x1_inter and y2_inter > y1_inter:
        area_inter = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    else:
        area_inter = 0

    # 计算手部 bbox 面积
    area_hand = (x2_hand - x1_hand) * (y2_hand - y1_hand)

    # 计算包含比例
    return area_inter / area_hand if area_hand > 0 else 0

def get_bbox_from_mask_numpy(mask):
    mask_2d = mask.squeeze()

    ys, xs = np.where(mask_2d > 0)

    if len(xs) == 0 or len(ys) == 0:
        # print("Mask 为空！")
        return []

    # 计算 BBox
    x1, y1 = np.min(xs), np.min(ys)
    x2, y2 = np.max(xs), np.max(ys)

    return [x1, y1, x2, y2]

def get_bbobx_from_mask(mask_file, rotate_image=True):
    bbox = {}
    bbox_left = {}
    bbox_right = {}
    if os.path.exists(mask_file):
        with open(mask_file, 'rb') as f:
            mask_dict = pickle.load(f)
        if rotate_image:
            for k, v in mask_dict.items():
                v[1] = cv2.rotate(v[1][0].astype(np.uint8), cv2.ROTATE_90_CLOCKWISE)[None, ...] # left hand 
                v[2] = cv2.rotate(v[2][0].astype(np.uint8), cv2.ROTATE_90_CLOCKWISE)[None, ...] # right hand
        for k, v in mask_dict.items():
            bbox[k] = get_bbox_from_mask_numpy(v[1][0] + v[2][0])
            bbox_left[k] = get_bbox_from_mask_numpy(v[1][0])
            bbox_right[k] = get_bbox_from_mask_numpy(v[2][0])
    return bbox, bbox_left, bbox_right

def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='/home/guest/Documents/Nymeria/20231222_s1_kenneth_fischer_act7_56uvqd/recording_head/imgs_1049_1990', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--body_detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime and reduces memory')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')

    args = parser.parse_args()
    if not os.path.exists(args.img_folder):
        args.img_folder = os.path.join(
            str(Path(__file__).resolve().parents[1]), 
            'data', args.img_folder, 'build', 'image')
        print('No images found in the specified folder.')
        print(f'Images will be searched in {args.img_folder}')
    
    mask_file = os.path.join(os.path.dirname(args.img_folder), f"mask_{os.path.basename(args.img_folder)}.pkl")
    bboxes_mask, bbx_left, bbx_right = get_bbobx_from_mask(mask_file)

    # Download and load checkpoints 
    download_models(CACHE_DIR_HAMER)
    cp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.checkpoint))
    if not os.path.exists(cp_path):
        print(f'Checkpoint {cp_path} not found. Please download it first.')
        exit(1)
    model, model_cfg = load_hamer(cp_path)

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # keypoint detector
    cpm = ViTPoseModel(device)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    # Get all demo images ends with .jpg or .png
    img_paths = [img for end in args.file_type for img in Path(args.img_folder).glob(end)]
    img_paths.sort(key=lambda p: float(p.stem))

    # Iterate over all images in folder
    for i, img_path in enumerate(tqdm(img_paths, desc="Processing Images")):
        img_cv2 = cv2.imread(str(img_path))
        img = img_cv2.copy()[:, :, ::-1]
        bboxes = bboxes_mask[img_path.name]
        bl = bbx_left[img_path.name]
        br = bbx_right[img_path.name]

        # Detect human keypoints for each person
        if len(bboxes) > 0:
            vitposes_out = cpm.predict_pose(
                img,
                [np.concatenate([[bboxes], [[0.999]]], axis=1)],
            )
        else:
            continue

        bboxes = []
        is_right = []

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Rejecting not confident detections
            keyp = left_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                if bbox_containment_ratio(bbox, bl) > 0.5:
                    bboxes.append(bbox)
                    is_right.append(0)
            keyp = right_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                if bbox_containment_ratio(bbox, br) > 0.5:
                    bboxes.append(bbox)
                    is_right.append(1)

        if len(bboxes) == 0:
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        
        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            multiplier = (2*batch['right']-1)
            pred_cam = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            multiplier = (2*batch['right']-1)
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            # Render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                # Get filename from path img_path
                img_fn, _ = os.path.splitext(os.path.basename(img_path))
                person_id = int(batch['personid'][n])
                white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
                input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                input_patch = input_patch.permute(1,2,0).numpy()

                regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                        out['pred_cam_t'][n].detach().cpu().numpy(),
                                        batch['img'][n],
                                        mesh_base_color=LIGHT_BLUE,
                                        scene_bg_color=(1, 1, 1),
                                        )

                if args.side_view:
                    side_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                            out['pred_cam_t'][n].detach().cpu().numpy(),
                                            white_img,
                                            mesh_base_color=LIGHT_BLUE,
                                            scene_bg_color=(1, 1, 1),
                                            side_view=True)
                    final_img = np.concatenate([input_patch, regression_img, side_img], axis=1)
                else:
                    final_img = np.concatenate([input_patch, regression_img], axis=1)

                # cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_{person_id}.png'), 255*final_img[:, :, ::-1])

                # Add all verts and cams to list
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                is_right = batch['right'][n].cpu().numpy()
                verts[:,0] = (2*is_right-1)*verts[:,0]
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)

                # Save all meshes to disk
                if args.save_mesh:
                    camera_translation = cam_t.copy()
                    tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE, is_right=is_right)
                    tmesh.export(os.path.join(args.out_folder, f'{img_fn}_{person_id}.obj'))

        # Render front view
        if args.full_frame and len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

            # Overlay image
            input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

            cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_all.jpg'), 255*input_img_overlay[:, :, ::-1])

def draw_bbox(image_path, bboxes):
    # 读取图片
    image = cv2.imread(image_path)
    
    # 确保图片正确加载
    if image is None:
        print(f"Error: 无法加载图片 {image_path}")
        return

    # 解析 bbox 并转换为整数
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)

        # 绘制矩形框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 显示图片
    cv2.imshow("Image with BBox", image)

if __name__ == '__main__':
    main()
