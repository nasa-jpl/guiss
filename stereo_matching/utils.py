
import matplotlib.pyplot as plt
import numpy as np
import os
import evaluation_metrics as em


def plot_imgs(im1, im2, title1='im1', title2='im2'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12))
    ax1.imshow(im1, aspect='equal')
    ax2.imshow(im2, aspect='equal')
    ax1.set_title(title1)
    ax2.set_title(title2)
    plt.tight_layout()
    plt.draw()
    plt.show()


def get_demo_pair(model_name, imgs_dir, view_id):
    img0_list, img1_list, depth0_gt_list = [], [], []
    save_disp_list, save_depth_list, save_depth_list_npy = [], [], []

    img0_path = imgs_dir + "view_"+ "%04i"%view_id +"_left.png"
    img1_path = imgs_dir + "view_"+ "%04i"%view_id +"_right.png"
    depth0_gt_path = imgs_dir + "view_"+ "%04i"%view_id +"_left.npy"

    save_path = imgs_dir + model_name +"_predictions/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    savename_disp = save_path + "view_"+ "%04i"%view_id + "_disparity.png"
    savename_depth = save_path + "view_"+ "%04i"%view_id + "_depth.png"
    savename_depth_npy = save_path + "view_"+ "%04i"%view_id + "_depth.npy"

    img0_list.append(img0_path)
    img1_list.append(img1_path)
    depth0_gt_list.append(depth0_gt_path)
    save_disp_list.append(savename_disp)
    save_depth_list.append(savename_depth)
    save_depth_list_npy.append(savename_depth_npy)

    return img0_list, img1_list, depth0_gt_list, save_disp_list, save_depth_list, save_depth_list_npy


def get_dataset(data_path, model_name):
    img0_list, img1_list, depth0_gt_list = [], [], []
    save_disp_list, save_depth_list, save_depth_list_npy = [], [], []

    scene_dirs = os.listdir(data_path)
    
    scene_dirs = [ x for x in scene_dirs if os.path.isdir(data_path+x) ]
    scene_dirs.sort()

    for scene_name in scene_dirs:
        scene_instances = os.listdir(data_path+scene_name) # scene instance is the same scene rendered with different yaml params
        scene_instances.sort()

        for inst in scene_instances:
            site_dirs = os.listdir(data_path+scene_name+"/"+inst)
            site_dirs.sort()

            for site_name in site_dirs:
                
                imgs_dir = data_path + scene_name + "/" + inst + "/" + site_name + "/"
                
                file_list = os.listdir(imgs_dir)
                file_list = [ x for x in file_list if not os.path.isdir(imgs_dir+x) ]
                file_list.sort()

                file_list_img0 = [ imgs_dir+x for x in file_list if x.split('_')[-1]=="left.png" ]
                file_list_img1 = [ imgs_dir+x for x in file_list if x.split('_')[-1]=="right.png" ]
                file_list_depth0_gt = [ imgs_dir+x for x in file_list if x.split('_')[-1]=="left.npy" ]

                save_path = imgs_dir + model_name + "_predictions/"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                # create the save file names
                file_list_save_name_disp, file_list_save_name_depth = [], []
                file_list_save_name_depth_npy = []
                for x in file_list_img0:
                    base_name = x.split('/')[-1].split('_')[:-1]
                    base_name = '_'.join(base_name)
                    file_list_save_name_disp.append(save_path + base_name + "_disparity.png")
                    file_list_save_name_depth.append(save_path + base_name + "_depth.png")
                    file_list_save_name_depth_npy.append(save_path + base_name + "_depth.npy")

                img0_list += file_list_img0
                img1_list += file_list_img1
                depth0_gt_list += file_list_depth0_gt
                save_disp_list += file_list_save_name_disp
                save_depth_list += file_list_save_name_depth
                save_depth_list_npy += file_list_save_name_depth_npy
    
    return img0_list, img1_list, depth0_gt_list, save_disp_list, save_depth_list, save_depth_list_npy


def calc_metrics(metrics, disp_pred, depth0_pred, depth0_gt, max_disp):
    # Remove invalid disparity values
    mask = np.logical_and(disp_pred>=0.001, disp_pred<=max_disp)
    pred_depth_mask = depth0_pred[mask] # flat N
    gt_depth_mask = depth0_gt[mask] # flat N

    # Remove invalid depth values
    mask = gt_depth_mask>0.05 # 
    pred_depth_mask = pred_depth_mask[mask]
    gt_depth_mask = gt_depth_mask[mask]
    #print(pred_depth_mask.shape)

    ## Calculate metrics
    rmse = em.RMSE(pred_depth_mask, gt_depth_mask)
    si_rmse = em.si_RMSE(pred_depth_mask, gt_depth_mask)
    l1_error = em.L1_error(pred_depth_mask, gt_depth_mask)
    l1_error_rate_10 = em.L1_error_rate(pred_depth_mask, gt_depth_mask, threshold=0.1)
    l1_error_rate_30 = em.L1_error_rate(pred_depth_mask, gt_depth_mask, threshold=0.3)
    # Sample pairs from images to estimate SDR
    pairs = em.get_pairs(N=pred_depth_mask.shape[0])
    sdr_error = em.SDR(pred_depth_mask, gt_depth_mask, pairs)

    metrics['rmse'].append(rmse)
    metrics['si_rmse'].append(si_rmse)
    metrics['l1_error'].append(l1_error)
    metrics['l1_error_rate_10'].append(l1_error_rate_10)
    metrics['l1_error_rate_30'].append(l1_error_rate_30)
    metrics['sdr_error'].append(sdr_error)