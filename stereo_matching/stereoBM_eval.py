
import cv2
import numpy as np
import yaml
from matplotlib import pyplot as plt
import os
import json
import sys
import argparse
import utils as ut
import time


def run_stereo(stereo, img0_path, img1_path, param_dict):
    imgL = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    #print(imgL.shape)

    # Setting the updated parameters before computing disparity map
    stereo.setNumDisparities(param_dict['numDisparities'])
    stereo.setBlockSize(param_dict['blockSize'])
    stereo.setPreFilterType(param_dict['preFilterType'])
    stereo.setPreFilterSize(param_dict['preFilterSize'])
    stereo.setPreFilterCap(param_dict['preFilterCap'])
    stereo.setTextureThreshold(param_dict['textureThreshold'])
    stereo.setUniquenessRatio(param_dict['uniquenessRatio'])
    stereo.setSpeckleRange(param_dict['speckleRange'])
    stereo.setSpeckleWindowSize(param_dict['speckleWindowSize'])
    stereo.setDisp12MaxDiff(param_dict['disp12MaxDiff'])
    stereo.setMinDisparity(param_dict['minDisparity'])
 
    # Calculating disparity using the StereoBM algorithm
    #start_t = time.time()
    disparity = stereo.compute(imgL,imgR)
    #print('Elapsed time:', time.time()-start_t)
    
    # NOTE: Code returns a 16bit signed single channel image,
    # CV_16S containing a disparity map scaled by 16. Hence it 
    # is essential to convert it to CV_32F and scale it down 16 times.
 
    # Converting to float32 
    disparity = disparity.astype(np.float32)

    #disparity = (disparity/16.0 - minDisparity)/numDisparities
    disparity = disparity/16.0

    disparity += 1e-4 # to avoid division with zero

    return disparity



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=".")
    parser.add_argument('--dataset', type=str, default='data/dataset/terrain_variation/')
    parser.add_argument('--crop_height', dest='crop_height', type=int, default=384)
    parser.add_argument('--crop_width', dest='crop_width', type=int, default=640)
    parser.add_argument('--demo', default=False, action='store_true', help='Run single frame inference')
    parser.add_argument('--save_images', default=False, action='store_true')

    args = parser.parse_args()

    options = yaml.safe_load(open('../configs/common.yaml', 'r'))
    camera_baseline = options['camera_baseline']*10 # 0.25 # found empirically that it needs to be scaled by 10
    camera_focal_length = options['camera_focal_length'] # 32 millimeters
    print("Camera focal length:", camera_focal_length, "millimeters")
    print("Camera baseline:", camera_baseline)

    ### Load data info ###

    if args.demo:
        ## Run inference on a single stereo image ##
        imgs_dir = '../data/example_pair/'
        img0_list, img1_list, depth0_gt_list, save_disp_list, save_depth_list, save_depth_list_npy = \
                                                                    ut.get_demo_pair(model_name='StereoBM', imgs_dir=imgs_dir, view_id=0)

    else:
        ## Run inference on a dataset ##
        data_path = "../" + args.dataset
        img0_list, img1_list, depth0_gt_list, save_disp_list, save_depth_list, save_depth_list_npy = \
                                                                    ut.get_dataset(data_path, model_name='StereoBM')

    print('Dataset size:', len(img0_list))

    ### Run inference and evaluation ###
    stereo = cv2.StereoBM_create()
    # The stereoBM param values were found empirically using stereoBM_demo.py
    params_dict = {'numDisparities':96, 
                   'blockSize':49,
                   'preFilterType': 1,
                   'preFilterSize': 17,
                   'preFilterCap':20,
                   'textureThreshold':18,
                   'uniquenessRatio':0,
                   'speckleRange':9,
                   'speckleWindowSize':0,
                   'disp12MaxDiff':25,
                   'minDisparity':4}

    metrics = {'rmse':[], 'si_rmse':[], 'l1_error':[], 'l1_error_rate_10':[], 'l1_error_rate_30':[], 'sdr_error':[]}

    for i in range(len(img0_list)):
        img0_path = img0_list[i]
        img1_path = img1_list[i]
        depth0_gt_path = depth0_gt_list[i]
        savename_disp = save_disp_list[i]
        savename_depth = save_depth_list[i]
        savename_depth_npy = save_depth_list_npy[i]

        prediction = run_stereo(stereo, img0_path, img1_path, params_dict)
        #print("DISP:", prediction)

        # Convert disparity to depth
        depth = camera_baseline*camera_focal_length / prediction
        #print("DEPTH:", depth)

        # store the predicted depth
        with open(savename_depth_npy, "wb") as f:
            np.save(savename_depth_npy, depth)	
        f.close()


        # Load gt depth
        depth0_gt = np.load(depth0_gt_path)

        # crop depth_gt to match the output size of DSMNet
        cut = int((depth0_gt.shape[0] - args.crop_height)/2)
        # also crop along width because depending on numDisparities, stereoBM 
        # does not give values at the begining and end
        start_x, end_x = 125, 600
        
        prediction = prediction[cut:-cut, start_x:end_x]
        depth = depth[cut:-cut, start_x:end_x]
        depth0_gt = depth0_gt[cut:-cut, start_x:end_x]

        if args.save_images:
            # Save absolute and normalized values
            plt.imsave(savename_disp, prediction, cmap='jet')
            prediction_vis = prediction / np.amax(prediction)
            cv2.imwrite(savename_disp.split('.png')[0]+"_norm.png", prediction_vis*256.0)

            # remove sky points not containing depth values
            inds = np.where(depth0_gt==0)
            depth[inds] = 0.0

            #inds = np.where(depth==np.amax(depth)) # for visualization purposes
            #depth[inds] = 0.0

            plt.imsave(savename_depth, depth, cmap='jet')
            depth_vis = depth / np.amax(depth)
            cv2.imwrite(savename_depth.split('.png')[0]+"_norm.png", depth_vis*256.0)

            plt.imsave(savename_depth.split('.png')[0]+"_gt.png", depth0_gt, cmap='jet')


        if args.demo:
            ut.plot_imgs(depth0_gt, depth, title1='GT', title2='Pred')

        ##### Quantitative evaluation over depth values (not disparity)

        ut.calc_metrics(metrics, disp_pred=prediction, depth0_pred=depth, depth0_gt=depth0_gt, max_disp=params_dict['numDisparities']) # populated metrics dict

        print("Stereo frame", i, img0_path)
        print("RMSE:", metrics['rmse'][i])
        print("si RMSE:", metrics['si_rmse'][i])
        print("L1 error:", metrics['l1_error'][i])
        print("L1 error rate 10:", metrics['l1_error_rate_10'][i])
        print("L1 error rate 30:", metrics['l1_error_rate_30'][i])
        print("SDR:", metrics['sdr_error'][i])
        print()


    ## Average results over dataset
    print("Overall Results")
    for met in metrics.keys():
        print("Mean", met, ":", np.mean(np.asarray(metrics[met])))