
import os
import torch
import torch.nn as nn
import sys
# Add DSMNet path
dsmnet_path = "./DSMNet-master/"
sys.path.insert(0, dsmnet_path)

import yaml
from models.DSMNet2x2 import DSMNet
from torch.autograd import Variable
#import skimage
#import skimage.io
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import cv2
import argparse
import utils as ut
import time


def test_transform(temp_data, crop_height, crop_width):
    _, h, w=np.shape(temp_data)

    if h <= crop_height and w <= crop_width:
        temp = temp_data
        temp_data = np.zeros([6, crop_height, crop_width], 'float32')
        temp_data[:, crop_height - h: crop_height, crop_width - w: crop_width] = temp
    else:
        start_x = int((w - crop_width) / 2)
        start_y = int((h - crop_height) / 2)
        temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]
    left = np.ones([1, 3,crop_height,crop_width],'float32')
    left[0, :, :, :] = temp_data[0: 3, :, :]
    right = np.ones([1, 3, crop_height, crop_width], 'float32')
    right[0, :, :, :] = temp_data[3: 6, :, :]
    return torch.from_numpy(left).float(), torch.from_numpy(right).float(), h, w

def load_data(leftname, rightname):
    left = Image.open(leftname)
    right = Image.open(rightname)
    size = np.shape(left)
    height = size[0]
    width = size[1]
#    opt.crop_height = int(height/48.)*48
#    opt.crop_width = int(width/48.)*48
    temp_data = np.zeros([6, height, width], 'float32')
    left = np.asarray(left)
    right = np.asarray(right)
    if len(size)>2:
        r = left[:, :, 0]
        g = left[:, :, 1]
        b = left[:, :, 2]
    else:
        r = left[:, :]
        g = left[:, :]
        b = left[:, :]
    temp_data[0, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[1, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[2, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    if len(size)>2:
        r = right[:, :, 0]
        g = right[:, :, 1]
        b = right[:, :, 2]	
    else:
        r = right[:, :]
        g = right[:, :]
        b = right[:, :]	
    #r,g,b,_ = right.split()
    temp_data[3, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[4, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[5, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    return temp_data


def test(model, leftname, rightname, crop_height, crop_width, use_cuda):
  #  count=0
    
    input1, input2, height, width = test_transform(load_data(leftname, rightname), crop_height, crop_width)

    
    input1 = Variable(input1, requires_grad = False)
    input2 = Variable(input2, requires_grad = False)

    model.eval()

    if use_cuda:
        input1 = input1.cuda()
        input2 = input2.cuda()
    
    with torch.no_grad():
        #print("Inputs:", input1.shape, input2.shape)
        # warm up
        prediction = model(input1, input2)
        
        # start_t = time.time()
        # prediction = model(input1, input2)
        # print('Elapsed time:', time.time()-start_t)
        
        prediction = prediction.detach().cpu().numpy()
        #print("Prediction:", prediction.shape)
        if height <= crop_height and width <= crop_width:
            prediction = prediction[0, crop_height - height: crop_height, crop_width - width: crop_width]
        else:
            prediction = prediction[0, :, :]
        return prediction


def load_DSMNet_model(model_path, max_disp, use_cuda):
    
    model = DSMNet(max_disp)
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()


    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage.cuda())
        msg=model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("=> loaded checkpoint '{}'".format(model_path))
        print(msg)
    else:
        print("=> no checkpoint found at '{}'".format(model_path))
        raise Exception('No model found!')
    return model


### Run DSNNet on the europa sim images to produce disparity maps

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description=".")
    parser.add_argument('--dataset', type=str, default='data/dataset/terrain_variation/')
    parser.add_argument('--model_path', type=str, default='DSMNet-master/weights/mixed_epoch_6.pth')
    parser.add_argument('--max_disp', dest='max_disp', type=int, default=192)
    parser.add_argument('--crop_height', dest='crop_height', type=int, default=384)
    parser.add_argument('--crop_width', dest='crop_width', type=int, default=640)
    parser.add_argument('--demo', default=False, action='store_true', help='Run single frame inference')
    parser.add_argument('--save_images', default=False, action='store_true')

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()

    model = load_DSMNet_model(args.model_path, args.max_disp, use_cuda)


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
                                                                    ut.get_demo_pair(model_name='DSMNet', imgs_dir=imgs_dir, view_id=0)

    else:
        ## Run inference on a dataset ##
        data_path = "../" + args.dataset
        img0_list, img1_list, depth0_gt_list, save_disp_list, save_depth_list, save_depth_list_npy = \
                                                                    ut.get_dataset(data_path, model_name='DSMNet')

    
    print('Dataset size:', len(img0_list))

    ### Run inference and evaluation ###

    metrics = {'rmse':[], 'si_rmse':[], 'l1_error':[], 'l1_error_rate_10':[], 'l1_error_rate_30':[], 'sdr_error':[]}

    for i in range(len(img0_list)):
        img0_path = img0_list[i]
        img1_path = img1_list[i]
        depth0_gt_path = depth0_gt_list[i]
        savename_disp = save_disp_list[i]
        savename_depth = save_depth_list[i]
        savename_depth_npy = save_depth_list_npy[i]

        prediction = test(model, img0_path, img1_path, args.crop_height, args.crop_width, use_cuda) # disparity prediction

        # Convert disparity to depth
        depth = camera_baseline*camera_focal_length / prediction
        
        # store the predicted depth
        with open(savename_depth_npy, "wb") as f:
            np.save(savename_depth_npy, depth)	
        f.close()

        # Load gt depth
        depth0_gt = np.load(depth0_gt_path)

        # crop depth_gt to match the output size of DSMNet
        cut = int((depth0_gt.shape[0] - args.crop_height)/2)
        depth0_gt = depth0_gt[cut:-cut, :]

        if args.save_images:
            # Save absolute and normalized values
            plt.imsave(savename_disp, prediction, cmap='jet')
            prediction_vis = prediction / np.amax(prediction)
            cv2.imwrite(savename_disp.split('.png')[0]+"_norm.png", prediction_vis*256.0)

            # remove sky points not containing depth values
            inds = np.where(depth0_gt==0)
            depth[inds] = 0.0

            plt.imsave(savename_depth, depth, cmap='jet')
            depth_vis = depth / np.amax(depth)
            cv2.imwrite(savename_depth.split('.png')[0]+"_norm.png", depth_vis*256.0)

            plt.imsave(savename_depth.split('.png')[0]+"_gt.png", depth0_gt, cmap='jet')


        if args.demo:
            ut.plot_imgs(depth0_gt, depth, title1='GT', title2='Pred')


        ##### Quantitative evaluation over depth values (not disparity)

        ut.calc_metrics(metrics, disp_pred=prediction, depth0_pred=depth, depth0_gt=depth0_gt, max_disp=args.max_disp) # populated metrics dict

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