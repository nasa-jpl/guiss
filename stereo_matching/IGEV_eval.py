
import sys
igev_path = "./IGEV-Stereo/"
sys.path.insert(0, igev_path)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import glob
import yaml
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from core.igev_stereo import IGEVStereo
from core.utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import os
import cv2
import utils as ut
import time


def load_image(imfile, use_cuda):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    if img.shape[2]>3:
        img = img[:,:,:3]
        #img = img[:,:,::-1]
        #cv2.imwrite('img3.png', img[:,:,::-1])
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    img = img[None]
    if use_cuda:
        img = img.to('cuda')
    #return img[None].to(DEVICE)
    return img

def load_IGEV_model(args, use_cuda):
    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    if use_cuda:
        model.to('cuda')
    model.eval()
    return model


def run_inference(model, img0_path, img1_path, valid_iters, use_cuda):
    with torch.no_grad():
        image1 = load_image(img0_path, use_cuda)
        image2 = load_image(img1_path, use_cuda)
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)
        #print("Image1:", image1.shape)
        #print(image2.shape)
        # warm up
        disp = model(image1, image2, iters=args.valid_iters, test_mode=True)
        
        # start_t = time.time()
        # disp = model(image1, image2, iters=args.valid_iters, test_mode=True)
        # print('Elapsed time:', time.time()-start_t)
        
        disp = disp.cpu().numpy()
        #print(disp.shape)
        disp = padder.unpad(disp)
        #print(disp.shape)
        return disp.squeeze().squeeze()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='data/dataset/terrain_variation/')
    parser.add_argument('--crop_height', dest='crop_height', type=int, default=384)
    parser.add_argument('--crop_width', dest='crop_width', type=int, default=640)
    parser.add_argument('--demo', default=False, action='store_true', help='Run single frame inference')
    parser.add_argument('--save_images', default=False, action='store_true')

    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./IGEV-Stereo/pretrained_models/sceneflow/sceneflow.pth')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=256, help='number of flow-field updates during forward pass') # 32

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=256, help="max disp of geometry encoding volume") # 192
    
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    #Path(args.output_directory).mkdir(exist_ok=True, parents=True)

    model = load_IGEV_model(args, use_cuda)


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
                                                                    ut.get_demo_pair(model_name='IGEV', imgs_dir=imgs_dir, view_id=0)
    
    else:
        ## Run inference on a dataset ##
        data_path = "../" + args.dataset
        img0_list, img1_list, depth0_gt_list, save_disp_list, save_depth_list, save_depth_list_npy = \
                                                                    ut.get_dataset(data_path, model_name='IGEV')

    print('Dataset size:', len(img0_list))


    metrics = {'rmse':[], 'si_rmse':[], 'l1_error':[], 'l1_error_rate_10':[], 'l1_error_rate_30':[], 'sdr_error':[]}

    for i in range(len(img0_list)):
        img0_path = img0_list[i]
        img1_path = img1_list[i]
        depth0_gt_path = depth0_gt_list[i]
        savename_disp = save_disp_list[i]
        savename_depth = save_depth_list[i]
        savename_depth_npy = save_depth_list_npy[i]

        prediction = run_inference(model, img0_path, img1_path, args.valid_iters, use_cuda)
        
        # Convert disparity to depth
        depth = camera_baseline*camera_focal_length / prediction

        # store the predicted depth
        with open(savename_depth_npy, "wb") as f:
            np.save(savename_depth_npy, depth)	
        f.close()


        # Load gt depth
        depth0_gt = np.load(depth0_gt_path)

        # Crop prediction and ground-truth to match DSMNet size (for fair comparison)
        cut = int((depth0_gt.shape[0] - args.crop_height)/2)
        prediction = prediction[cut:-cut, :]
        depth = depth[cut:-cut, :]
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