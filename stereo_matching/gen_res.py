

import os
import numpy as np
import json
import argparse
import evaluation_metrics as em


def collect_results_info(datapath, baseline):
    depth_gt_list, depth_pred_list, params_list = [], [], []

    scene_dirs = os.listdir(datapath)
    scene_dirs.sort()

    for scene_name in scene_dirs:
        scene_instances = os.listdir(datapath+scene_name) # scene instance is the same scene rendered with different yaml params
        scene_instances.sort()

        for inst in scene_instances:
            site_dirs = os.listdir(datapath+scene_name+"/"+inst)
            site_dirs.sort()

            # due to a mistake only the last site_dir hold the json with the params...
            params_file = datapath+scene_name+"/"+inst+"/"+site_dirs[-1]+"/params.json"
            with open(params_file) as f:
                params = json.load(f)

            for site_name in site_dirs:

                # get the gt depth
                imgs_dir = datapath + scene_name + "/" + inst + "/" + site_name + "/"                
                file_list = os.listdir(imgs_dir)
                file_list = [ x for x in file_list if not os.path.isdir(imgs_dir+x) ]
                file_list.sort()
                file_list_depth0_gt = [ imgs_dir+x for x in file_list if x.split('_')[-1]=="left.npy" ]
                
                # get the predicted depth
                pred_dir = imgs_dir + baseline + "_predictions/"
                file_list = os.listdir(pred_dir)
                file_list = [ x for x in file_list if not os.path.isdir(pred_dir+x) ]
                file_list.sort()
                file_list_depth0_pred = [ pred_dir+x for x in file_list if x.split('_')[-1]=="depth.npy" ]

                # create a copy of the params for each test example
                params_local = [params]*len(file_list_depth0_pred)

                depth_gt_list += file_list_depth0_gt
                depth_pred_list += file_list_depth0_pred
                params_list += params_local

                #print(len(params_list))

                
    return depth_gt_list, depth_pred_list, params_list


def calc_metrics_offline(depth_gt, depth_pred, max_depth=100.0, min_depth=0.05):

    # remove invalid depth predictions
    mask = np.logical_and(depth_pred>=0.01, depth_pred<=100.0)
    pred_depth_mask = depth_pred[mask] # flat N
    gt_depth_mask = depth_gt[mask] # flat N

    # remove invalid depth gt values
    mask = gt_depth_mask > min_depth 
    pred_depth_mask = pred_depth_mask[mask]
    gt_depth_mask = gt_depth_mask[mask]

    mask = gt_depth_mask < max_depth
    pred_depth_mask = pred_depth_mask[mask]
    gt_depth_mask = gt_depth_mask[mask]

    ## Calculate metrics
    rmse = em.RMSE(pred_depth_mask, gt_depth_mask)
    si_rmse = em.si_RMSE(pred_depth_mask, gt_depth_mask)
    l1_error = em.L1_error(pred_depth_mask, gt_depth_mask)
    l1_error_rate_10 = em.L1_error_rate(pred_depth_mask, gt_depth_mask, threshold=0.1)
    l1_error_rate_30 = em.L1_error_rate(pred_depth_mask, gt_depth_mask, threshold=0.3)
    l1_error_rate_100 = em.L1_error_rate(pred_depth_mask, gt_depth_mask, threshold=1.0)
    # Sample pairs from images to estimate SDR
    pairs = em.get_pairs(N=pred_depth_mask.shape[0])
    sdr_error = em.SDR(pred_depth_mask, gt_depth_mask, pairs)

    return {'rmse':rmse, 'si_rmse':si_rmse, 'l1_error':l1_error, 'l1_error_rate_10':l1_error_rate_10,
            'l1_error_rate_30':l1_error_rate_30, 'l1_error_rate_100':l1_error_rate_100, 'sdr_error':sdr_error}


########
# Generate the results from the stereo estimation and store in a single file
# along with the corresponsing rendering params for each example
########


if __name__ == '__main__':

    '''
    To estimate results over our entire generated dataset:
    $ python gen_res.py --dataset_list scene_reconstructions texture_variation gaea_texture_variation \
                                         generative_texture terrain_variation rocks generative_texture_snow
    '''

    parser = argparse.ArgumentParser(description=".")
    parser.add_argument('--baseline', type=str, default='IGEV', choices=['IGEV', "DSMNet", "StereoBM"])
    parser.add_argument('--dataset_list', nargs='+')
    parser.add_argument('--crop_height', dest='crop_height', type=int, default=384)
    parser.add_argument('--crop_width', dest='crop_width', type=int, default=640)
    parser.add_argument('--max_depth', dest='max_depth', type=float, default=100.0)
    parser.add_argument('--min_depth', dest='min_depth', type=float, default=0.05)

    args = parser.parse_args()


    # Collect all ground-truth depths, predicted depths and json files with params
    depth_gt_list_all, depth_pred_list_all, params_list_all = [], [], []
    results_list_all = []

    for dataset in args.dataset_list:

        datapath = "../data/dataset/" + dataset + "/"

        depth_gt_list, depth_pred_list, params_list = collect_results_info(datapath, args.baseline)

        depth_gt_list_all += depth_gt_list
        depth_pred_list_all += depth_pred_list
        params_list_all += params_list

    print("Size of dataset:", len(depth_gt_list_all))

    assert len(depth_gt_list_all) == len(depth_pred_list_all) == len(params_list_all), "Number of gt and preds don't match!"


    depth_gt_list_all = np.asarray(depth_gt_list_all)
    depth_pred_list_all = np.asarray(depth_pred_list_all)
    params_list_all = np.asarray(params_list_all)

    # Estimate metrics for each test example and store in list
    for i in range(len(depth_gt_list_all)):

        depth_gt_path = depth_gt_list_all[i]
        depth_pred_path = depth_pred_list_all[i]
        params_path = params_list_all[i]

        print("Generating", args.baseline, "res for", i,"/", len(depth_gt_list_all)-1)
        print(depth_gt_path, " / ", depth_pred_path)

        # Load gt depth
        depth_gt = np.load(depth_gt_path)
        # crop depth_gt to match the output size of DSMNet
        cut = int((depth_gt.shape[0] - args.crop_height)/2)
        depth_gt = depth_gt[cut:-cut, :]

        # Load pred depth
        depth_pred = np.load(depth_pred_path)

        if args.baseline=="StereoBM":
            # For StereoBM I need to crop along width because depending on numDisparities, stereoBM 
            # does not give values at the begining and end
            start_x, end_x = 125, 600
            depth_gt = depth_gt[:, start_x:end_x]
            depth_pred = depth_pred[cut:-cut, start_x:end_x]
        elif args.baseline=="IGEV":
            # For IGEV crop also prediction to match DSMNet size (for fair comparison)
            depth_pred = depth_pred[cut:-cut, :]


        # returns dict with results
        res = calc_metrics_offline(depth_gt, depth_pred, args.max_depth, args.min_depth)
        results_list_all.append(res)


    results_list_all = np.asarray(results_list_all)

    savefile = "results/results_"+args.baseline+"_tmp.npz"
    with open(savefile, "wb") as f:
        np.savez(savefile, depth_gt_list_all=depth_gt_list_all,
                        depth_pred_list_all=depth_pred_list_all,
                        params_list_all=params_list_all,
                        results_list_all=results_list_all)	
    f.close()

