
import numpy as np
import os
import json


####
# Load the results generated by gen_res.py and estimate stats for different subsets of the data
####


def gen_comb_subsets(baseline, results_list_all, params_list_all):

    # Get results over combinations of the data, e.g. a subset of the dataset types
    param_name = [ ['dataset_type'],
                    ['dataset_type']
                 ] 

    param_values = [    ['scene_reconstructions'],
                        ['texture_variation', 'gaea_texture_variation', 'generative_texture', 'terrain_variation', 'rocks', 'generative_texture_snow'],
                   ]

    subset_name = [ 'real',
                    'synthetic',
                  ]
    
    for i in range(len(param_name)):
        
        all_param_values_res = {}

        metrics_dict_mean = filter_res(results_list_all, params_list_all, param_name[i], param_values[i], comb=True)

        # convert list to string
        param_values_string = "".join( x+" " for x in param_values[i] )
        all_param_values_res[param_values_string] = metrics_dict_mean

        postfix = ""
        for p in param_name[i]:
            postfix += "_"
            postfix += p

        # file to hold results for every dataset subset
        results_save_file = "results/result_"+baseline+postfix+"_comb_"+subset_name[i]+".json"
        with open(results_save_file, "w") as outfile:
            json.dump(all_param_values_res, outfile, indent=4)



def filter_res(results_list_all, params_list_all, param_name, param_value, comb=False):

    
    metrics_dict = {'rmse':[], 'si_rmse':[], 'l1_error':[], 'l1_error_rate_10':[], 'l1_error_rate_30':[], 'l1_error_rate_100':[], 'sdr_error':[]}
    metrics_list = metrics_dict.keys()
    counter = 0

    for i in range(len(results_list_all)):

        params = params_list_all[i]

        if len(param_name)==1: # first indent
            value = params[param_name[0]]
        elif len(param_name)==2: # second indent
            if param_name[1] in params[param_name[0]].keys():
                value = params[param_name[0]][param_name[1]]
            else:
                continue

        if comb:
            # check whether the value is part of a larger subset
            if value in param_value:
                results = results_list_all[i]
                counter += 1
                for met in metrics_list:
                    metrics_dict[met].append(results[met])
        
        else:
            # check whether the value is identical to the one given
            if value==param_value:
                results = results_list_all[i]
                counter += 1
                for met in metrics_list:
                    metrics_dict[met].append(results[met])
    

    metrics_dict_mean = {'rmse_mean':0, 'si_rmse_mean':0, 'l1_error_mean':0, 'l1_error_rate_10_mean':0, 
                            'l1_error_rate_30_mean':0, 'l1_error_rate_100_mean':0, 'sdr_error_mean':0, 'data_size':counter}

    for met in metrics_list:
        metrics_dict_mean[met+'_mean'] = np.mean(np.asarray(metrics_dict[met])).astype(np.float64)
        #print("Mean", met, ":", np.mean(np.asarray(metrics[met])))
    return metrics_dict_mean


def get_overall_res(results_list_all):

    metrics_dict = {'rmse':[], 'si_rmse':[], 'l1_error':[], 'l1_error_rate_10':[], 'l1_error_rate_30':[], 'l1_error_rate_100':[], 'sdr_error':[]}

    metrics_dict_mean = {'rmse_mean':0, 'si_rmse_mean':0, 'l1_error_mean':0, 'l1_error_rate_10_mean':0, 
                            'l1_error_rate_30_mean':0, 'l1_error_rate_100_mean':0, 'sdr_error_mean':0, 'data_size':len(results_list_all)}

    for i in range(len(results_list_all)):
        
        for met in metrics_dict.keys():
            metrics_dict[met].append(results_list_all[i][met])

    for met in metrics_dict.keys():
        metrics_dict_mean[met+'_mean'] = np.mean(np.asarray(metrics_dict[met])).astype(np.float64)
    
    return metrics_dict_mean


if __name__ == '__main__':

    baseline_list = ["DSMNet", "IGEV", "StereoBM"]

    for baseline in baseline_list:

        results_file = "results/results_"+baseline+".npz"

        # results keys: ['depth_gt_list_all', 'depth_pred_list_all', 'params_list_all', 'results_list_all']
        results = np.load(results_file, allow_pickle=True)

        params_list_all = results['params_list_all']
        results_list_all = results['results_list_all']

        ## Get results over entire dataset ##
        overall_metrics_dict_mean = get_overall_res(results_list_all)
        overall_res = {'ALL': overall_metrics_dict_mean}
        results_save_file = "results/result_"+baseline+"_ALL.json"
        with open(results_save_file, "w") as outfile:
            json.dump(overall_res, outfile, indent=4)

        ## Get results separated over real and synthetic sets ##
        gen_comb_subsets(baseline, results_list_all, params_list_all)