
import numpy as np


# RMSE
def RMSE(pred, gt):
    return np.sqrt(np.mean((pred - gt)**2))

# Scale-invariant RMSE (si-RMSE from Megadepth paper)
# https://github.com/zhengqili/MegaDepth/blob/master/models/HG_model.py
# Uses ratios of pairs of depths which are preserved under scaling
# Difference between pairs of log-depths in the log-depth domain
def si_RMSE(pred, gt):
    N = pred.shape[0]
    log_diff = np.log(pred) - np.log(gt)
    s1 = np.sum(log_diff**2) / N
    s2 = np.sum(log_diff)**2 / (N*N)
    return np.sqrt(s1-s2)

# L1 dist
def L1_error(pred, gt):
    return np.mean(np.abs(pred - gt))

# L1 error rates for 10cm, 30cm
def L1_error_rate(pred, gt, threshold):
    return np.sum(np.abs(pred - gt) > threshold) / pred.shape[0]    


def get_pairs(N):
    # Returns a set of index pairs Mx2 to be used for SDR (indices correspond to flattened images)
    # N is number of pixels in the image
    idx0 = np.arange(5000, N-5000, step=100) # cut corners
    idx1 = idx0[::10]
    pairs_idx = np.array(np.meshgrid(idx0, idx1)).T.reshape(-1, 2)
    
    # Remove pairs with the same element
    non_dupl = np.where(pairs_idx[:,0]-pairs_idx[:,1]!=0)[0]
    pairs_idx = pairs_idx[non_dupl,:]
    
    # Removed duplicate pairs (i.e. swapped)
    pairs_idx = np.unique(np.sort(pairs_idx, axis=1).view(','.join([pairs_idx.dtype.char]*2))).view(pairs_idx.dtype).reshape(-1, 2)
    return pairs_idx
    

# Depth ordering - SfM Disagreement Rate (SDR) from Megadepth paper
# https://github.com/zhengqili/MegaDepth/blob/master/models/HG_model.py
# Measures the preservation of depth ordering
# We are using simulated gt depth so our delta can be almost 0 (no uncertainty from SfM)
def SDR(pred, gt, pairs, delta=0.01):

    pred_A = pred[pairs[:,0]]
    pred_B = pred[pairs[:,1]]
    gt_A = gt[pairs[:,0]]
    gt_B = gt[pairs[:,1]]

    pred_ratio = np.divide(pred_A, pred_B)
    gt_ratio = np.divide(gt_A, gt_B)

    pred_labels = np.zeros((pred_ratio.shape[0]))
    pred_labels[ pred_ratio > (1+delta) ] = 1
    pred_labels[ pred_ratio < (1-delta) ] = -1

    gt_labels = np.zeros((gt_ratio.shape[0]))
    gt_labels[ gt_ratio > (1+delta) ] = 1
    gt_labels[ gt_ratio < (1-delta) ] = -1

    diff = pred_labels - gt_labels
    diff[ diff!=0 ] = 1
    error_per = np.sum(diff) / pred_ratio.shape[0]
    return error_per