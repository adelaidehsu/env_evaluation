from __future__ import division
import numpy as np
import operator

def compute_MA(vt, vr, set_possible_ids):
    """
    Compute Mean Average Precision/Recall/Specificity of each testcase using voc07 standard
    Args:
        vt should be an 1-D ndarray containing the ranked ground truth disids
            >>> vt = [7,1,4,6,2,3]             
        vr should be an 1-D ndarray containing the ranked predicted disids 
            >>> vr = [3,4,1,2]
        possible_ids should be a set containing all known possible disids

    Returns:
    Mean Average Precision, Mean Average Recall, Mean Average Specificity (all in float)
    """
    result = np.asarray([[],[],[],[]])
    for top_n in range(vr.shape[0]):
        consider_vr = vr[0:top_n+1]
        tp = set(vt).intersection(set(consider_vr))
        fp = (set_possible_ids - set(vt)).intersection(set(consider_vr))
        tn = (set_possible_ids-set(vt)).intersection(set_possible_ids-set(consider_vr))
        fn = set(vt).intersection(set_possible_ids-set(consider_vr))
        c_precision = len(tp)/(len(tp)+len(fp)) if len(tp)+len(fp)!=0 else 0
        c_recall = len(tp)/(len(tp)+len(fn)) if len(tp)+len(fn)!=0 else 0
        c_spec = len(tn)/(len(tn)+len(fp)) if len(tn)+len(fp)!=0 else 0
        c_oppo = len(tn)/(len(tn)+len(fn)) if len(tn)+len(fn)!=0 else 0
        tmp_arr = np.asarray([[c_precision],[c_recall],[c_spec],[c_oppo]])
        result = np.append(result, tmp_arr, axis=1)
    a_result = np.asarray([[],[],[]])
    #MAP
    for r_th, p_th, o_th in zip(np.arange(0.,1.1,0.1),np.arange(0.,1.1,0.1),np.arange(0.,1.1,0.1)):
        max_p = -1
        max_r = -1
        max_s = -1
        for j in range(result[1].shape[0]):
            max_p = result[0][j] if result[1][j]>=r_th and result[0][j]>max_p else max_p
            max_r = result[1][j] if result[0][j]>=p_th and result[1][j]>max_r else max_r
            max_s = result[2][j] if result[3][j]>=o_th and result[2][j]>max_s else max_s
        max_p = 0 if max_p==-1 else max_p
        max_r = 0 if max_r==-1 else max_r
        max_s = 0 if max_s==-1 else max_s
        tmp_a = np.asarray([[max_p],[max_r],[max_s]])
        a_result = np.append(a_result, tmp_a, axis=1)
    return np.sum(a_result[0])/11, np.sum(a_result[1])/11, np.sum(a_result[2])/11


def MA_precision_recall_specifity_support(yt, yr, possible_ids, k=None):
    """
    Calculating Precision, Recall, Specificity, MAP, MAR, MAS scores
    Args:
        yt should be a ndarray with dimension n x d containing ground truth disids
            > n being the number of testcases
            > d being the max(number of labeled ground truth diagnosis) of all testcases
            >>> y_true = [[1,3,5,6],[2,3,9,13],[7,13,15],...]

        yr should be a ndarray of tuples (disid, confidence) with dimension n x m.
            > confidence should be the predicted confidence of the correspoding disid (real values)
            > n being the number of testcases
            > m being the global value of the number of all diagnosis
            >>> y_pred = [[(2,0.05),(1,0.77),(5,-0.30),(3,0.01),(10,-1),...],[(9,0.13),(7,0.99),(1,0.01),(2,0.00),(2307,-1),...],...
        
        possible_ids should be a list containing all known possible disids
        
        k should be a postive integer indicating taking the first k highest confidence into account
            >>> default : None 
    Returns:
        Precision, Recall, Specificity, MAP, MAR, MAS (all in float)
    """
    result = np.asarray([[],[],[],[],[],[]])
    for i in range(yr.shape[0]):
        tmp_list = list(yr[i])
        tmp_list.sort(key = operator.itemgetter(1), reverse = True)
        if k != None: 
            if k > yt[i].shape[0] or k > len(tmp_list):
                raise ValueError('Your trancate size is larger than y_true and y_pred lengths. Please check again.')
            processed_y_pred = np.asarray([int(tmp_list[j][0]) for j in range(k)])
            processed_y_true = yt[i][0:k]
        else:
            processed_y_pred = np.asarray([int(tmp_list[j][0]) for j in range(len(tmp_list))])
            processed_y_true = yt[i]
        #calculate TP, FP, TN, FN
        set_possible_ids = set(possible_ids)
        tp = set(processed_y_true).intersection(set(processed_y_pred))
        fp = (set_possible_ids - set(processed_y_true)).intersection(set(processed_y_pred))
        tn = (set_possible_ids - set(processed_y_true)).intersection(set_possible_ids-set(processed_y_pred))
        fn = set(processed_y_true).intersection(set_possible_ids-set(processed_y_pred))
        #preicsion, specificity, recall, map, mar, mas of each testcase
        c_precision = len(tp)/(len(tp)+len(fp)) if len(tp)+len(fp) != 0 else 0
        c_recall = len(tp)/(len(tp)+len(fn)) if len(tp)+len(fn) != 0 else 0
        c_spec = len(tn)/(len(tn)+len(fp)) if len(tn)+len(fp) != 0 else 0
        c_map, c_mar, c_mas = compute_MA(processed_y_true, processed_y_pred, set_possible_ids)
        tmp_arr = np.asarray([[c_precision],[c_recall],[c_spec],[c_map],[c_mar],[c_mas]])
        result = np.append(result, tmp_arr, axis = 1)

    precision = np.sum(result[0])/result[0].shape[0]
    recall = np.sum(result[1])/result[1].shape[0]
    specificity = np.sum(result[2])/result[2].shape[0]
    map = np.sum(result[3])/result[3].shape[0]
    mar = np.sum(result[4])/result[4].shape[0]
    mas = np.sum(result[5])/result[5].shape[0]
    return precision, recall, specificity, map, mar, mas


#Individual GET funtions for different score usage
def precision_score(yt, yr, possible_ids, possible_ids, k=None):
    p,_,_ ,_,_,_= MA_precision_recall_specifity_support(yt, yr, possible_ids, k)
    return p

def recall_score(yt, yr, possible_ids, k=None):
    _,r,_,_,_,_ = MA_precision_recall_specifity_support(yt, yr, possible_ids, k)
    return r

def specificity_score(yt, yr, possible_ids, k=None):
    _,_,s,_,_,_ = MA_precision_recall_specifity_support(yt, yr, possible_ids, k)
    return s

def map_score(yt, yr, possible_ids, k=None):
    _,_,_,map,_,_ = MA_precision_recall_specifity_support(yt, yr, possible_ids, k)
    return map

def mar_score(yt, yr, possible_ids, k=None):
    _,_,_,_,mar,_ = MA_precision_recall_specifity_support(yt, yr, possible_ids, k)
    return mar

def mas_score(yt, yr, possible_ids, k=None):
    _,_,_,_,_,mas = MA_precision_recall_specifity_support(yt, yr, possible_ids, k)
    return mas

"""
d_set = [i for i in range(1,11)]
yt = np.asarray([[1,2,3]])
yr = np.asarray([[(1,0.6),(2,0.3),(4,0.1)]])
p,r,s,map,mar,mas = MA_precision_recall_specifity_support(yt, yr, d_set)
print("precision: ", p)
print("recall: ", r)
print("specificity: ", s)
print("MAP: ", map)
print("MAR: ", mar)
print("MAS: ", mas)
"""
