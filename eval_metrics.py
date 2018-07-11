from __future__ import division
import numpy as np
import operator

def compute_MA(vt, vr, set_possible_ids):
    """
    Compute Mean Average Precision/Recall/Specificity of each testcase using voc07 standard

    Args:
        vt should be an 1-D ndarray containing the ranked ground truth diagnosis 
            >>> vt = [7,1,4,6,2,3]             
        vr should be an 1-D ndarray containing the ranked predicted diagnosis 
            >>> vr = [3,4,1,2]
        possible_ids should be a set containing all known possible chids

    Returns:
    Mean Average Precision, Mean Average Recall, Mean Average Specificity (all in float)
    """
    P=[]
    R=[]
    S=[]
    O=[]
    for top_n in range(len(vr)):
        consider_vr = vr[0:top_n+1]
        tp_list = list((set(vt).intersection(set(consider_vr))))
        fp_list = list(((set_possible_ids - set(vt)).intersection(set(consider_vr))))
        tn_list = list((set_possible_ids-set(vt)).intersection(set_possible_ids-set(consider_vr)))
        fn_list = list(set(vt).intersection(set_possible_ids-set(consider_vr)))
        c_precision = len(tp_list)/(len(tp_list)+len(fp_list))
        c_recall = len(tp_list)/(len(tp_list)+len(fn_list))
        c_spec = len(tn_list)/(len(tn_list)+len(fp_list))
        c_oppo = len(tn_list)/(len(tn_list)+len(fn_list))
        P.append(c_precision)
        R.append(c_recall)
        S.append(c_spec)
        O.append(c_oppo)

    ap_list = []
    ar_list = []
    as_list = []
    #MAP
    for r_th in np.arange(0.,1.1,0.1):
        max_p = -1
        for j in range(len(R)):
            if R[j] >= r_th and P[j] > max_p:
                max_p = P[j]
        if max_p==-1:
            ap_list.append(0)
        else:
            ap_list.append(max_p)
    #MAR
    for p_th in np.arange(0.,1.1,0.1):
        max_r = -1
        for j in range(len(P)):
            if P[j] >= p_th and R[j] > max_r:
                max_r = R[j]
        if max_r==-1:
            ar_list.append(0)
        else:
            ar_list.append(max_r)
    #MAS
    for o_th in np.arange(0.,1.1,0.1):
        max_s = -1
        for j in range(len(O)):
            if O[j] >= o_th and S[j] > max_s:
                max_s = S[j]
        if max_s==-1:
            as_list.append(0)
        else:
            as_list.append(max_s)

    return sum(ap_list)/11, sum(ar_list)/11, sum(as_list)/11


def MA_precision_recall_specifity_support(yt, yr, possible_ids, k=None):
    
    """
    Calculating Precision, Recall, Specificity, MAP, MAR, MAS scores
    
    Args:
        yt should be a ndarray with dimension n x d containing chids values
            > n being the number of testcases
            > d being the max(number of labeled ground truth diagnosis) of all testcases
            >>> y_true = [[1,3,5,6],[2,3,9,13],[7,13,15],...]

        yr should be a ndarray of tuples (id, confidence) with dimension n x m.
            > id should be predicted chids (positive integers)
            > confidence should be the predicted confidence of the correspoding disease (positive real values between [0,1])
            > n being the number of testcases
            > m being the global value of the number of all diagnosis
            >>> y_pred = [[(2,0.05),(1,0.77),(5,0.30),(3,0.01),(10,-1),...],[(9,0.13),(7,0.99),(1,0.01),(2,0.00),(2307,-1),...],...
        
        possible_ids should be a list containing all known possible disease ids
        
        k should be a postive integer indicating taking the first k highest probabilities into account
            >>> default : None 

    Returns:
        Precision, Recall, Specificity, MAP, MAR, MAS (all in float)
    """
    PPV = []
    REC = []
    SPC = []
    MAP = []
    MAR = []
    MAS = []
    cnt = 0
    for i in range(len(yr)):
        tmp_list = []
        #remove NONE(confidence = -1) from each testcase if there's any
        for j in range(len(yr[i])):
            if yr[i][j][1] != -1:
                tmp_list.append(yr[i][j])
        #rank y_pred
        tmp_list.sort(key = operator.itemgetter(1), reverse = True)

        processed_y_pred = []
        if k != None: 
            if k > len(yt[i]) or k > len(tmp_list):
                raise ValueError('Your trancate size is larger than y_true and y_pred lengths. Please check again.')
            for x in range(k):
                processed_y_pred.append(tmp_list[x][0])
            processed_y_true = yt[i][0:k]
        else:
            for x in range(len(tmp_list)):
                processed_y_pred.append(tmp_list[x][0])
            processed_y_true = yt[i]
        
        #calculate TP, FP, TN, FN id lists
        set_possible_ids = set(possible_ids)
        tp_list = list((set(processed_y_true).intersection(set(processed_y_pred))))
        fp_list = list(((set_possible_ids - set(processed_y_true)).intersection(set(processed_y_pred))))
        tn_list = list((set_possible_ids-set(processed_y_true)).intersection(set_possible_ids-set(processed_y_pred)))
        fn_list = list(set(processed_y_true).intersection(set_possible_ids-set(processed_y_pred)))
        
        #preicsion, specificity, recall, map, mar, mas of each testcase
        c_precision = len(tp_list)/(len(tp_list)+len(fp_list))
        c_recall = len(tp_list)/(len(tp_list)+len(fn_list))
        c_spec = len(tn_list)/(len(tn_list)+len(fp_list))
        c_map, c_mar, c_mas = compute_MA(processed_y_true, processed_y_pred, set_possible_ids)
        PPV.append(c_precision)
        REC.append(c_recall)
        SPC.append(c_spec)
        MAP.append(c_map)
        MAR.append(c_mar)
        MAS.append(c_mas)
        cnt+=1
  
    precision = sum(PPV)/cnt
    recall = sum(REC)/cnt
    specificity = sum(SPC)/cnt
    map = sum(MAP)/cnt
    mar = sum(MAR)/cnt
    mas = sum(MAS)/cnt
    
    return precision, recall, specificity, map, mar, mas


"""
Individual GET funtions for different score usage
"""
def precision_score(yt, yr, d_set, possible_ids, k=None):
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
yt = np.asarray([[1,2,3,4], [5,6,7,8]])
yr = np.asarray([[(1,0.7), (10,0.5), (4,0.3), (2,0.3), (12, -1), (8, -1)], [(5,0.3), (6,0.7), (7,0.9)]])
p,r,s,map,mar,mas = MA_precision_recall_specifity_support(yt, yr, d_set, 2)
print("precision: ", p)
print("recall: ", r)
print("specificity: ", s)
print("MAP: ", map)
print("MAR: ", mar)
print("MAS: ", mas)
"""
