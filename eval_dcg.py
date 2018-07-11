import numpy as np 


def tuple2list(tl):
    return [ x[-1] for x in tl ]

def tuple2array(ta):
    return [ tuple2list(tl) for tl in ta ]

def get_rank(yr,k=None):
    """
    Args: 
        yr should be a ndarray with dimension n x m containing positive real values within [0,1]
            > n being the number of testcases
            > m being the global value of the number of all diagnosis
            >>> y_pred = [[0.05,0.77,0.30,0.01,-1,...],[0.13,0.99,0.01,0.00,-1,...],...
        k should be a postive integer indicating using the first k highest probabilities into account
            >>> default : None 
    Returns:
        Matrix w/ Ranked Index
    """
    if k == None:
        return np.asarray([x[::-1]+1 for x in np.argsort(yr)])
    return np.asarray([x[::-1][:k]+1 for x in np.argsort(yr)])

def get_relevance_score(vt,vr,rk,normalize=False):
    """
    Args:
        vt should be an 1-D ndarray containing the ranked ground truth diagnosis 
            >>> vt = [7,1,4,6,2,3]

        vr should be an 1-D ndarray containing the ranked predict diagnosis 
            >>> vr = [3,4,1,2]

        rk should be a 1-D ndarray with entities containing positive real values 
            >>> rk = [1,0.5,0.2,0.1]

        normalize should be a boolean indicating whether to caculate normalized DCG or not

    Returns
        Discounted Cumulative Gain Score
    """
    r = []
    for i in range(vr.shape[0]):
        try:
            r.append(rk[vt.tolist().index(vr[i])])
        except ValueError:
            r.append(0)
    if normalize:
        return np.sum(np.asarray(r)/np.log2(np.arange(2,np.asarray(r).size+2)))/np.sum(np.asarray(sorted(rk,reverse=True))/np.log2(np.arange(2,np.asarray(r).size+2)))
    return np.sum(np.asarray(r)/np.log2(np.arange(2,np.asarray(r).size+2)))

def get_dcg_score(yt,yr,rk=None,k=None,normalize=False):

    """
    Score is Discounted Cumulative Gain (DCG) 
    
    Args:
        yt should be a ndarray with dimension n x d containing index values
            > n being the number of testcases
            > d being the max(number of labeled ground truth diagnosis) of all testcases
            >>> y_true = [[1,3,5,6],[2,3,9,13],[7,13,15],...]

        yr should be a ndarray with dimension n x m containing positive real values within [0,1]
            > n being the number of testcases
            > m being the global value of the number of all diagnosis
            >>> y_pred = [[(2,0.05),(1,0.77),(5,0.30),(3,0.01),(10,-1),...],[(9,0.13),(7,0.99),(1,0.01),(2,0.00),(2307,-1),...],...

        rk should be a 1-D ndarray with entities containing positive real values 
            >>> default : np.linspace(1,0,k) 

        k should be a postive integer indicating using the first k highest probabilities into account
            >>> default : None 

    Returns:
        Average Discounted Cumulative Gain (in float)
    """
    yr = tuple2array(yr)
    if yt.shape[0] != yr.shape[0]:
        raise ValueError('Testcases mismatch between truth and test')
    if rk.any() == None:
        raise Warning('No ranking metrices was given')
        if k == None:
            rk = np.asarray([1])
        else:
            rk = np.linspace(1,0,k)
    yr = get_rank(yr,k)
    print('prdicted ranked index mertrix:')
    print(yr)
    print([get_relevance_score(yt[i],yr[i],rk,normalize=normalize) for i in range(yt.shape[0])])
    return np.average([get_relevance_score(yt[i],yr[i],rk,normalize=normalize) for i in range(yt.shape[0])])

def micro():
    
    return None 

# def demo():

#     print('==============================demo==============================')
#     rk = np.asarray([3,2,1])
#     print('ranked weights:',rk)
#     yt = np.asarray([[1,2,5],[3,4,1],[1,2,3]])
#     print('true ranked index metrix:')
#     print(yt)
#     print('=>testcase:',yt.shape[0])
#     print('=>k:',yt.shape[1])
#     yr = np.asarray([[0.99,0.94,0.43,0.10,0.00,0.00],[0.12,0.33,0.99,0.4,0.02,0.53],[1.00,0.99,0.98,0.05,0.02,0.01]])
#     print('predicted ranked index metrix:')
#     print(yr)
#     print('=>testcase:',yr.shape[0])
#     print('=>total diagonsis:',yr.shape[1])

#     print('DCG:',get_dcg_score(yt,yr,rk,3,False))
#     print('nDCG:',get_dcg_score(yt,yr,rk,3,True))
