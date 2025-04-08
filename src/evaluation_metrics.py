import numpy as np
from scipy.special import comb
from scipy.stats import iqr
from scipy.stats import median_abs_deviation


def monotonicity(s, smaller_is_stronger=False):
    """
    expects a 1D array of scores sorted by biological perturbation strength! 
    """
    #TODO round to 2 significant decimals here
    L = len(s)
    result = 0
    for l in range(L):
        if smaller_is_stronger:
            result += np.sum(s[l+1:] <= s[l]) 
        else:
            result += np.sum(s[l+1:] >= s[l]) 
    return result / comb(L, 2)


def SSNR(s_0, s, smaller_is_stronger=False):
    """
    expects:
        - a 1D array of scores s_0 with length L, retrieved for testing each of the L groups against reference
        - a 2D array of scores s, with dimensions L x |P|, where each s[l] contains scores for comparison of disjunct
        subsets within the group l
        - the 1D array needs to be in the same order as the layers of the 3D array!
    """
    # TODO round to 2 significant decimals here
    assert len(s_0) == s.shape[0]
    L = len(s_0)
    result = 0
    for l in range(L):
        if smaller_is_stronger:
            result += np.sum(s[l] >= s_0[l]) 
        else:
            result += np.sum(s[l] <= s_0[l])
            
    return result / (np.prod(s.shape)) 
        

def robustness(s):
    """
    expects a 2D array of scores for comparison of reference and group l, each axis holds probability values to which each of the 2 groups were subsampled
    """
    assert s.shape[0] == s.shape[1]
    if np.max(s) != np.min(s):
        return 1 / median_abs_deviation((s - np.median(s)) / iqr(s), axis=None)
    return np.inf


def tests():
    print(monotonicity(np.array([1, 2, 3, 4, 5]), smaller_is_stronger=False)) #1.0
    print(monotonicity(np.array([1, 2, 5, 4, 5]), smaller_is_stronger=False)) #0.9
    print(monotonicity(np.array([1, 2, 3, 4, 5]), smaller_is_stronger=True)) #0.0
    
    s_0 = np.array([0.01, 0.02, 0.03])
    s = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    print(SSNR(s_0, s, smaller_is_stronger=True)) #1.0
    s = np.array([[0.4, 0.2, 1, 0.3], [0.05, 0.2, 1, 0.3], [0.01, 0.01, 0.01, 0.3]])
    print(SSNR(s_0, s, smaller_is_stronger=False)) #0.25
    

    s = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    print(robustness(s)) #inf (most likely never going to happen, just for completeness)
        
    s = np.array([[1, 2, 3], [5, 6, 7], [8, 9, 10]])
    print(robustness(s)) #2.8

    s *= 10
    print(robustness(s)) #2.8

    s[0, 2] = 90
    print(s)
    print(robustness(s)) #4.1
    s[0, 2] = 2000
    print(s)
    print(robustness(s)) #4.5


    
if __name__ == "__main__":
    tests()