import numpy as np
from scipy.special import comb

def monotonicity(s, smaller_is_stronger=False):
    """
    expects a 1D array of scores sorted by biological perturbation strength! 
    """
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
        return 1 / (np.std((s - np.min(s))/(np.max(s) - np.min(s))) ** 2)
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
        
    s = np.array([[1, 1, 1], [1, 1, 4], [1, 1, 1]])
    print(robustness(s)) #10.12
    # multiply by 10
    s = np.array([[10, 10, 10], [10, 10, 14], [10, 10, 10]])
    print(robustness(s)) #10.12
    # replace 14 by 40
    s = np.array([[10, 10, 10], [10, 10, 40], [10, 10, 10]])
    print(robustness(s)) #10.12
    # change 1 other value
    s = np.array([[10, 10, 10], [10, 10, 40], [20, 10, 10]])
    print(robustness(s)) #9.85
    # set 1 value to a really high-fold number compared to the others
    s = np.array([[10, 10, 98], [10, 10, 40], [20, 10, 10]])
    print(robustness(s)) #10.06

    
if __name__ == "__main__":
    tests()