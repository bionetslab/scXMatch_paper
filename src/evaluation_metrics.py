import numpy as np
from scipy.special import comb
from scipy.stats import iqr
from scipy.stats import median_abs_deviation

def round_sig(x, sig=2):
    if x == 0:
        return 0
    return round(x, -int(np.floor(np.log10(abs(x)))) + (sig - 1))


def monotonicity(s, smaller_is_stronger=False):
    """
    Expects a 1D Series or array of scores sorted by biological perturbation strength.
    Returns a monotonicity score between 0 and 1.
    """
    f = np.vectorize(round_sig)
    s = f(s.values)  # round to 2 significant decimal place

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
        - a scalar score s_0, retrieved for testing a group against reference
        - a 1D array of scores s, with dimensions 1 x |P|, which contains scores for comparison of disjunct
        subsets within the group l
    """
    f = np.vectorize(round_sig)
    s = f(s)
    s_0 = f(s_0)
    
    if smaller_is_stronger:
        result = np.sum(s >= s_0) 
    else:
        result = np.sum(s <= s_0)
            
    return result / (np.prod(s.shape)) 
        

def tests():
    print(monotonicity(np.array([1, 2, 3, 4, 5]), smaller_is_stronger=False)) #1.0
    print(monotonicity(np.array([1, 2, 5, 4, 5]), smaller_is_stronger=False)) #0.9
    print(monotonicity(np.array([1, 2, 3, 4, 5]), smaller_is_stronger=True)) #0.0
    
    s_0 = np.array([0.01, 0.02, 0.03])
    s = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    print(SSNR(s_0, s, smaller_is_stronger=True)) #1.0
    s = np.array([[0.4, 0.2, 1, 0.3], [0.05, 0.2, 1, 0.3], [0.01, 0.01, 0.01, 0.3]])
    print(SSNR(s_0, s, smaller_is_stronger=False)) #0.25
    
    
if __name__ == "__main__":
    tests()