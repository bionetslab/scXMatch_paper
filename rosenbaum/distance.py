
def calculate_distances(samples, metric):
    try:
        if GPU:
            print("trying to use GPU to calculate distance matrix.")
        distances = cp.asnumpy(gpu_cdist(cp.array(samples), cp.array(samples), metric=metric)) 

    except:
        if GPU:
            print("using CPU to calculate distance matrix due to chosen metric.")
        else:
            print("using CPU to calculate distance matrix.")
        distances = cpu_cdist(samples, samples, metric=metric)
    #
    num_samples = len(samples)

    if num_samples % 2 != 0: # with an uneven number of samples, a minimal-distance column is added 
        distances = np.pad(distances, [(0, 1), (0, 1)], mode='constant', constant_values=0)
        num_samples += 1

    max_distance = np.max(distances)
    distances = max_distance + 1 - distances
    return distances
