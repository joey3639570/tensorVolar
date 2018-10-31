from scipy.stats import wasserstein_distance
import numpy as np
np.set_printoptions(threshold=np.inf)

def min_max_distance(a,b):
    if np.mean(a) > np.mean(b):
        distance = np.min(a) - np.max(b)
    else:
        distance = np.min(b) - np.max(a)
    return distance

def mean_distance(a,b):
    return np.abs(np.mean(a) - np.mean(b))

def num_overlap(a,b):
    min_b = np.min(b)
    max_b = np.max(b)
    min_a = np.min(a)
    max_a = np.max(a)
    if min_b < max_a < max_b:
        if min_a < min_b: # partial overlap
            range_min = min_b
            range_max = max_a
        else:
            range_min = min_a
            range_max = max_a
    elif max_b < max_a:
        if min_b < min_a < max_b:
            range_min = min_a
            range_max = max_b
        elif min_a < min_b:
            range_min = min_b
            range_max = max_b
        else: # max_b < min_a
            return 0
    else: # max_a < min_b
        return 0

    count =  np.count_nonzero(np.logical_and(range_min<a, a<range_max))
    count += np.count_nonzero(np.logical_and(range_min<b, b<range_max))
    return count


fft = np.log(np.load("saved/fft_all.npy"))
labels = np.load("saved/labels.npy")
sort_idx = np.argsort(labels)
fft = fft[sort_idx]
labels = labels[sort_idx]
unique, counts = np.unique(labels, return_counts=True)
num_samples = dict(zip(unique, counts))
start_index = { type_key:np.nonzero(labels==type_key)[0][0] for type_key in num_samples}
print("number of sample in each type: ", num_samples)
print("start index for each type: ", start_index)

std_vals = np.std(fft, axis=0)
mean_vals = np.mean(fft, axis=0)
fft = (fft-mean_vals)/std_vals

distances = np.zeros([12000])
for i in range(0,12000):
    sum_types = []
    for j in range(3): 
        # select two of three types as t1, t2 
        # then calculate distance between these two types
        t1, t2 = np.mod([j, j+1], 3) + 1
        sum_types.append(wasserstein_distance(
                fft[start_index[t1]:start_index[t1]+num_samples[t1], i],
                fft[start_index[t2]:start_index[t2]+num_samples[t2], i]))
    distances[i] = np.sum(sum_types)

select_freq = []
select_dist = []
sample_len = 10
for i in range(0, len(distances), sample_len):
    select_freq.append(i+np.argmax(distances[i:i+sample_len]))
    select_dist.append(np.max(distances[i:i+sample_len]))
#print(len(select_freq))
#sort_idx = np.argsort(select_dist)[-1200:]
#select_freq = np.array(select_freq)[sort_idx]
#select_freq = np.array(select_freq)
#select_dist = np.array(select_dist)
#reselect = np.argsort(distances[select_freq])[-1100:]
#select_freq = select_freq[reselect]
#select_dist = select_dist[reselect]

#select_freq = np.array(select_freq)
#select_dist = np.array(select_dist)
#reselect = np.argsort(select_dist)[-1200:]
#select_freq = select_freq[reselect]
#select_dist = select_dist[reselect]

np.save("saved/select_freq.npy", select_freq)
#sorted_dict = dict(zip(select_freq, distances[select_freq]))
#print(sorted_dict)
print(len(select_freq))
print("min:", np.min(select_dist))
print("max:", np.max(select_dist))
print("mean:", np.mean(select_dist))

## test feature correlation
#select_freq = np.arange(0,12001)
#selected_feature = fft[:, select_freq]
#print(selected_feature.shape)
#corrs = np.abs(np.corrcoef(selected_feature, rowvar=False))
#print(corrs.shape)
#high_corrs = np.nonzero(corrs>0.7)
#no_diag = high_corrs[0] != high_corrs[1]
#print(high_corrs[0][no_diag], high_corrs[1][no_diag])
##print(len(high_corrs[0]))
