from scipy.stats import wasserstein_distance
import numpy as np
np.set_printoptions(threshold=np.inf)

fft = np.log(np.load("fft_all.npy"))
std_vals = np.std(fft, axis=0)
mean_vals = np.mean(fft, axis=0)
fft = (fft-mean_vals)/std_vals

distances = np.zeros([12000])
for i in range(1,12000):
    sum_types = 0
    for j in range(3):
        types = np.mod([j, j+1], 3)
        sum_types += wasserstein_distance(fft[types[0]*100:(types[0]+1)*100, i], fft[types[1]*100:(types[1]+1)*100, i])
    distances[i] = sum_types

sort_idx = np.argsort(distances)[-4000:]
np.save("select_freq.npy", sort_idx)
sorted_dict = dict(zip(sort_idx, distances[sort_idx]))
print(sort_idx)
#print(repr(sorted_dict).replace(",", "\n"))
