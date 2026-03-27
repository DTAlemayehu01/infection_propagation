import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from scipy import stats
from collections import defaultdict

# Assuming Dict Data
def time_distribution(path_times):
    compound_data = [times for times in path_times.values()] # unflattened data
    full_data = [] # flattened datea
    for times in path_times.values():
        full_data = full_data + times
    _,_,bars = plt.hist(full_data, bins="rice") #, 
    plt.title("Infection time distribution, all paths")
    plt.bar_label(bars)
    
def path_distribution(path_counts, threshold=1, h=0.5):
    path_names = [f"{path}" for path in path_counts.keys()]
    # Threshold
    path_names_reduced = []
    path_counts_reduced = []
    if threshold != 1:
        for name, count in path_counts.items():
            if count >= threshold:
                path_names_reduced = path_names_reduced + [f"{name}"]
                path_counts_reduced = path_counts_reduced + [count]
    else:
        path_names_reduced = path_names
        path_counts_reduced = list(path_counts.values())
    # Plotting
    plt.figure(figsize=(8, len(path_names_reduced)*h))
    plt.margins(y=0)
    bar = plt.barh(path_names_reduced, path_counts_reduced, height=h) # Bar Height?
    plt.bar_label(bar)
    plt.title("Path distribution")

# Assuming Dict Data
def time_per_path(path_times, threshold=1):
    path_count = 0
    if threshold != 1:
        for trial in path_times.values():
            if len(trial) >= threshold:
                path_count = path_count + 1
    else:
        path_count = len(path_times.keys())
    print(path_count)
    fig, axs = plt.subplots(path_count, 1, figsize=(16, 4*path_count))
    i = 0
    for path in path_times.keys():
        if threshold != 1:
            if len(path_times[path]) >= threshold:
                _,_,bars = axs[i].hist(path_times[path], bins="rice")
                axs[i].set_title(f"Infection time distribution, condtioned on path {path}")
                axs[i].bar_label(bars)
                i = i+1
        else:
            _,_,bars = axs[i].hist(path_times[path], bins="rice")
            axs[i].set_title(f"Infection time distribution, condtioned on path {path}")
            axs[i].bar_label(bars)
            i = i+1
        
# Only really relevant on iid edges?
def nodes_vs_time(path_times):
    path_lens = []
    path_len_times = []
    for key in path_times.keys():
        pts = list(product([len(key)-1], path_times[key]))
        path_lens = path_lens + [pt[0] for pt in pts]
        path_len_times = path_len_times + [pt[1] for pt in pts]
        
    path_lens = np.array(path_lens)
    path_len_times = np.array(path_len_times)

    slope, intercept, r, p, err = stats.linregress(path_lens, path_len_times)    
    lin_fit = slope*path_lens + intercept
    
    time_len_dict = defaultdict(list)
    for pt in zip(path_lens, path_len_times):
        time_len_dict[pt[0]] = time_len_dict[pt[0]] + [pt[1]]
    
    ordered_keys = sorted(time_len_dict.keys())
    for x in ordered_keys:
        times = time_len_dict[x]
        times = np.array(times)
        print(f"Path Length = {x} Units: Average={np.mean(times)}, Variance={np.var(times)}, Min={np.min(times)}, Max={np.max(times)}")
    print(f"Slope={slope}")
    print(f"Y-Intercept={intercept}")

    plt.scatter(path_lens, path_len_times)
    plt.plot(path_lens, lin_fit, label=f"LSRL, r={r: .2f}, p={p: .2f}, std-err={err: .2f}", color='orange', linestyle='--')
    plt.title("Infection Time vs Path Length")
    plt.xlabel("Path Length")
    plt.ylabel("Infection Time")
    plt.legend()
    return r, p, err