import matplotlib.pyplot as plt

# Assuming Dict Data
def time_distribution(path_times):
    compound_data = [times for times in path_times.values()] # unflattened data
    full_data = [] # flattened datea
    for times in path_times.values():
        full_data = full_data + times
    _,_,bars = plt.hist(full_data, bins="rice") #, 
    plt.set_title("Infection time distribution, all paths")
    plt.bar_label(bars)
    
def path_distribution(path_counts):
    path_names = [f"{path}" for path in path_counts.keys()]
    bar = plt.barh(path_names, list(path_counts.values()))
    plt.bar_label(bar)
    plt.set_title("Path distribution")

# Assuming Dict Data
def time_per_path(path_times):
    path_count = len(path_times.keys())
    fig, axs = plt.subplots(path_count, 1, figsize=(16, 4*path_count))
    for i, path in enumerate(path_times.keys()):
        _,_,bars = axs[i].hist(path_times[path], bins="rice")
        axs[i].set_title(f"Infection time distribution, condtioned on path {path}")
        axs[i].bar_label(bars)