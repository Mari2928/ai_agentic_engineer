import json
import os
import os.path as osp

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# LOAD FINAL RESULTS:
datasets = ["ml_regression"]
folders = os.listdir("./")
final_results = {}
results_info = {}


for folder in folders:
    if folder.startswith("run") and osp.isdir(folder):
        with open(osp.join(folder, "final_info.json"), "r") as f:
            final_results[folder] = json.load(f)
        results_dict = np.load(osp.join(folder, "all_results.npy"), allow_pickle=True).item()
        run_info = {}
        for dataset in datasets:
            run_info[dataset] = {}
            val_losses = []
            train_losses = []
            for k in results_dict.keys():
                if dataset in k and "train_info" in k:
                    run_info[dataset]["iters"] = [info["iter"] for info in results_dict[k]]
                    train_losses.append([info["mse"] for info in results_dict[k]])

                mean_train_losses = np.mean(train_losses, axis=0) if len(train_losses) > 0 else 0.0

                if len(train_losses) > 0:
                    stderr_train_losses = np.std(train_losses, axis=0) / np.sqrt(len(train_losses))
                else:
                    stderr_train_losses = np.zeros_like(mean_train_losses)
                run_info[dataset]["mse"] = mean_train_losses
                run_info[dataset]["mse_sterr"] = stderr_train_losses
        results_info[folder] = run_info

# CREATE LEGEND -- ADD RUNS HERE THAT WILL BE PLOTTED
labels = {
    "run_0": "Baselines",
}


# Create a programmatic color palette
def generate_color_palette(n):
    cmap = plt.get_cmap('tab20')
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, n)]


# Get the list of runs and generate the color palette
runs = list(labels.keys())
colors = generate_color_palette(len(runs))

# Plot 1: Line plot of training loss for each dataset across the runs with labels
for dataset in datasets:
    plt.figure(figsize=(10, 6))
    for i, run in enumerate(runs):
        iters = results_info[run][dataset]["iters"]
        mean = results_info[run][dataset]["mse"]
        sterr = results_info[run][dataset]["mse_sterr"]
        plt.plot(iters, mean, label=labels[run], color=colors[i])
        plt.fill_between(iters, mean - sterr, mean + sterr, color=colors[i], alpha=0.2)

    plt.title(f"MSE Across Runs for {dataset} Pipeline")
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"mse_{dataset}.png")
    plt.close()

# Plot 2: Line plot of validation loss for each dataset across the runs with labels
# for dataset in datasets:
#     plt.figure(figsize=(10, 6))
#     for i, run in enumerate(runs):
#         iters = results_info[run][dataset]["iters"]
#         mean = results_info[run][dataset]["val_loss"]
#         sterr = results_info[run][dataset]["val_loss_sterr"]
#         plt.plot(iters, mean, label=labels[run], color=colors[i])
#         plt.fill_between(iters, mean - sterr, mean + sterr, color=colors[i], alpha=0.2)

#     plt.title(f"Validation Loss Across Runs for {dataset} Dataset")
#     plt.xlabel("Iteration")
#     plt.ylabel("Validation Loss")
#     plt.legend()
#     plt.grid(True, which="both", ls="-", alpha=0.2)
#     plt.tight_layout()
#     plt.savefig(f"val_loss_{dataset}.png")
#     plt.close()
