import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from rabbit import io_tools


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputFile",
        type=str,
        help="fitresults output hdf5 file from fit",
    )
    parser.add_argument(
        "--numToys",
        default=1,
        type=int,
        help="number of toys to consider",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["bayes", "freq"],
        help="which type of toys",
    )
    parser.add_argument(
        "-o",
        "--outpath",
        type=str,
        default="./pullhists",
        help="Folder path for output",
    )
    return parser.parse_args()


def get_color(label):
    if label[0] == "A":
        return "red"
    elif label[0] == "e":
        return "brown"
    elif label[0] == "M":
        return "green"
    elif label[0:5] == "zmass":
        return "purple"


num_eta_bins = 24


def get_group_subset(input, group):
    if group == "A":
        return input[1 : 1 * num_eta_bins + 1]
    elif group == "e":
        return input[num_eta_bins + 1 : 2 * num_eta_bins + 1]
    elif group == "M":
        return input[2 * num_eta_bins + 1 : 3 * num_eta_bins + 1]
    elif group == "zmass":
        return input[[0]]


args = parseArgs()
if args.outpath[-1] != "/":
    args.outpath = args.outpath + "/"
if not os.path.isdir(args.outpath):
    os.mkdir(args.outpath)
mode = args.mode

pulls_info = []
params_info = []
# constraints_info = []
labels_info = None

poi = "zmass"
group = False
diff_pulls = "gen"
asym = False
impact_type = "traditional"

for i in range(1, args.numToys + 1):
    result = f"toy{i}"
    fitresult, meta = io_tools.get_fitresult(args.inputFile, result, meta=True)
    labels, pulls, constraints = io_tools.get_pulls_and_constraints(
        fitresult, prefit=False, gen=False, asym=False
    )
    params_info.append(pulls)

    if mode == "bayes":
        _, pulls_gen, constraints_gen = io_tools.get_pulls_and_constraints(
            fitresult, prefit=False, gen=True, asym=False
        )
        pulls_info.append((pulls - pulls_gen) / constraints)
    else:
        pulls_info.append(pulls / constraints)

    if i == 1:
        labels_info = labels
print(labels_info)
pulls_info = np.array(pulls_info)
params_info = np.array(params_info)
print(pulls_info.shape)

# constraints_info = np.array(constraints_info)
# adjust scale for zmass to MeV
# pulls_info[:,0] = 100 * pulls_info[:,0]
# constraints_info[:,0] = 100 * constraints_info[:,0]

mean_pulls = np.mean(pulls_info, axis=0)
std_pulls = np.std(pulls_info, axis=0)
mean_params = np.mean(params_info, axis=0)
std_params = np.std(params_info, axis=0)
# print(mean_pulls[[0, 1+17, 1+17+num_eta_bins, 1+17+2*num_eta_bins]])
# print(std_pulls[[0, 1+17, 1+17+num_eta_bins, 1+17+2*num_eta_bins]])
# exit()

corr = np.corrcoef(params_info.T)

hist_range_pulls = np.array([-6, 6])
hist_range_params = np.array([-3, 3])
bins = 50

# scale factors and units for params hists
scale_factors = {"zmass": 91.1876, "A": 1e-3, "e": 1e-2, "M": 1e-4}
# scale_factors["zmass"] = 100
units = {"zmass": "(MeV)", "A": "", "e": "(GeV)", "M": "(GeV^-1)"}
params_ranges = {
    "zmass": (-6, 6),
    "A": (-6e-5, 6e-5),
    "e": (-36e-5, 36e-5),
    "M": (-36e-7, 36e-7),
}

for i, label in enumerate(labels_info):
    # hist_range = (-2, 2) if label != "zmass" else (-200, 200)
    plt.hist(
        pulls_info[:, i], range=hist_range_pulls, bins=bins, color=get_color(label)
    )
    plt.xlabel("pull ((fit param - 0) / constraint)")
    plt.ylabel("number")
    plt.title(
        f"{label} {args.numToys} {mode} toys (mean {mean_pulls[i]:.2e}, std {std_pulls[i]:.2e})"
    )
    plt.savefig(f"{args.outpath}pullhist_{label}.png")
    plt.clf()

    group = label if label == "zmass" else label[0]
    scale_factor = scale_factors[group]
    unit = units[group]

    # current_hist_range_params = hist_range_params*scale_factor if group != "zmass" else np.array([-6, 6])
    current_hist_range_params = params_ranges[group]
    current_hist_range_params = np.array(
        current_hist_range_params
    )  # *10 to zoom out for bad z only fits
    plt.hist(
        params_info[:, i] * scale_factor,
        range=current_hist_range_params,
        bins=bins,
        color=get_color(label),
    )
    plt.xlabel(f"difference (fit param - 0) {unit}")
    plt.ylabel("number")
    plt.title(
        f"{label} {args.numToys} {mode} toys (mean {mean_params[i]*scale_factor:.2e}, std {std_params[i]*scale_factor:.2e})"
    )
    plt.gca().ticklabel_format(
        style="sci", axis="x", scilimits=(0, 0)
    )  # use sci notation on all plot axes
    plt.savefig(f"{args.outpath}paramshist_{label}.png")
    plt.clf()

    # if i != 0:
    #     plt.hist(params_info[:, i], range=hist_range_params, bins=bins, color=get_color(label))
    #     plt.xlabel("difference (fit param - 0)")
    #     plt.ylabel("number")
    #     plt.title(
    #         f"{label} {args.numToys} {mode} toys (mean {mean_params[i]:.2f}, std {std_params[i]:.2f})"
    #     )
    #     plt.savefig(f"{args.outpath}paramshist_{label}.png")
    #     plt.clf()
    # else:
    #     plt.hist(params_info[:, i]*91.1876, range=(-6, 6), bins=bins, color=get_color(label))
    #     plt.xlabel("difference (fit param - 0) (MeV)")
    #     plt.ylabel("number")
    #     plt.title(
    #         f"{label} {args.numToys} {mode} toys (mean {mean_params[i]*91.1876:.2f}, std {std_params[i]*91.1876:.2f})"
    #     )
    #     plt.savefig(f"{args.outpath}paramshist_{label}.png")
    #     plt.clf()

    # for group in ["A", "e", "M", "zmass"]:
#     group_mean_pulls = get_group_subset(mean_pulls, group)
#     group_std_pulls = get_group_subset(std_pulls, group)
#     group_mean_constraints = get_group_subset(mean_constraints, group)
#     col = get_color(group)
#     plt.errorbar(np.arange(len(group_mean_pulls)), group_mean_pulls, yerr=group_std_pulls, label="mean of toys ± std of toys", fmt="o", color=col, capsize=5)
#     plt.errorbar(np.arange(len(group_mean_pulls)), np.zeros(len(group_mean_pulls)), yerr=group_mean_constraints, label="0 ± mean constraint of toys", fmt="o", color="black", capsize=5)
#     plt.title(f"Observed spreads of (fit {group} - gen {group}) \ncompared with fit constraints ({group} {args.numToys} toys)")
#     if group == "zmass":
#         plt.ylabel("fit - gen (MeV)")
#         plt.xticks([])
#     else:
#         plt.ylabel("fit - gen")
#         plt.xlabel("index")
#     plt.tight_layout()
#     plt.legend()
#     plt.savefig(f"{args.outpath}pullhist_comparison_{group}.png")
#     plt.clf()

labels[0] = "mz"  # looks nicer on plot


# sns.set(font_scale=0.2)
def plot(matrix, labels, title, name):
    vmax = np.max(np.abs(matrix))
    # plt.figure(figsize=(20,20), facecolor=(1,1,1))
    plt.figure(facecolor=(1, 1, 1))
    ax = sns.heatmap(
        matrix,
        cmap="coolwarm",  # Choose a divergent colormap
        annot=True,  # Keep False for large matrices
        square=True,
        linewidths=0.5,
        linecolor="lightgray",
        # cbar_kws={'label': 'Covariance'},
        vmax=vmax,
        vmin=-vmax,
        xticklabels=labels,
        yticklabels=labels,
    )
    # cbar_kws = {'shrink':0.805})
    plt.title(title, fontsize=16)
    # plt.xlabel('Random Variable', fontsize=12)
    # plt.ylabel('Random Variable', fontsize=12)
    ##ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=0)
    ##ax.set_yticks(np.arange(len(labels)), labels=labels, rotation=0)
    # ax.set_xticklabels(labels, rotation=30)
    # ax.set_yticklabels(labels, rotation=0)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    # plt.xticks(rotation=0, fontsize=5)
    # plt.yticks(rotation=0, fontsize=5)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    # grid_interval = 3
    # grid_line_color = 'white'
    # grid_line_width = 2
    # line_positions = np.arange(grid_interval, 3*num_eta_bins, grid_interval)
    # ax.hlines(line_positions, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1],
    #          color=grid_line_color, linewidth=grid_line_width)
    # ax.vlines(line_positions, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1],
    #          color=grid_line_color, linewidth=grid_line_width)

    plt.savefig(name, dpi=400)


plot(
    corr,
    labels,
    f"Empirical Correlation ({args.numToys} {mode} toys)",
    f"{args.outpath}corr.png",
)
# native order is zmass, A0, A1, ..., e0, e1,... - convert to zmass, A0, e0, M0, A1, e1, M1, ...
num_vars = 3  # 3 for AeM
# num_eta_bins = int((len(labels)-1)/num_vars)
# print(num_eta_bins)
S = np.zeros_like(corr)
reordered_labels = np.copy(labels)
S[0, 0] = 1
for param_index in range(num_vars):
    for eta_bin in range(num_eta_bins):
        current = 1 + param_index * num_eta_bins + eta_bin
        final = 1 + num_vars * eta_bin + param_index
        S[final, current] = 1
        reordered_labels[final] = labels[current]
reordered_corr = S @ corr @ S.T

plot(
    reordered_corr,
    reordered_labels,
    f"Empirical Correlation ({args.numToys} {mode} toys)",
    f"{args.outpath}corr_reordered1.png",
)
print(S)
print(labels)
print(reordered_labels)
