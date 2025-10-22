import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

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
        return input[24 + 1 : 2 * num_eta_bins + 1]
    elif group == "M":
        return input[2 * 24 + 1 : 3 * num_eta_bins + 1]
    elif group == "zmass":
        return input[[0]]


args = parseArgs()
if args.outpath[-1] != "/":
    args.outpath = args.outpath + "/"
if not os.path.isdir(args.outpath):
    os.mkdir(args.outpath)

pulls_info = []
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
    _, pulls_gen, constraints_gen = io_tools.get_pulls_and_constraints(
        fitresult, prefit=False, gen=True, asym=False
    )
    pulls_info.append((pulls - pulls_gen) / constraints)
    # constraints_info.append(constraints)
    if i == 1:
        labels_info = labels
pulls_info = np.array(pulls_info)

# constraints_info = np.array(constraints_info)
# adjust scale for zmass to MeV
# pulls_info[:,0] = 100 * pulls_info[:,0]
# constraints_info[:,0] = 100 * constraints_info[:,0]

mean_pulls = np.mean(pulls_info, axis=0)
std_pulls = np.std(pulls_info, axis=0)
# print(mean_pulls[[0, 1+17, 1+17+num_eta_bins, 1+17+2*num_eta_bins]])
# print(std_pulls[[0, 1+17, 1+17+num_eta_bins, 1+17+2*num_eta_bins]])
# exit()

hist_range = (-20, 20)
bins = 50
for i, label in enumerate(labels_info):
    # hist_range = (-2, 2) if label != "zmass" else (-200, 200)
    plt.hist(pulls_info[:, i], range=hist_range, bins=bins, color=get_color(label))
    plt.xlabel("pull ((fit param - gen param) / constraint)")
    plt.ylabel("number")
    plt.title(
        f"{label} {args.numToys} bayesian toys (mean {mean_pulls[i]:.2f}, std {std_pulls[i]:.2f})"
    )
    plt.savefig(f"{args.outpath}pullhist_{label}.png")
    plt.clf()

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
