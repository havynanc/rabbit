#!/usr/bin/env python

import argparse
import importlib

import numpy as np
from wums import logging

from rabbit import debugdata, inputdata, tensorwriter


# for this script, modifiying the input object should be okay and will save memory but keep this in mind for other applications
def clip_hist(hist_in, thresh):
    values = np.copy(hist_in.values())
    variances = np.copy(hist_in.variances())
    hist_in.values()[...] = np.clip(values, thresh, np.inf)
    hist_in.variances()[...] = np.clip(variances, thresh, np.inf)
    return hist_in


# returns an array such that indexing it by the syst index returns a list of all the (renamed) groups that syst is in
def get_groups_per_syst(indata, rename_groups, logger):
    groups_per_syst = [[]] * indata.nsyst
    for g, idxs in zip(indata.systgroups, indata.systgroupidxs):
        g = g.decode("UTF-8")
        if g not in rename_groups.keys():
            new_g = g
        else:
            new_g = rename_groups[g]
            logger.debug(f"Renaming group {g} to {new_g}")
        for idx in idxs:
            groups_per_syst[idx] = groups_per_syst[idx] + [new_g]
    return groups_per_syst


# used to rename systs
def load_module(path, name="module"):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="./", help="output directory")
    parser.add_argument("--outname", default="merged_tensor", help="output file name")
    parser.add_argument(
        "--systematicType",
        choices=["log_normal", "normal"],
        default="log_normal",
        help="probability density for systematic variations",
    )
    parser.add_argument(
        "-t",
        "--tensors",
        nargs="+",
        help="multiple rabbit input tensors to merge into a single rabbit input tensor",
    )
    parser.add_argument(
        "-c",
        "--clipTensors",
        nargs="+",
        default=[],
        help="clip the specified tensors at the specified thresholds: -c tensor1 thresh1 tensor2 thresh2...",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=3,
        choices=[0, 1, 2, 3, 4],
        help="Set verbosity level with logging, the larger the more verbose",
    )
    parser.add_argument(
        "--noColorLogger",
        action="store_true",
        help="Do not use logging with colors",
    )
    parser.add_argument(
        "--rename",
        default=None,
        help="file path of a python module containing dictionaries to rename systs, groups, processes, and channels (named rename_systs, rename_groups, etc). Old names are keys and new names are values",
    )
    parser.add_argument(
        "--sparse", action="store_true", help="make the merged tensor sparse"
    )
    args = parser.parse_args()
    return args


def main():

    args = parseArgs()
    global logger
    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)
    rename_systs = {}
    rename_groups = {}
    rename_processes = {}
    rename_channels = {}
    if args.rename is not None:
        module = load_module(args.rename)
        try:
            rename_systs = module.rename_systs
        except:
            logger.debug("Renaming file did not have a systs dictionary")
        try:
            rename_groups = module.rename_groups
        except:
            logger.debug("Renaming file did not have a groups dictionary")
        try:
            rename_processes = module.rename_processes
        except:
            logger.debug("Renaming file did not have a processes dictionary")
        try:
            rename_channels = module.rename_channels
        except:
            logger.debug("Renaming file did not have a channels dictionary")

    clip_dict = {}
    if len(args.clipTensors) % 2 != 0:
        raise ValueError(
            "--clipTensors must take an even number of arguments (tensor1, thresh1, tensor2, thresh2, ...)"
        )
    for i in range(len(args.clipTensors) // 2):
        clip_dict[args.clipTensors[2 * i]] = float(args.clipTensors[2 * i + 1])

    writer = tensorwriter.TensorWriter(
        systematic_type=args.systematicType, sparse=args.sparse
    )

    for tensor in args.tensors:

        logger.info(f"Now at input tensor {tensor}")
        if tensor in clip_dict.keys():
            logger.info(f"(clipping at thresh of {clip_dict[tensor]})")
        logger.info("Loading inputdata")
        indata = inputdata.FitInputData(tensor)
        signals = np.array([s.decode("UTF-8") for s in indata.signals])
        logger.info("Loading debugdata")
        debug_data = debugdata.FitDebugData(indata)
        nominal_all_channels = debug_data.nominal_hists

        for ch in nominal_all_channels.keys():

            if ch not in rename_channels.keys():
                new_ch = ch
                logger.info(f"Setting up channel {new_ch}")
            else:
                new_ch = rename_channels[ch]
                logger.info(
                    f"Setting up channel {new_ch} (renamed from input channel {ch})"
                )

            if tensor not in clip_dict.keys():
                nominal = nominal_all_channels[ch]
                syst = debug_data.syst_hists[ch]
            else:
                nominal = clip_hist(nominal_all_channels[ch], clip_dict[tensor])
                syst = clip_hist(debug_data.syst_hists[ch], clip_dict[tensor])

            procs = np.array(nominal.axes["processes"])
            new_procs = np.copy(procs)
            for i, proc in enumerate(new_procs):
                if proc in rename_processes.keys():
                    new_procs[i] = rename_processes[proc]
                    logger.debug(f"Renaming process {proc} to {new_procs[i]}")

            writer.add_channel(nominal[{"processes": procs[0]}].axes, new_ch)
            for proc, new_proc in zip(procs, new_procs):
                writer.add_process(
                    nominal[{"processes": proc}],
                    new_proc,
                    new_ch,
                    signal=(proc in signals),
                )
            data = debug_data.data_obs_hists[ch]
            writer.add_data(data, new_ch)

            nois = np.array(
                [noi.decode("UTF-8") for noi in indata.systs[indata.noiidxs]]
            )
            systsnoconstraint = np.array(
                [snc.decode("UTF-8") for snc in indata.systsnoconstraint]
            )
            groups_per_syst = get_groups_per_syst(
                indata, rename_groups, logger
            )  # handles renaming of groups too since we don't directly read groups again

            logger.info("Adding systs")
            for idx in range(indata.nsyst):

                syst_name = indata.systs[idx].decode("UTF-8")
                groups = groups_per_syst[idx]
                procs = debug_data.procsForNonzeroSysts(
                    channels=[ch], systs=[syst_name]
                )

                is_noi = syst_name in nois
                is_constrained = syst_name not in systsnoconstraint
                if is_noi:
                    logger.debug(f"Treating {syst_name} as noi")
                if not is_constrained:
                    logger.debug(f"Treating {syst_name} as not constrained")

                if syst_name not in rename_systs.keys():
                    new_syst_name = syst_name
                else:
                    new_syst_name = rename_systs[syst_name]
                    logger.debug(f"Renaming syst {syst_name} to {new_syst_name}")

                for proc in procs:
                    proc = proc.decode("UTF-8")
                    new_proc = (
                        proc
                        if proc not in rename_processes.keys()
                        else rename_processes[proc]
                    )
                    up = syst[{"processes": proc, "systs": syst_name, "DownUp": "Up"}]
                    down = syst[
                        {"processes": proc, "systs": syst_name, "DownUp": "Down"}
                    ]
                    writer.add_systematic(
                        [up, down],
                        new_syst_name,
                        new_proc,
                        new_ch,
                        groups=groups,
                        symmetrize="none",  # inherit symmetrization of input tensors
                        noi=is_noi,
                        constrained=is_constrained,
                    )

    logger.info("Writing output tensor")
    directory = args.output
    if directory == "":
        directory = "./"
    filename = args.outname
    writer.write(outfolder=directory, outfilename=filename)


if __name__ == "__main__":
    main()
