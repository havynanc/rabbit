import hist
import numpy as np
import tensorflow as tf


class FitDebugData:
    def __init__(self, indata):

        if indata.sparse:
            raise NotImplementedError("sparse mode is not supported yet")

        self.indata = indata

        self.axis_procs = self.indata.axis_procs
        self.axis_systs = hist.axis.StrCategory(indata.systs, name="systs")
        self.axis_downup = hist.axis.StrCategory(["Down", "Up"], name="DownUp")

        self.data_obs_hists = {}
        self.nominal_hists = {}
        self.syst_hists = {}
        self.syst_active_hists = {}

        ibin = 0
        for channel, info in self.indata.channel_info.items():
            axes = info["axes"]
            shape = [len(a) for a in axes]
            stop = ibin + np.prod(shape)

            shape_norm = [*shape, self.indata.nproc]
            if self.indata.symmetric_tensor:
                shape_logk = [*shape, self.indata.nproc, self.indata.nsyst]
            else:
                shape_logk = [*shape, self.indata.nproc, 2, self.indata.nsyst]

            if not info.get("masked", False):
                data_obs_hist = hist.Hist(
                    *axes, name=f"{channel}_data_obs", storage=hist.storage.Weight()
                )
                data_obs_hist.values()[...] = memoryview(
                    tf.reshape(self.indata.data_obs[ibin:stop], shape)
                )
                if (
                    hasattr(self.indata, "data_var")
                    and self.indata.data_var is not None
                ):
                    data_obs_hist.variances()[...] = memoryview(
                        tf.reshape(self.indata.data_var[ibin:stop], shape)
                    )
                else:
                    data_obs_hist.variances()[...] = data_obs_hist.values()[...]

            nominal_hist = hist.Hist(
                *axes, self.axis_procs, name=f"{channel}_nominal", storage="Weight"
            )
            nominal_hist.values()[...] = memoryview(
                tf.reshape(self.indata.norm[ibin:stop, :], shape_norm)
            )
            nominal_hist.variances()[...] = memoryview(
                tf.reshape(self.indata.sumw2[ibin:stop, :], shape_norm)
            )

            # TODO do these operations on logk in tensorflow instead of numpy to use
            # multiple cores
            logk_array = np.asarray(
                memoryview(tf.reshape(self.indata.logk[ibin:stop, :], shape_logk))
            )

            if self.indata.systematic_type == "log_normal":

                def get_syst(logk, nominal):
                    return np.exp(logk) * nominal

            elif self.indata.systematic_type == "normal":

                def get_syst(logk, nominal):
                    return logk + nominal

            norm = nominal_hist.values()[..., None]
            if self.indata.symmetric_tensor:
                logkavg = logk_array[..., :]
                syst_down = get_syst(-logkavg, norm)
                syst_up = get_syst(logkavg, norm)

                nonzero = np.abs(logkavg) > 0.0
            else:
                logkavg = logk_array[..., 0, :]
                logkhalfdiff = logk_array[..., 1, :]

                syst_down = get_syst(-logkavg + logkhalfdiff, norm)
                syst_up = get_syst(logkavg + logkhalfdiff, norm)

                nonzero = np.logical_or(
                    np.abs(logkavg) > 0.0, np.abs(logkhalfdiff) > 0.0
                )

            syst_hist = hist.Hist(
                *axes,
                self.axis_procs,
                self.axis_systs,
                self.axis_downup,
                name=f"{channel}_syst",
            )
            syst_hist[{"DownUp": "Down"}] = syst_down
            syst_hist[{"DownUp": "Up"}] = syst_up

            syst_active_hist = hist.Hist(
                self.axis_procs,
                self.axis_systs,
                name=f"{channel}_syst_active",
                storage=hist.storage.Int64(),
            )
            syst_active_hist.values()[...] = np.sum(
                nonzero, axis=tuple(range(len(axes)))
            )

            if not info.get("masked", False):
                self.data_obs_hists[channel] = data_obs_hist
            self.nominal_hists[channel] = nominal_hist
            self.syst_hists[channel] = syst_hist
            self.syst_active_hists[channel] = syst_active_hist

            ibin = stop

    def nonzeroSysts(self, channels=None, procs=None):
        if channels is None:
            channels = self.indata.channel_info.keys()

        if procs is None:
            procs = list(self.axis_procs)

        nonzero_syst_idxs = set()

        for channel in channels:
            syst_active_hist = self.syst_active_hists[channel]
            syst_active_hist = syst_active_hist[{"processes": procs}]
            idxs = np.nonzero(syst_active_hist)
            syst_axis_idxs = syst_active_hist.axes.name.index("systs")
            syst_idxs = idxs[syst_axis_idxs]
            nonzero_syst_idxs.update(syst_idxs)

        nonzero_systs = []
        for isyst, syst in enumerate(self.indata.systs):
            if isyst in nonzero_syst_idxs:
                nonzero_systs.append(syst.decode("utf-8"))

        return nonzero_systs

    def channelsForNonzeroSysts(self, procs=None, systs=None):
        if procs is None:
            procs = list(self.axis_procs)

        if systs is None:
            systs = list(self.axis_systs)

        channels_out = []

        for channel in self.indata.channel_info:
            syst_active_hist = self.syst_active_hists[channel]
            syst_active_hist = syst_active_hist[{"processes": procs, "systs": systs}]
            if np.count_nonzero(syst_active_hist.values()) > 0:
                channels_out.append(channel)

        return channels_out

    def procsForNonzeroSysts(self, channels=None, systs=None):
        if channels is None:
            channels = self.indata.channel_info.keys()

        if systs is None:
            systs = list(self.axis_systs)

        proc_idxs_out = set()

        for channel in channels:
            syst_active_hist = self.syst_active_hists[channel]
            syst_active_hist = syst_active_hist[{"systs": systs}]
            idxs = np.nonzero(syst_active_hist)
            proc_axis_idxs = syst_active_hist.axes.name.index("processes")
            proc_idxs = idxs[proc_axis_idxs]
            proc_idxs_out.update(proc_idxs)

        nonzero_procs = []
        for iproc, proc in enumerate(self.indata.procs):
            if iproc in proc_idxs_out:
                nonzero_procs.append(proc)

        return nonzero_procs
