import functools
import itertools

import numpy as np
import tensorflow as tf


def _active_axis_cells(indata, channel, proc_idx, selected_axes):
    """Return occupied selected-axis coordinates for one channel process."""
    info = indata.channel_info[channel]
    channel_axes = info["axes"]
    channel_shape = tuple(a.size for a in channel_axes)
    start = info["start"]
    stop = info["stop"]

    if indata.sparse:
        indices = indata.norm.indices.numpy()
        values = indata.norm.values.numpy()
        mask = (
            (indices[:, 0] >= start)
            & (indices[:, 0] < stop)
            & (indices[:, 1] == proc_idx)
            & (values != 0.0)
        )
        local_rows = indices[mask, 0] - start
    else:
        norm = indata.norm.numpy()[start:stop, proc_idx]
        local_rows = np.flatnonzero(norm)

    if len(local_rows) == 0:
        return np.empty((0, len(selected_axes)), dtype=np.int32)

    channel_coords = np.stack(np.unravel_index(local_rows, channel_shape), axis=-1)
    channel_axis_names = [a.name for a in channel_axes]
    axis_positions = [channel_axis_names.index(a.name) for a in selected_axes]
    return np.unique(channel_coords[:, axis_positions], axis=0).astype(np.int32)


def _sparse_channel_entries(indata, channel, proc_idx):
    """Return sparse norm positions and channel-local coordinates for a process."""
    info = indata.channel_info[channel]
    channel_shape = tuple(a.size for a in info["axes"])
    indices = indata.norm.indices.numpy()
    values = indata.norm.values.numpy()
    mask = (
        (indices[:, 0] >= info["start"])
        & (indices[:, 0] < info["stop"])
        & (indices[:, 1] == proc_idx)
        & (values != 0.0)
    )
    positions = np.flatnonzero(mask).astype(np.int32)
    local_rows = indices[mask, 0] - info["start"]
    if len(local_rows) == 0:
        return positions, np.empty((0, len(channel_shape)), dtype=np.int32)
    coords = np.stack(np.unravel_index(local_rows, channel_shape), axis=-1)
    return positions, coords.astype(np.int32)


class ParamModel:

    def __init__(self, indata, *args, **kwargs):
        self.indata = indata

        # # a param model must set these attributes
        # self.npoi = # number of true parameters of interest (POIs), reported as POIs in outputs
        # self.npou = # number of model nuisance parameters (parameters of uninterest)
        # self.params = # list of names for all parameters (POIs first, then model nuisances)
        # self.xparamdefault = # default values for all parameters (length nparams)
        # self.is_linear = # define if the model is linear in the parameters
        # self.allowNegativeParam = # define if the POI parameters can be negative or not

    @property
    def nparams(self):
        """Total number of parameters: npoi + npou."""
        return self.npoi + self.npou

    @property
    def param_constraint_means(self):
        return getattr(
            self,
            "_param_constraint_means",
            tf.zeros([self.nparams], dtype=self.indata.dtype),
        )

    @property
    def param_constraint_weights(self):
        return getattr(
            self,
            "_param_constraint_weights",
            tf.zeros([self.nparams], dtype=self.indata.dtype),
        )

    # class function to parse strings as given by the argparse input e.g. --paramModel <Model> <arg[0]> <args[1]> ...
    @classmethod
    def parse_args(cls, indata, *args, **kwargs):
        return cls(indata, *args, **kwargs)

    def compute(self, param, full=False):
        """
        Compute an array for the rate per process
        :param param: 1D tensor of explicit parameters in the fit (length nparams)
        :return 2D tensor to be multiplied with [proc,bin] tensor
        """

    def compute_sparse(self, param):
        """Compute scale factors aligned with the stored sparse norm entries."""
        rnorm = self.compute(param, full=True)
        rnorm = tf.broadcast_to(rnorm, [self.indata.nbinsfull, self.indata.nproc])
        return tf.gather_nd(rnorm, self.indata.norm.indices)

    def set_param_default(self, expectSignal, allowNegativeParam=False):
        """
        Set default parameter values, used by different param models.
        Only the first npoi entries (true POIs) support the squaring transform;
        model nuisance parameters (npou entries) are always stored directly.
        """
        paramdefault = tf.ones([self.nparams], dtype=self.indata.dtype)
        if expectSignal is not None:
            indices = []
            updates = []
            for signal, value in expectSignal:
                if signal.encode() not in self.params:
                    raise ValueError(
                        f"{signal.encode()} not in list of params: {self.params}"
                    )
                idx = np.where(np.isin(self.params, signal.encode()))[0][0]

                indices.append([idx])
                updates.append(float(value))

            paramdefault = tf.tensor_scatter_nd_update(paramdefault, indices, updates)

        # squaring transform applies only to the npoi true POI entries
        poi_part = paramdefault[: self.npoi]
        nui_part = paramdefault[self.npoi :]

        if allowNegativeParam:
            xpoi_part = poi_part
        else:
            xpoi_part = tf.sqrt(poi_part)

        self.xparamdefault = tf.concat([xpoi_part, nui_part], axis=0)


class CompositeParamModel(ParamModel):
    """
    multiply different param models together
    """

    def __init__(
        self,
        param_models,
        allowNegativeParam=False,
    ):

        self.param_models = param_models

        self.npoi = sum([m.npoi for m in param_models])
        self.npou = sum([m.npou for m in param_models])

        self.params = np.concatenate([m.params for m in param_models])

        self.allowNegativeParam = allowNegativeParam

        self.is_linear = self.nparams == 0 or self.allowNegativeParam

        self.xparamdefault = tf.concat([m.xparamdefault for m in param_models], axis=0)
        self._param_constraint_means = tf.concat(
            [m.param_constraint_means for m in param_models], axis=0
        )
        self._param_constraint_weights = tf.concat(
            [m.param_constraint_weights for m in param_models], axis=0
        )

    @property
    def param_constraint_means(self):
        return self._param_constraint_means

    @property
    def param_constraint_weights(self):
        return self._param_constraint_weights

    def compute(self, param, full=False):
        start = 0
        results = []
        for m in self.param_models:
            results.append(m.compute(param[start : start + m.nparams], full))
            start += m.nparams

        rnorm = functools.reduce(lambda a, b: a * b, results)
        return rnorm

    def compute_sparse(self, param):
        start = 0
        results = []
        for m in self.param_models:
            results.append(m.compute_sparse(param[start : start + m.nparams]))
            start += m.nparams

        return functools.reduce(lambda a, b: a * b, results)


class Ones(ParamModel):
    """
    multiply all processes with ones
    """

    def __init__(self, indata, **kwargs):
        self.indata = indata
        self.npoi = 0
        self.npou = 0
        self.params = np.array([])
        self.xparamdefault = tf.zeros([0], dtype=self.indata.dtype)
        self._param_constraint_means = tf.zeros([0], dtype=self.indata.dtype)
        self._param_constraint_weights = tf.zeros([0], dtype=self.indata.dtype)

        self.allowNegativeParam = False
        self.is_linear = True

    def compute(self, param, full=False):
        rnorm = tf.ones(self.indata.nproc, dtype=self.indata.dtype)
        rnorm = tf.reshape(rnorm, [1, -1])
        return rnorm


class Mu(ParamModel):
    """
    multiply unconstrained parameter to signal processes, and ones otherwise
    """

    def __init__(self, indata, expectSignal=None, allowNegativeParam=False, **kwargs):
        self.indata = indata

        self.npoi = self.indata.nsignals
        self.npou = 0

        self.params = np.array([s for s in self.indata.signals])

        self.allowNegativeParam = allowNegativeParam

        self.is_linear = self.nparams == 0 or self.allowNegativeParam

        self.set_param_default(expectSignal, allowNegativeParam)

    def compute(self, param, full=False):
        rnorm = tf.concat(
            [
                param,
                tf.ones([self.indata.nproc - param.shape[0]], dtype=self.indata.dtype),
            ],
            axis=0,
        )

        rnorm = tf.reshape(rnorm, [1, -1])
        return rnorm


class Mixture(ParamModel):
    """
    Based on unconstrained parameters x_i
    multiply `primary` process by x_i
    multiply `complementary` process by 1-x_i
    """

    def __init__(
        self,
        indata,
        primary_processes,
        complementary_processes,
        expectSignal=None,
        allowNegativeParam=False,
        **kwargs,
    ):
        self.indata = indata

        if type(primary_processes) == str:
            primary_processes = [primary_processes]

        if type(complementary_processes) == str:
            complementary_processes = [complementary_processes]

        primary_processes = np.array(primary_processes).astype("S")
        complementary_processes = np.array(complementary_processes).astype("S")

        if len(primary_processes) != len(complementary_processes):
            raise ValueError(
                f"Length of pimary and complementary processes has to be the same, but got {len(primary_processes)} and {len(complementary_processes)}"
            )

        if any(n not in self.indata.procs for n in primary_processes):
            not_found = [n for n in primary_processes if n not in self.indata.procs]
            raise ValueError(f"{not_found} not found in processes {self.indata.procs}")

        if any(n not in self.indata.procs for n in complementary_processes):
            not_found = [
                n for n in complementary_processes if n not in self.indata.procs
            ]
            raise ValueError(f"{not_found} not found in processes {self.indata.procs}")

        self.primary_idxs = np.where(np.isin(self.indata.procs, primary_processes))[0]
        self.complementary_idxs = np.where(
            np.isin(self.indata.procs, complementary_processes)
        )[0]
        self.all_idx = np.concatenate([self.primary_idxs, self.complementary_idxs])

        self.npoi = len(primary_processes)
        self.npou = 0
        self.params = np.array(
            [
                f"{p}_{c}_mixing".encode()
                for p, c in zip(
                    primary_processes.astype(str), complementary_processes.astype(str)
                )
            ]
        )

        self.allowNegativeParam = allowNegativeParam
        self.is_linear = False

        self.set_param_default(expectSignal, allowNegativeParam)

    @classmethod
    def parse_args(cls, indata, *args, **kwargs):
        """
        parsing the input arguments into the constructor, is has to be called as
        --paramModel Mixture <proc_0>,<proc_1>,... <proc_a>,<proc_b>,...
        to introduce a mixing parameter for proc_0 with proc_a, and proc_1 with proc_b, etc.
        """

        if len(args) != 2:
            raise ValueError(
                f"Expected exactly 2 arguments for Mixture model but got {len(args)}"
            )

        primaries = args[0].split(",")
        complementaries = args[1].split(",")

        return cls(indata, primaries, complementaries, **kwargs)

    def compute(self, param, full=False):

        ones = tf.ones(self.nparams, dtype=self.indata.dtype)
        updates = tf.concat([ones * param, ones * (1 - param)], axis=0)

        # Single scatter update
        rnorm = tf.tensor_scatter_nd_update(
            tf.ones(self.indata.nproc, dtype=self.indata.dtype),
            self.all_idx[:, None],
            updates,
        )

        rnorm = tf.reshape(rnorm, [1, -1])
        return rnorm


class SaturatedProjectModel(ParamModel):
    """
    For computing the saturated test statistic of a projection.
    Add one free parameter for each projected bin
    """

    def __init__(
        self,
        indata,
        channel_info,
        expectSignal=None,
        allowNegativeParam=False,
        **kwargs,
    ):
        self.indata = indata
        self.channel_info_mapping = channel_info

        self.npoi = int(
            np.sum(
                [
                    np.prod([a.size for a in v["axes"]]) if len(v["axes"]) else 1
                    for v in channel_info.values()
                ]
            )
        )
        self.npou = 0

        names = []
        for k, v in self.channel_info_mapping.items():
            for idxs in itertools.product(*[range(a.size) for a in v["axes"]]):
                label = "_".join(f"{a.name}{i}" for a, i in zip(v["axes"], idxs))
                names.append(f"saturated_{k}_{label}".encode())

        self.params = np.array(names)

        self.allowNegativeParam = allowNegativeParam

        self.is_linear = self.nparams == 0 or self.allowNegativeParam

        self.set_param_default(expectSignal, allowNegativeParam)

    def compute(self, param, full=False):
        start = 0
        rnorms = []
        for k, v in self.indata.channel_info.items():
            if v["masked"] and not full:
                continue
            shape_input = [a.size for a in v["axes"]]

            irnorm = tf.ones(shape_input, dtype=self.indata.dtype)
            if k in self.channel_info_mapping.keys():
                mapping_axes = self.channel_info_mapping[k]["axes"]
                shape_mapping = [a.size if a in mapping_axes else 1 for a in v["axes"]]
                n_mapping_params = np.prod([a.size for a in mapping_axes])
                iparam = param[start : start + n_mapping_params]
                irnorm *= tf.reshape(iparam, shape_mapping)
                start += n_mapping_params

            irnorm = tf.reshape(
                irnorm,
                [
                    -1,
                ],
            )
            rnorms.append(irnorm)

        rnorm = tf.concat(rnorms, axis=0)
        rnorm = tf.reshape(rnorm, [-1, 1])

        return rnorm


class AxisNormModel(ParamModel):
    """
    One independent normalization parameter per (process, bin-combination) of a
    caller-specified set of axes, within a named channel.  Each process in
    proc_spec gets its own set of per-cell parameters; they are never shared
    across processes.  All other channels and processes are left at scale factor 1.

    Usage::

        --paramModel AxisNormModel <channel> <proc_spec> <axes> (<poiOrPou>) (constraint:<sigma>) (exp)

    where proc_spec is ``all`` or a comma-separated list of process names,
    axes is a comma-separated list of axis names, and poiOrPou defaults to
    indicates if the parameters should be pois or pous (defaults to poi)

    Example (btojpsik: independent per-cell norms for signal and flat bkg)::

        --paramModel AxisNormModel btojpsik_stuff signal,flatBkg bkmm_kaon_pt,bkmm_kaon_eta,bkmm_kaon_charge
    """

    @classmethod
    def parse_args(cls, indata, *args, **kwargs):
        args = list(args)
        constraint_sigma = None
        use_exp = False
        constraint_args = [
            i for i, arg in enumerate(args) if str(arg).startswith("constraint:")
        ]
        if len(constraint_args) > 1:
            raise ValueError(
                f"AxisNormModel accepts at most one constraint:<sigma> token, got {args}"
            )
        if constraint_args:
            arg = args.pop(constraint_args[0])
            parts = str(arg).split(":")
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid AxisNormModel constraint token '{arg}'. "
                    "Expected constraint:<sigma>"
                )
            try:
                constraint_sigma = float(parts[1])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid AxisNormModel constraint token '{arg}'. "
                    "Constraint sigma must be a number."
                ) from exc
            if constraint_sigma <= 0:
                raise ValueError(
                    f"Invalid AxisNormModel constraint token '{arg}'. "
                    "Constraint sigma must be positive."
                )
        exp_args = [i for i, arg in enumerate(args) if str(arg) == "exp"]
        if len(exp_args) > 1:
            raise ValueError(f"AxisNormModel accepts at most one exp token, got {args}")
        if exp_args:
            args.pop(exp_args[0])
            use_exp = True

        if len(args) not in (3, 4):
            raise ValueError(
                f"AxisNormModel requires exactly 3 or 4 positional arguments "
                f"(channel, proc_spec, axes[, poiOrPou][, constraint:<sigma>][, exp]) "
                f"but got {len(args)}: {args}"
            )
        channel, proc_spec, axes_csv = args[:3]
        if len(args) == 3:
            usePois = True
        else:
            if args[3] not in ("poi", "pou"):
                raise ValueError(f"poiOrPou must be poi or pou, but got {args[3]}")
            usePois = args[3] == "poi"
        return cls(
            indata,
            channel,
            proc_spec,
            axes_csv,
            usePois=usePois,
            constraint_sigma=constraint_sigma,
            use_exp=use_exp,
            **kwargs,
        )

    def __init__(
        self,
        indata,
        channel,
        proc_spec,
        axes_csv,
        usePois=True,
        constraint_sigma=None,
        use_exp=False,
        expectSignal=None,
        allowNegativeParam=False,
        **kwargs,
    ):
        self.indata = indata

        if channel not in indata.channel_info:
            raise ValueError(
                f"Channel '{channel}' not found in tensor. "
                f"Available: {list(indata.channel_info.keys())}"
            )
        self.channel = channel
        axes = indata.channel_info[channel]["axes"]
        axis_by_name = {a.name: a for a in axes}

        requested_names = [n.strip() for n in axes_csv.split(",")]
        for name in requested_names:
            if name not in axis_by_name:
                raise ValueError(
                    f"Axis '{name}' not found in channel '{channel}'. "
                    f"Available: {list(axis_by_name.keys())}"
                )
        self.requested_axis_names = set(requested_names)
        self.requested_axes = [axis_by_name[n] for n in requested_names]

        if proc_spec == "all":
            target_encoded = list(indata.procs)
        else:
            target_encoded = []
            for name in [p.strip() for p in proc_spec.split(",")]:
                encoded = name.encode() if isinstance(name, str) else name
                if encoded not in indata.procs:
                    raise ValueError(
                        f"Process '{name}' not found in tensor. "
                        f"Available: {[p.decode() if isinstance(p, bytes) else p for p in indata.procs]}"
                    )
                target_encoded.append(encoded)
        self.proc_idxs = [
            int(np.where(indata.procs == p)[0][0]) for p in target_encoded
        ]

        self.cell_shape = [a.size for a in self.requested_axes]
        self.active_cells = [
            _active_axis_cells(indata, channel, proc_idx, self.requested_axes)
            for proc_idx in self.proc_idxs
        ]
        self.n_cells = [len(cells) for cells in self.active_cells]
        self.npoi = sum(self.n_cells) if usePois else 0
        self.npou = sum(self.n_cells) if not usePois else 0
        self.sparse_param_indices = (
            np.full(len(indata.norm.values), -1, dtype=np.int32)
            if indata.sparse
            else None
        )

        names = []
        start = 0
        for proc_encoded, proc_idx, cells in zip(
            target_encoded, self.proc_idxs, self.active_cells
        ):
            proc_name = (
                proc_encoded.decode()
                if isinstance(proc_encoded, bytes)
                else str(proc_encoded)
            )
            cell_to_param = {}
            for idxs in cells:
                label = "_".join(
                    f"{a.name}{i}" for a, i in zip(self.requested_axes, idxs)
                )
                names.append(f"norm_{proc_name}_{label}".encode())
                cell_to_param[tuple(idxs)] = start
                start += 1
            if indata.sparse:
                positions, coords = _sparse_channel_entries(indata, channel, proc_idx)
                channel_axis_names = [a.name for a in axes]
                axis_positions = [
                    channel_axis_names.index(a.name) for a in self.requested_axes
                ]
                self.sparse_param_indices[positions] = [
                    cell_to_param[tuple(coord[axis_positions])] for coord in coords
                ]
        self.params = np.array(names)
        if indata.sparse:
            self.sparse_param_indices = tf.constant(
                self.sparse_param_indices, dtype=tf.int32
            )
            self.sparse_entry_mask = self.sparse_param_indices >= 0
            self.sparse_param_indices = tf.maximum(self.sparse_param_indices, 0)
        else:
            self.sparse_param_indices = tf.constant([], dtype=tf.int32)
            self.sparse_entry_mask = tf.constant([], dtype=tf.bool)
        print(
            f"AxisNormModel {channel}: {self.npoi + self.npou} active parameters "
            f"across {len(self.proc_idxs)} process(es)"
        )

        # Enforce non-negativity via x^2 or exp applied inside compute()
        # so this works correctly whether called standalone or inside a composite.
        # allowNegativeParam=True tells the fitter/composite to pass raw x through.
        self.allowNegativeParam = True
        self.is_linear = False
        self.use_exp = use_exp
        paramdefault = np.ones(self.npoi + self.npou, dtype=np.float64)
        if expectSignal is not None:
            for signal, value in expectSignal:
                encoded = signal.encode() if isinstance(signal, str) else signal
                matches = np.where(np.isin(self.params, encoded))[0]
                if len(matches) == 0:
                    raise ValueError(f"{encoded} not in list of params: {self.params}")
                paramdefault[matches[0]] = float(value)
        if self.use_exp:
            raw_paramdefault = np.log(paramdefault)
            print(f"AxisNormModel {channel}: using exp parameterization")
        else:
            raw_paramdefault = np.sqrt(paramdefault)
        self.xparamdefault = tf.constant(raw_paramdefault, dtype=self.indata.dtype)
        if constraint_sigma is None:
            constraint_weights = np.zeros(self.npoi + self.npou, dtype=np.float64)
        else:
            constraint_weights = np.full(
                self.npoi + self.npou,
                1.0 / (constraint_sigma * constraint_sigma),
                dtype=np.float64,
            )
            print(
                f"AxisNormModel {channel}: Gaussian constraints with "
                f"sigma(norm parameter)={constraint_sigma:g}"
            )
        self._param_constraint_means = tf.constant(
            raw_paramdefault, dtype=self.indata.dtype
        )
        self._param_constraint_weights = tf.constant(
            constraint_weights, dtype=self.indata.dtype
        )

    def compute(self, param, full=False):
        reshape = [
            a.size if a.name in self.requested_axis_names else 1
            for a in self.indata.channel_info[self.channel]["axes"]
        ]
        shape_input = [a.size for a in self.indata.channel_info[self.channel]["axes"]]

        rnorms = []
        for k, v in self.indata.channel_info.items():
            if v["masked"] and not full:
                continue
            nbins_channel = int(np.prod([a.size for a in v["axes"]]))
            irnorm = tf.ones(
                [nbins_channel, self.indata.nproc], dtype=self.indata.dtype
            )
            if k == self.channel:
                start = 0
                for proc_idx, cells, n_cell in zip(
                    self.proc_idxs, self.active_cells, self.n_cells
                ):
                    ipoiu = param[start : start + n_cell]
                    start += n_cell
                    # x^2
                    if self.use_exp:
                        updates = tf.exp(ipoiu)
                    else:
                        updates = tf.square(ipoiu)
                    cell_scaling = tf.tensor_scatter_nd_update(
                        tf.ones(self.cell_shape, dtype=self.indata.dtype),
                        cells,
                        updates,
                    )
                    scaling = tf.reshape(
                        tf.broadcast_to(tf.reshape(cell_scaling, reshape), shape_input),
                        [-1, 1],
                    )
                    # softplus
                    # scaling = tf.reshape(
                    #    tf.broadcast_to(tf.reshape(tf.nn.softplus(ipoiu), reshape), shape_input), [-1, 1]
                    # )
                    proc_col = tf.one_hot(
                        proc_idx, self.indata.nproc, dtype=self.indata.dtype
                    )
                    irnorm = irnorm + (scaling - 1.0) * tf.reshape(proc_col, [1, -1])
            rnorms.append(irnorm)

        return tf.concat(rnorms, axis=0)

    def compute_sparse(self, param):
        if not self.indata.sparse:
            return super().compute_sparse(param)
        raw_values = tf.gather(param, self.sparse_param_indices)
        values = tf.exp(raw_values) if self.use_exp else tf.square(raw_values)
        return tf.where(
            self.sparse_entry_mask,
            values,
            tf.ones_like(self.indata.norm.values),
        )


class AxisExpModel(ParamModel):
    """
    Per-(process, cell) exponential background param model.

    For each process in proc_spec and each bin of the cell axes, assigns two
    independent parameters (lnAmpl, slope).  In compute() produces::

        rnorm = exp(lnAmpl_ijk + slope_ijk · x_m)

    where x_m is the normalized center of shape-axis bin m (range [0, 1]).
    Both parameters are reals (allowNegativeParam always True):
      lnAmpl controls the per-cell log-amplitude (exp(lnAmpl) is the yield at x=0).
      slope < 0 gives a falling exponential, slope = 0 is flat, slope > 0 is rising.
    The flat-background case (slope = 0) is an interior point, so the Hessian is
    non-degenerate there.  All other channels and processes are left at 1.0.

    Usage::

        --paramModel AxisExpModel <channel> <proc_spec> <shape_axis> <cell_axes> (<slope_axes>) (<amplitude_axes>) (<poiOrPou>) (constraint:<lnAmpl_sigma>:<slope_sigma>)

    The optional slope_axes list can use ``axis:ngroups`` entries to share
    slopes across coarser contiguous groups of an existing cell axis, e.g.
    ``eta1:12,eta2:12,pt2:2``.
    The optional amplitude_axes list uses the same syntax to share amplitudes
    across coarser contiguous groups. If omitted, amplitudes remain per-cell.

    Example::

        --paramModel AxisExpModel btojpsik_stuff bkgExp \\
            bkmm_jpsimc_mass \\
            bkmm_kaon_pt,bkmm_kaon_eta,bkmm_kaon_charge
    """

    @classmethod
    def parse_args(cls, indata, *args, **kwargs):
        args = list(args)
        constraint_sigmas = None
        constraint_args = [
            i for i, arg in enumerate(args) if str(arg).startswith("constraint:")
        ]
        if len(constraint_args) > 1:
            raise ValueError(
                f"AxisExpModel accepts at most one constraint:<lnAmpl_sigma>:<slope_sigma> token, got {args}"
            )
        if constraint_args:
            arg = args.pop(constraint_args[0])
            parts = str(arg).split(":")
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid AxisExpModel constraint token '{arg}'. "
                    "Expected constraint:<lnAmpl_sigma>:<slope_sigma>"
                )
            try:
                constraint_sigmas = (float(parts[1]), float(parts[2]))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid AxisExpModel constraint token '{arg}'. "
                    "Constraint sigmas must be numbers."
                ) from exc
            if constraint_sigmas[0] <= 0 or constraint_sigmas[1] <= 0:
                raise ValueError(
                    f"Invalid AxisExpModel constraint token '{arg}'. "
                    "Constraint sigmas must be positive."
                )

        if len(args) not in (4, 5, 6, 7):
            raise ValueError(
                f"AxisExpModel requires 4, 5, 6, or 7 positional arguments "
                f"(channel, proc_spec, shape_axis, cell_axes[, slope_axes[, amplitude_axes], poiOrPou][, constraint:<lnAmpl_sigma>:<slope_sigma>]) "
                f"but got {len(args)}: {args}"
            )
        channel, proc_spec, shape_axis, cell_axes_csv = args[:4]
        amplitude_axes_csv = None
        if len(args) >= 5:
            if args[4] in ("poi", "pou"):
                usePois = args[4] == "poi"
                slope_axes_csv = args[5] if (len(args) == 6) else None
            elif len(args) == 5:
                usePois = True
                slope_axes_csv = args[4]
            elif len(args) == 6:
                if args[5] not in ("poi", "pou"):
                    raise ValueError(
                        f"if passing 6 arguments to AxisExpModel, require one to be poi or pou, "
                        f"but got {args}"
                    )
                usePois = args[5] == "poi"
                slope_axes_csv = args[4]
            else:
                if args[6] not in ("poi", "pou"):
                    raise ValueError(
                        f"if passing 7 arguments to AxisExpModel, require the last one to be poi or pou, "
                        f"but got {args}"
                    )
                usePois = args[6] == "poi"
                slope_axes_csv = args[4]
                amplitude_axes_csv = args[5]
        else:
            usePois = True
            slope_axes_csv = None
        return cls(
            indata,
            channel,
            proc_spec,
            shape_axis,
            cell_axes_csv,
            slope_axes_csv=slope_axes_csv,
            amplitude_axes_csv=amplitude_axes_csv,
            usePois=usePois,
            constraint_sigmas=constraint_sigmas,
            **kwargs,
        )

    def __init__(
        self,
        indata,
        channel,
        proc_spec,
        shape_axis,
        cell_axes_csv,
        slope_axes_csv=None,
        amplitude_axes_csv=None,
        usePois=True,
        constraint_sigmas=None,
        expectSignal=None,
        allowNegativeParam=False,
        **kwargs,
    ):
        self.indata = indata

        if channel not in indata.channel_info:
            raise ValueError(
                f"Channel '{channel}' not found in tensor. "
                f"Available: {list(indata.channel_info.keys())}"
            )
        self.channel = channel
        channel_axes = indata.channel_info[channel]["axes"]
        axis_by_name = {a.name: a for a in channel_axes}

        if shape_axis not in axis_by_name:
            raise ValueError(
                f"Shape axis '{shape_axis}' not found in channel '{channel}'. "
                f"Available: {list(axis_by_name.keys())}"
            )

        cell_names = [n.strip() for n in cell_axes_csv.split(",")]
        for name in cell_names:
            if name not in axis_by_name:
                raise ValueError(
                    f"Cell axis '{name}' not found in channel '{channel}'. "
                    f"Available: {list(axis_by_name.keys())}"
                )
            if name == shape_axis:
                raise ValueError(
                    f"Axis '{name}' appears in both shape_axis and cell_axes."
                )
        self.cell_axis_names = set(cell_names)
        self.cell_axes = [axis_by_name[n] for n in cell_names]
        self.shape_axis = shape_axis

        def parse_grouped_axis_specs(specs_csv, label):
            if specs_csv is None:
                specs = [(name, None) for name in cell_names]
                names = cell_names
            else:
                specs = []
                for spec in specs_csv.split(","):
                    spec = spec.strip()
                    if ":" in spec:
                        name, ngroups = spec.split(":", 1)
                        name = name.strip()
                        try:
                            ngroups = int(ngroups)
                        except ValueError as exc:
                            raise ValueError(
                                f"Invalid grouped {label} axis specification '{spec}'"
                            ) from exc
                        if ngroups <= 0:
                            raise ValueError(
                                f"Invalid grouped {label} axis specification '{spec}'"
                            )
                    else:
                        name = spec
                        ngroups = None
                    specs.append((name, ngroups))
                names = [name for name, _ in specs]
                bad = [n for n in names if n not in self.cell_axis_names]
                if bad:
                    raise ValueError(
                        f"{label.capitalize()} axes {bad} are not in cell_axes '{cell_axes_csv}'. "
                        f"{label.capitalize()} axes must be a subset of cell axes."
                    )
            return specs, names

        def make_group_indices(specs, label):
            group_indices = []
            shape = []
            param_label_names = []
            for name, ngroups in specs:
                axis = axis_by_name[name]
                if ngroups is None:
                    groups = axis.size
                else:
                    groups = min(ngroups, axis.size)
                group_index = np.zeros(axis.size, dtype=np.int32)
                for igroup, original_bins in enumerate(
                    np.array_split(np.arange(axis.size), groups)
                ):
                    group_index[original_bins] = igroup
                group_indices.append(group_index)
                shape.append(groups)
                param_label_names.append(
                    f"{name}{label}{groups}" if ngroups is not None else name
                )
            return group_indices, shape, param_label_names

        # Slope axes: subset of cell axes; default = all cell axes (per-cell slopes)
        slope_specs, slope_names = parse_grouped_axis_specs(slope_axes_csv, "slope")
        self.slope_axis_names = set(slope_names)
        self.slope_axes = [axis_by_name[n] for n in slope_names]
        (
            self.slope_group_indices,
            self.slope_shape,
            self.slope_param_label_names,
        ) = make_group_indices(slope_specs, "Slope")

        # Amplitude axes: subset of cell axes; default = all cell axes (per-cell amplitudes)
        amplitude_specs, amplitude_names = parse_grouped_axis_specs(
            amplitude_axes_csv, "amplitude"
        )
        self.amplitude_axis_names = set(amplitude_names)
        self.amplitude_axes = [axis_by_name[n] for n in amplitude_names]
        (
            self.amplitude_group_indices,
            self.amplitude_shape,
            self.amplitude_param_label_names,
        ) = make_group_indices(amplitude_specs, "Amplitude")

        if proc_spec == "all":
            target_encoded = list(indata.procs)
        else:
            target_encoded = []
            for name in [p.strip() for p in proc_spec.split(",")]:
                encoded = name.encode() if isinstance(name, str) else name
                if encoded not in indata.procs:
                    raise ValueError(
                        f"Process '{name}' not found in tensor. "
                        f"Available: {[p.decode() if isinstance(p, bytes) else p for p in indata.procs]}"
                    )
                target_encoded.append(encoded)
        self.proc_idxs = [
            int(np.where(indata.procs == p)[0][0]) for p in target_encoded
        ]

        self.cell_shape = [a.size for a in self.cell_axes]
        self.active_cells = [
            _active_axis_cells(indata, channel, proc_idx, self.cell_axes)
            for proc_idx in self.proc_idxs
        ]
        amplitude_positions = [cell_names.index(name) for name in amplitude_names]
        slope_positions = [cell_names.index(name) for name in slope_names]
        self.active_amplitude_groups = [
            (
                np.unique(
                    np.stack(
                        [
                            group_index[cells[:, position]]
                            for position, group_index in zip(
                                amplitude_positions, self.amplitude_group_indices
                            )
                        ],
                        axis=1,
                    ),
                    axis=0,
                ).astype(np.int32)
                if len(cells)
                else np.empty((0, len(amplitude_positions)), dtype=np.int32)
            )
            for cells in self.active_cells
        ]
        self.active_cell_amplitude_groups = [
            (
                np.stack(
                    [
                        group_index[cells[:, position]]
                        for position, group_index in zip(
                            amplitude_positions, self.amplitude_group_indices
                        )
                    ],
                    axis=1,
                ).astype(np.int32)
                if len(cells)
                else np.empty((0, len(amplitude_positions)), dtype=np.int32)
            )
            for cells in self.active_cells
        ]
        self.active_slope_groups = [
            (
                np.unique(
                    np.stack(
                        [
                            group_index[cells[:, position]]
                            for position, group_index in zip(
                                slope_positions, self.slope_group_indices
                            )
                        ],
                        axis=1,
                    ),
                    axis=0,
                ).astype(np.int32)
                if len(cells)
                else np.empty((0, len(slope_positions)), dtype=np.int32)
            )
            for cells in self.active_cells
        ]
        self.active_cell_slope_groups = [
            (
                np.stack(
                    [
                        group_index[cells[:, position]]
                        for position, group_index in zip(
                            slope_positions, self.slope_group_indices
                        )
                    ],
                    axis=1,
                ).astype(np.int32)
                if len(cells)
                else np.empty((0, len(slope_positions)), dtype=np.int32)
            )
            for cells in self.active_cells
        ]
        self.n_cells = [len(cells) for cells in self.active_cells]
        self.n_amplitude_groups = [
            len(groups) for groups in self.active_amplitude_groups
        ]
        self.n_slope_groups = [len(groups) for groups in self.active_slope_groups]
        self.npoi = (
            sum(self.n_amplitude_groups) + sum(self.n_slope_groups) if usePois else 0
        )
        self.npou = (
            sum(self.n_amplitude_groups) + sum(self.n_slope_groups)
            if not usePois
            else 0
        )
        if indata.sparse:
            sparse_size = len(indata.norm.values)
            self.sparse_amplitude_param_indices = np.full(
                sparse_size, -1, dtype=np.int32
            )
            self.sparse_slope_param_indices = np.full(sparse_size, -1, dtype=np.int32)
            self.sparse_shape_values = np.zeros(sparse_size, dtype=np.int32)
        else:
            self.sparse_amplitude_param_indices = None
            self.sparse_slope_param_indices = None
            self.sparse_shape_values = None

        names = []
        start = 0
        self.active_cell_amplitude_param_indices = []
        self.active_cell_slope_param_indices = []
        for (
            proc_encoded,
            proc_idx,
            cells,
            amplitude_groups,
            cell_amplitude_groups,
            slope_groups,
            cell_slope_groups,
        ) in zip(
            target_encoded,
            self.proc_idxs,
            self.active_cells,
            self.active_amplitude_groups,
            self.active_cell_amplitude_groups,
            self.active_slope_groups,
            self.active_cell_slope_groups,
        ):
            proc_name = (
                proc_encoded.decode()
                if isinstance(proc_encoded, bytes)
                else str(proc_encoded)
            )
            amplitude_to_param = {}
            amplitude_start = start
            for idxs in amplitude_groups:
                label = "_".join(
                    f"{name}{i}"
                    for name, i in zip(self.amplitude_param_label_names, idxs)
                )
                names.append(f"lnAmpl_{proc_name}_{label}".encode())
                amplitude_to_param[tuple(idxs)] = start
                start += 1
            self.active_cell_amplitude_param_indices.append(
                np.array(
                    [
                        amplitude_to_param[tuple(idxs)] - amplitude_start
                        for idxs in cell_amplitude_groups
                    ],
                    dtype=np.int32,
                )
            )
            slope_to_param = {}
            slope_start = start
            for idxs in slope_groups:
                label = "_".join(
                    f"{name}{i}"
                    for name, i in zip(self.slope_param_label_names, idxs)
                )
                names.append(f"slope_{proc_name}_{label}".encode())
                slope_to_param[tuple(idxs)] = start
                start += 1
            self.active_cell_slope_param_indices.append(
                np.array(
                    [
                        slope_to_param[tuple(idxs)] - slope_start
                        for idxs in cell_slope_groups
                    ],
                    dtype=np.int32,
                )
            )
            if indata.sparse:
                positions, coords = _sparse_channel_entries(indata, channel, proc_idx)
                channel_axis_names = [a.name for a in channel_axes]
                cell_positions = [
                    channel_axis_names.index(a.name) for a in self.cell_axes
                ]
                amplitude_positions = [
                    channel_axis_names.index(a.name) for a in self.amplitude_axes
                ]
                slope_positions = [
                    channel_axis_names.index(a.name) for a in self.slope_axes
                ]
                shape_position = channel_axis_names.index(shape_axis)
                self.sparse_amplitude_param_indices[positions] = [
                    amplitude_to_param[
                        tuple(
                            group_index[coord[position]]
                            for position, group_index in zip(
                                amplitude_positions, self.amplitude_group_indices
                            )
                        )
                    ]
                    for coord in coords
                ]
                self.sparse_slope_param_indices[positions] = [
                    slope_to_param[
                        tuple(
                            group_index[coord[position]]
                            for position, group_index in zip(
                                slope_positions, self.slope_group_indices
                            )
                        )
                    ]
                    for coord in coords
                ]
                self.sparse_shape_values[positions] = coords[:, shape_position]
        self.params = np.array(names)
        if indata.sparse:
            self.sparse_amplitude_param_indices = tf.constant(
                self.sparse_amplitude_param_indices, dtype=tf.int32
            )
            self.sparse_slope_param_indices = tf.constant(
                self.sparse_slope_param_indices, dtype=tf.int32
            )
            self.sparse_entry_mask = self.sparse_amplitude_param_indices >= 0
            self.sparse_amplitude_param_indices = tf.maximum(
                self.sparse_amplitude_param_indices, 0
            )
            self.sparse_slope_param_indices = tf.maximum(
                self.sparse_slope_param_indices, 0
            )
            self.sparse_shape_values = tf.constant(
                self.sparse_shape_values, dtype=tf.int32
            )
        else:
            self.sparse_amplitude_param_indices = tf.constant([], dtype=tf.int32)
            self.sparse_slope_param_indices = tf.constant([], dtype=tf.int32)
            self.sparse_entry_mask = tf.constant([], dtype=tf.bool)
            self.sparse_shape_values = tf.constant([], dtype=tf.int32)
        print(
            f"AxisExpModel {channel}: {sum(self.n_amplitude_groups)} active amplitudes and "
            f"{sum(self.n_slope_groups)} active slopes across "
            f"{len(self.proc_idxs)} process(es)"
        )

        # Normalized shape-axis bin centers in [0, 1]
        centers = np.asarray(axis_by_name[shape_axis].centers, dtype=np.float32)
        span = max(float(centers[-1] - centers[0]), 1e-6)
        x_m = (centers - centers[0]) / span
        self.x_m = tf.constant(x_m, dtype=indata.dtype)

        # Reshape helpers built from channel axis ordering
        full_shape = [a.size for a in channel_axes]
        self.full_shape = full_shape
        self.cell_reshape = [
            a.size if a.name in self.cell_axis_names else 1 for a in channel_axes
        ]
        self.slope_cell_reshape = [
            a.size if a.name in self.slope_axis_names else 1 for a in channel_axes
        ]
        self.shape_reshape = [
            a.size if a.name == shape_axis else 1 for a in channel_axes
        ]

        # Always unconstrained: exp(lnAmpl + slope*x) is positive for any real (lnAmpl, slope).
        self.allowNegativeParam = True
        self.is_linear = False
        # Default: lnAmpl=0 → amplitude=1, slope=0 → flat shape.
        self.xparamdefault = tf.zeros([self.npoi + self.npou], dtype=indata.dtype)
        constraint_weights = np.zeros(self.npoi + self.npou, dtype=np.float64)
        if constraint_sigmas is not None:
            ln_ampl_sigma, slope_sigma = constraint_sigmas
            weights = {
                "lnAmpl_": 1.0 / (ln_ampl_sigma * ln_ampl_sigma),
                "slope_": 1.0 / (slope_sigma * slope_sigma),
            }
            for i, name in enumerate(self.params.astype(str)):
                for prefix, weight in weights.items():
                    if name.startswith(prefix):
                        constraint_weights[i] = weight
                        break
            print(
                f"AxisExpModel {channel}: Gaussian constraints with "
                f"sigma(lnAmpl)={ln_ampl_sigma:g}, sigma(slope)={slope_sigma:g}"
            )
        self._param_constraint_means = tf.zeros(
            [self.npoi + self.npou], dtype=indata.dtype
        )
        self._param_constraint_weights = tf.constant(
            constraint_weights, dtype=indata.dtype
        )

    def compute(self, param, full=False):
        x_reshaped = tf.reshape(self.x_m, self.shape_reshape)

        rnorms = []
        for k, v in self.indata.channel_info.items():
            if v["masked"] and not full:
                continue
            nbins_channel = int(np.prod([a.size for a in v["axes"]]))
            irnorm = tf.ones(
                [nbins_channel, self.indata.nproc], dtype=self.indata.dtype
            )
            if k == self.channel:
                start = 0
                for (
                    proc_idx,
                    cells,
                    cell_amplitude_param_indices,
                    cell_slope_param_indices,
                    n_amplitude,
                    n_slope,
                ) in zip(
                    self.proc_idxs,
                    self.active_cells,
                    self.active_cell_amplitude_param_indices,
                    self.active_cell_slope_param_indices,
                    self.n_amplitude_groups,
                    self.n_slope_groups,
                ):
                    a_poiu = param[start : start + n_amplitude]
                    start += n_amplitude
                    b_poiu = param[start : start + n_slope]
                    start += n_slope
                    a_for_cells = tf.gather(a_poiu, cell_amplitude_param_indices)
                    a_cells = tf.tensor_scatter_nd_update(
                        tf.zeros(self.cell_shape, dtype=self.indata.dtype),
                        cells,
                        a_for_cells,
                    )
                    b_for_cells = tf.gather(b_poiu, cell_slope_param_indices)
                    b_cells = tf.tensor_scatter_nd_update(
                        tf.zeros(self.cell_shape, dtype=self.indata.dtype),
                        cells,
                        b_for_cells,
                    )
                    a = tf.reshape(a_cells, self.cell_reshape)
                    b = tf.reshape(b_cells, self.cell_reshape)
                    scaling = tf.reshape(
                        tf.broadcast_to(tf.exp(a + b * x_reshaped), self.full_shape),
                        [-1, 1],
                    )
                    proc_col = tf.one_hot(
                        proc_idx, self.indata.nproc, dtype=self.indata.dtype
                    )
                    irnorm = irnorm + (scaling - 1.0) * tf.reshape(proc_col, [1, -1])
            rnorms.append(irnorm)

        return tf.concat(rnorms, axis=0)

    def compute_sparse(self, param):
        if not self.indata.sparse:
            return super().compute_sparse(param)
        amplitude = tf.gather(param, self.sparse_amplitude_param_indices)
        slope = tf.gather(param, self.sparse_slope_param_indices)
        x = tf.gather(self.x_m, self.sparse_shape_values)
        values = tf.exp(amplitude + slope * x)
        return tf.where(
            self.sparse_entry_mask,
            values,
            tf.ones_like(self.indata.norm.values),
        )


class AxisSignalBackgroundModel(ParamModel):
    """
    Per-cell signal/background mixture model.

    This mirrors the low-mass binned fitter model

        N_data(cell) * [(1 - f_bkg(cell)) * signal_shape(cell, mass)
                        + f_bkg(cell) * exp_shape(cell, mass)]

    by returning multiplicative factors for one signal and one background
    process. The signal shape is the nominal signal template in each cell,
    normalized to unit integral over the shape axis. The background shape is
    an exponential normalized over the shape axis.

    Usage::

        --paramModel AxisSignalBackgroundModel <channel> <signal_proc> <background_proc> <shape_axis> <cell_axes> [constraint:<fbkg_sigma>:<slope_sigma>]

    The parameters are model nuisances (npou): one bounded background fraction
    and one bounded positive slope per active cell.
    """

    @classmethod
    def parse_args(cls, indata, *args, **kwargs):
        args = list(args)
        constraint_sigmas = None
        constraint_args = [
            i for i, arg in enumerate(args) if str(arg).startswith("constraint:")
        ]
        if len(constraint_args) > 1:
            raise ValueError(
                "AxisSignalBackgroundModel accepts at most one "
                f"constraint:<fbkg_sigma>:<slope_sigma> token, got {args}"
            )
        if constraint_args:
            arg = args.pop(constraint_args[0])
            parts = str(arg).split(":")
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid AxisSignalBackgroundModel constraint token '{arg}'. "
                    "Expected constraint:<fbkg_sigma>:<slope_sigma>"
                )
            try:
                constraint_sigmas = (float(parts[1]), float(parts[2]))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid AxisSignalBackgroundModel constraint token '{arg}'. "
                    "Constraint sigmas must be numbers."
                ) from exc
            if constraint_sigmas[0] <= 0 or constraint_sigmas[1] <= 0:
                raise ValueError(
                    f"Invalid AxisSignalBackgroundModel constraint token '{arg}'. "
                    "Constraint sigmas must be positive."
                )

        if len(args) != 5:
            raise ValueError(
                "AxisSignalBackgroundModel requires exactly 5 positional arguments "
                "(channel, signal_proc, background_proc, shape_axis, cell_axes"
                "[, constraint:<fbkg_sigma>:<slope_sigma>]) "
                f"but got {len(args)}: {args}"
            )
        return cls(indata, *args, constraint_sigmas=constraint_sigmas, **kwargs)

    def __init__(
        self,
        indata,
        channel,
        signal_proc,
        background_proc,
        shape_axis,
        cell_axes_csv,
        constraint_sigmas=None,
        fbkg_default=0.05,
        slope_default=1.0,
        **kwargs,
    ):
        self.indata = indata
        if channel not in indata.channel_info:
            raise ValueError(
                f"Channel '{channel}' not found in tensor. "
                f"Available: {list(indata.channel_info.keys())}"
            )
        self.channel = channel
        channel_info = indata.channel_info[channel]
        self.channel_axes = channel_info["axes"]
        axis_by_name = {a.name: a for a in self.channel_axes}

        if shape_axis not in axis_by_name:
            raise ValueError(
                f"Shape axis '{shape_axis}' not found in channel '{channel}'. "
                f"Available: {list(axis_by_name.keys())}"
            )
        self.shape_axis = shape_axis
        self.shape_axis_index = [a.name for a in self.channel_axes].index(shape_axis)
        self.shape_axis_size = axis_by_name[shape_axis].size

        cell_names = [n.strip() for n in cell_axes_csv.split(",")]
        for name in cell_names:
            if name not in axis_by_name:
                raise ValueError(
                    f"Cell axis '{name}' not found in channel '{channel}'. "
                    f"Available: {list(axis_by_name.keys())}"
                )
            if name == shape_axis:
                raise ValueError(
                    f"Axis '{name}' appears in both shape_axis and cell_axes."
                )
        self.cell_axis_names = set(cell_names)
        self.cell_axes = [axis_by_name[n] for n in cell_names]
        self.cell_shape = [a.size for a in self.cell_axes]

        def proc_index(proc_name):
            encoded = proc_name.encode() if isinstance(proc_name, str) else proc_name
            if encoded not in indata.procs:
                raise ValueError(
                    f"Process '{proc_name}' not found in tensor. "
                    f"Available: {[p.decode() if isinstance(p, bytes) else p for p in indata.procs]}"
                )
            return int(np.where(indata.procs == encoded)[0][0])

        self.signal_proc_idx = proc_index(signal_proc)
        self.background_proc_idx = proc_index(background_proc)
        self.signal_proc_name = signal_proc
        self.background_proc_name = background_proc

        self.active_cells = _active_axis_cells(
            indata, channel, self.signal_proc_idx, self.cell_axes
        )
        self.n_cell = len(self.active_cells)
        self.npoi = 0
        self.npou = 2 * self.n_cell

        names = []
        cell_to_param = {}
        for i, idxs in enumerate(self.active_cells):
            label = "_".join(f"{a.name}{j}" for a, j in zip(self.cell_axes, idxs))
            names.append(f"fbkg_{background_proc}_{label}".encode())
            cell_to_param[tuple(idxs)] = i
        for idxs in self.active_cells:
            label = "_".join(f"{a.name}{j}" for a, j in zip(self.cell_axes, idxs))
            names.append(f"slope_{background_proc}_{label}".encode())
        self.params = np.array(names)

        fbkg_default = float(fbkg_default)
        slope_default = float(slope_default)
        if not 0.0 < fbkg_default < 1.0:
            raise ValueError("fbkg_default must be in (0, 1)")
        if not 0.0 < slope_default < 2.0:
            raise ValueError("slope_default must be in (0, 2)")
        fbkg_raw_default = np.log(fbkg_default / (1.0 - fbkg_default))
        slope_raw_default = np.log(slope_default / (2.0 - slope_default))
        self.xparamdefault = tf.constant(
            np.concatenate(
                [
                    np.full(self.n_cell, fbkg_raw_default, dtype=np.float64),
                    np.full(self.n_cell, slope_raw_default, dtype=np.float64),
                ]
            ),
            dtype=indata.dtype,
        )
        self._param_constraint_means = self.xparamdefault
        constraint_weights = np.zeros(self.nparams, dtype=np.float64)
        if constraint_sigmas is not None:
            fbkg_sigma, slope_sigma = constraint_sigmas
            constraint_weights[: self.n_cell] = 1.0 / (fbkg_sigma * fbkg_sigma)
            constraint_weights[self.n_cell :] = 1.0 / (slope_sigma * slope_sigma)
            print(
                f"AxisSignalBackgroundModel {channel}: Gaussian constraints with "
                f"sigma(fbkg raw)={fbkg_sigma:g}, sigma(slope raw)={slope_sigma:g}"
            )
        self._param_constraint_weights = tf.constant(
            constraint_weights, dtype=indata.dtype
        )

        self.allowNegativeParam = True
        self.is_linear = False

        full_shape = [a.size for a in self.channel_axes]
        self.full_shape = full_shape
        axis_names = [a.name for a in self.channel_axes]
        self.cell_positions = [axis_names.index(a.name) for a in self.cell_axes]

        start = channel_info["start"]
        stop = channel_info["stop"]
        channel_slice = slice(start, stop)
        channel_shape = tuple(full_shape)

        if indata.sparse:
            positions, coords = _sparse_channel_entries(
                indata, channel, self.signal_proc_idx
            )
            values = indata.norm.values.numpy()[positions]
            sig_dense = np.zeros(channel_shape, dtype=np.float64)
            sig_dense[tuple(coords.T)] = values

            bkg_positions, bkg_coords = _sparse_channel_entries(
                indata, channel, self.background_proc_idx
            )
            bkg_values = indata.norm.values.numpy()[bkg_positions]
            bkg_dense = np.zeros(channel_shape, dtype=np.float64)
            bkg_dense[tuple(bkg_coords.T)] = bkg_values
        else:
            sig_dense = (
                indata.norm.numpy()[channel_slice, self.signal_proc_idx]
                .reshape(channel_shape)
                .astype(np.float64)
            )
            bkg_dense = (
                indata.norm.numpy()[channel_slice, self.background_proc_idx]
                .reshape(channel_shape)
                .astype(np.float64)
            )

        data_dense = indata.data_obs.numpy()[channel_slice].reshape(channel_shape)
        self.data_cell_total = tf.constant(
            np.sum(data_dense, axis=self.shape_axis_index), dtype=indata.dtype
        )
        self.signal_cell_total = tf.constant(
            np.sum(sig_dense, axis=self.shape_axis_index), dtype=indata.dtype
        )
        def full_cell_index(cell):
            cell_by_axis = {
                position: value for position, value in zip(self.cell_positions, cell)
            }
            return tuple(
                slice(None) if i == self.shape_axis_index else cell_by_axis[i]
                for i in range(len(self.channel_axes))
            )

        active_data_cell_total = np.array(
            [np.sum(data_dense[full_cell_index(cell)]) for cell in self.active_cells],
            dtype=np.float64,
        )
        active_signal_cell_total = np.array(
            [np.sum(sig_dense[full_cell_index(cell)]) for cell in self.active_cells],
            dtype=np.float64,
        )
        self.active_data_cell_total = tf.constant(
            active_data_cell_total, dtype=indata.dtype
        )
        self.active_signal_cell_total_safe = tf.constant(
            np.where(active_signal_cell_total > 0, active_signal_cell_total, 1.0),
            dtype=indata.dtype,
        )

        self.background_nominal_dense = tf.constant(bkg_dense, dtype=indata.dtype)
        self.signal_cell_total_safe = tf.where(
            self.signal_cell_total > 0,
            self.signal_cell_total,
            tf.ones_like(self.signal_cell_total),
        )

        shape_axis_obj = axis_by_name[shape_axis]
        shape_metadata = getattr(shape_axis_obj, "metadata", None)
        if (
            isinstance(shape_metadata, dict)
            and "physical_centers_nd" in shape_metadata
        ):
            centers_nd = np.asarray(
                shape_metadata["physical_centers_nd"], dtype=np.float64
            )
            widths_nd = np.asarray(
                shape_metadata.get("physical_widths_nd", np.ones_like(centers_nd)),
                dtype=np.float64,
            )
            if centers_nd.shape != tuple(channel_shape):
                raise ValueError(
                    f"Physical center tensor for shape axis '{shape_axis}' has shape "
                    f"{centers_nd.shape}, expected {tuple(channel_shape)}"
                )
            if widths_nd.shape != tuple(channel_shape):
                raise ValueError(
                    f"Physical width tensor for shape axis '{shape_axis}' has shape "
                    f"{widths_nd.shape}, expected {tuple(channel_shape)}"
                )

            active_centers = np.asarray(
                [centers_nd[full_cell_index(cell)] for cell in self.active_cells],
                dtype=np.float64,
            )
            active_widths = np.asarray(
                [widths_nd[full_cell_index(cell)] for cell in self.active_cells],
                dtype=np.float64,
            )
            span = np.maximum(active_centers[:, -1] - active_centers[:, 0], 1e-12)
            x = (active_centers - active_centers[:, :1]) / span[:, None]
            widths = active_widths
            print(
                f"AxisSignalBackgroundModel {channel}: using conditional physical "
                f"centers from metadata for shape axis '{shape_axis}'"
            )
        elif (
            isinstance(shape_metadata, dict)
            and "physical_centers" in shape_metadata
        ):
            centers = np.asarray(shape_metadata["physical_centers"], dtype=np.float64)
            widths = np.asarray(
                shape_metadata.get("physical_widths", np.ones_like(centers)),
                dtype=np.float64,
            )
            if len(centers) != self.shape_axis_size:
                raise ValueError(
                    f"Physical centers for shape axis '{shape_axis}' have length "
                    f"{len(centers)}, expected {self.shape_axis_size}"
                )
            if len(widths) != self.shape_axis_size:
                raise ValueError(
                    f"Physical widths for shape axis '{shape_axis}' have length "
                    f"{len(widths)}, expected {self.shape_axis_size}"
                )
            print(
                f"AxisSignalBackgroundModel {channel}: using physical centers "
                f"from metadata for shape axis '{shape_axis}'"
            )
            x = (centers - centers[0]) / max(float(centers[-1] - centers[0]), 1e-12)
        else:
            centers = np.asarray(shape_axis_obj.centers, dtype=np.float64)
            widths = np.asarray(shape_axis_obj.widths, dtype=np.float64)
            x = (centers - centers[0]) / max(float(centers[-1] - centers[0]), 1e-12)
        self.shape_x = tf.constant(x, dtype=indata.dtype)
        self.shape_widths = tf.constant(widths, dtype=indata.dtype)

        if indata.sparse:
            sparse_size = len(indata.norm.values)
            self.sparse_param_indices = np.full(sparse_size, -1, dtype=np.int32)
            self.sparse_shape_values = np.zeros(sparse_size, dtype=np.int32)
            self.sparse_proc_codes = np.zeros(sparse_size, dtype=np.int32)
            self.sparse_nominal_values = indata.norm.values.numpy().astype(np.float64)
            for proc_idx, proc_code in [
                (self.signal_proc_idx, 1),
                (self.background_proc_idx, 2),
            ]:
                positions, coords = _sparse_channel_entries(indata, channel, proc_idx)
                for position, coord in zip(positions, coords):
                    cell = tuple(coord[self.cell_positions])
                    iparam = cell_to_param.get(cell, -1)
                    if iparam < 0:
                        continue
                    self.sparse_param_indices[position] = iparam
                    self.sparse_shape_values[position] = coord[self.shape_axis_index]
                    self.sparse_proc_codes[position] = proc_code
            self.sparse_entry_mask = tf.constant(
                self.sparse_param_indices >= 0, dtype=tf.bool
            )
            self.sparse_param_indices = tf.constant(
                np.maximum(self.sparse_param_indices, 0), dtype=tf.int32
            )
            self.sparse_shape_values = tf.constant(
                self.sparse_shape_values, dtype=tf.int32
            )
            self.sparse_proc_codes = tf.constant(self.sparse_proc_codes, dtype=tf.int32)
            self.sparse_nominal_values = tf.constant(
                self.sparse_nominal_values, dtype=indata.dtype
            )
        else:
            self.sparse_entry_mask = tf.constant([], dtype=tf.bool)
            self.sparse_param_indices = tf.constant([], dtype=tf.int32)
            self.sparse_shape_values = tf.constant([], dtype=tf.int32)
            self.sparse_proc_codes = tf.constant([], dtype=tf.int32)
            self.sparse_nominal_values = tf.constant([], dtype=indata.dtype)

        print(
            f"AxisSignalBackgroundModel {channel}: {self.n_cell} active "
            f"signal/background mixture cells"
        )

    def _fbkg_slope(self, param):
        fbkg = tf.math.sigmoid(param[: self.n_cell])
        slope = 2.0 * tf.math.sigmoid(param[self.n_cell :])
        return fbkg, slope

    def _background_mass_counts(self, fbkg, slope):
        shape_x = self.shape_x
        shape_widths = self.shape_widths
        if shape_x.shape.rank == 1:
            shape_x = tf.reshape(shape_x, [1, -1])
            shape_widths = tf.reshape(shape_widths, [1, -1])
        raw = tf.exp(tf.reshape(-slope, [-1, 1]) * shape_x)
        norm = tf.reduce_sum(raw * shape_widths, axis=1)
        pdf_counts = raw * shape_widths / norm[:, None]
        return tf.reshape(fbkg, [-1, 1]) * pdf_counts

    def compute(self, param, full=False):
        fbkg, slope = self._fbkg_slope(param)
        bkg_counts = self._background_mass_counts(fbkg, slope)

        data_cell = tf.tensor_scatter_nd_update(
            tf.zeros(self.cell_shape, dtype=self.indata.dtype),
            self.active_cells,
            tf.gather(tf.reshape(self.data_cell_total, [-1]), tf.range(self.n_cell)),
        )
        sig_total = tf.tensor_scatter_nd_update(
            tf.ones(self.cell_shape, dtype=self.indata.dtype),
            self.active_cells,
            tf.gather(tf.reshape(self.signal_cell_total_safe, [-1]), tf.range(self.n_cell)),
        )
        fbkg_cells = tf.tensor_scatter_nd_update(
            tf.zeros(self.cell_shape, dtype=self.indata.dtype),
            self.active_cells,
            fbkg,
        )
        bkg_cells = tf.tensor_scatter_nd_update(
            tf.zeros(self.cell_shape + [self.shape_axis_size], dtype=self.indata.dtype),
            np.insert(self.active_cells, self.shape_axis_index, 0, axis=1)
            if self.shape_axis_index <= len(self.cell_shape)
            else self.active_cells,
            tf.zeros([self.n_cell], dtype=self.indata.dtype),
        )
        # Dense mode is not used for the current low-mass sparse tensors.
        # Fall back to sparse logic by constructing factors in a dense flat array.
        raise NotImplementedError(
            "AxisSignalBackgroundModel dense compute is not implemented; use sparse tensors"
        )

    def compute_sparse(self, param):
        if not self.indata.sparse:
            return super().compute_sparse(param)
        fbkg, slope = self._fbkg_slope(param)
        data_cell_flat = tf.gather(
            self.active_data_cell_total, self.sparse_param_indices
        )
        signal_total_flat = tf.gather(
            self.active_signal_cell_total_safe, self.sparse_param_indices
        )
        fbkg_flat = tf.gather(fbkg, self.sparse_param_indices)
        slope_flat = tf.gather(slope, self.sparse_param_indices)

        signal_scale = (1.0 - fbkg_flat) * data_cell_flat / signal_total_flat

        if self.shape_x.shape.rank == 1:
            x = tf.gather(self.shape_x, self.sparse_shape_values)
            width = tf.gather(self.shape_widths, self.sparse_shape_values)
            raw_all = tf.exp(
                tf.reshape(-slope, [-1, 1]) * tf.reshape(self.shape_x, [1, -1])
            )
            norm_all = tf.reduce_sum(
                raw_all * tf.reshape(self.shape_widths, [1, -1]), axis=1
            )
        else:
            x = tf.gather_nd(
                self.shape_x,
                tf.stack([self.sparse_param_indices, self.sparse_shape_values], axis=1),
            )
            width = tf.gather_nd(
                self.shape_widths,
                tf.stack([self.sparse_param_indices, self.sparse_shape_values], axis=1),
            )
            raw_all = tf.exp(tf.reshape(-slope, [-1, 1]) * self.shape_x)
            norm_all = tf.reduce_sum(raw_all * self.shape_widths, axis=1)
        raw = tf.exp(-slope_flat * x)
        norm_flat = tf.gather(norm_all, self.sparse_param_indices)
        bkg_counts = fbkg_flat * data_cell_flat * raw * width / norm_flat
        bkg_scale = bkg_counts / tf.where(
            self.sparse_nominal_values > 0,
            self.sparse_nominal_values,
            tf.ones_like(self.sparse_nominal_values),
        )

        values = tf.where(
            self.sparse_proc_codes == 1,
            signal_scale,
            tf.where(self.sparse_proc_codes == 2, bkg_scale, tf.ones_like(bkg_scale)),
        )
        return tf.where(
            self.sparse_entry_mask,
            values,
            tf.ones_like(self.indata.norm.values),
        )


class AxisBernsteinModel(ParamModel):
    """
    Per-(process, cell) first-order Bernstein background param model.

    For each process in proc_spec and each cell, assigns two non-negative
    parameters (c0, c1).  In compute() produces::

        rnorm(x_m) = c0 · (1 − x_m) + c1 · x_m

    where x_m is the normalized center of shape-axis bin m (range [0, 1]).
    c0 is the relative rate at the low edge of the mass window; c1 at the high
    edge.  Non-negativity is enforced via softplus applied inside compute().
    Default c0=c1=1 gives a flat unit background.
    All other channels and processes are left at 1.0.

    Usage::

        --paramModel AxisBernsteinModel <channel> <proc_spec> <shape_axis> <cell_axes>

    Example::

        --paramModel AxisBernsteinModel btojpsik_stuff bkgBernstein \\
            bkmm_jpsimc_mass \\
            bkmm_kaon_pt,bkmm_kaon_eta,bkmm_kaon_charge
    """

    @classmethod
    def parse_args(cls, indata, *args, **kwargs):
        if len(args) != 4:
            raise ValueError(
                f"AxisBernsteinModel requires exactly 4 positional arguments "
                f"(channel, proc_spec, shape_axis, cell_axes) but got {len(args)}: {args}"
            )
        channel, proc_spec, shape_axis, cell_axes_csv = args
        return cls(indata, channel, proc_spec, shape_axis, cell_axes_csv, **kwargs)

    def __init__(
        self,
        indata,
        channel,
        proc_spec,
        shape_axis,
        cell_axes_csv,
        expectSignal=None,
        allowNegativeParam=False,
        **kwargs,
    ):
        self.indata = indata

        if channel not in indata.channel_info:
            raise ValueError(
                f"Channel '{channel}' not found in tensor. "
                f"Available: {list(indata.channel_info.keys())}"
            )
        self.channel = channel
        channel_axes = indata.channel_info[channel]["axes"]
        axis_by_name = {a.name: a for a in channel_axes}

        if shape_axis not in axis_by_name:
            raise ValueError(
                f"Shape axis '{shape_axis}' not found in channel '{channel}'. "
                f"Available: {list(axis_by_name.keys())}"
            )

        cell_names = [n.strip() for n in cell_axes_csv.split(",")]
        for name in cell_names:
            if name not in axis_by_name:
                raise ValueError(
                    f"Cell axis '{name}' not found in channel '{channel}'. "
                    f"Available: {list(axis_by_name.keys())}"
                )
            if name == shape_axis:
                raise ValueError(
                    f"Axis '{name}' appears in both shape_axis and cell_axes."
                )
        self.cell_axis_names = set(cell_names)
        self.cell_axes = [axis_by_name[n] for n in cell_names]
        self.shape_axis = shape_axis

        if proc_spec == "all":
            target_encoded = list(indata.procs)
        else:
            target_encoded = []
            for name in [p.strip() for p in proc_spec.split(",")]:
                encoded = name.encode() if isinstance(name, str) else name
                if encoded not in indata.procs:
                    raise ValueError(
                        f"Process '{name}' not found in tensor. "
                        f"Available: {[p.decode() if isinstance(p, bytes) else p for p in indata.procs]}"
                    )
                target_encoded.append(encoded)
        self.proc_idxs = [
            int(np.where(indata.procs == p)[0][0]) for p in target_encoded
        ]

        cell_shape = [a.size for a in self.cell_axes]
        self.n_cell = int(np.prod(cell_shape))
        self.npoi = len(self.proc_idxs) * 2 * self.n_cell

        names = []
        for proc_encoded in target_encoded:
            proc_name = (
                proc_encoded.decode()
                if isinstance(proc_encoded, bytes)
                else str(proc_encoded)
            )
            # nominal: independent softplus endpoints
            for prefix in ("c0", "c1"):
                # alternative (lnAmpl+frac): decouples amplitude from shape, avoids 2D null space
                # for prefix in ("lnAmpl", "frac"):
                for idxs in itertools.product(*[range(s) for s in cell_shape]):
                    label = "_".join(
                        f"{a.name}{i}" for a, i in zip(self.cell_axes, idxs)
                    )
                    names.append(f"{prefix}_{proc_name}_{label}".encode())
        self.params = np.array(names)

        # Normalized shape-axis bin centers in [0, 1]
        centers = np.asarray(axis_by_name[shape_axis].centers, dtype=np.float32)
        span = max(float(centers[-1] - centers[0]), 1e-6)
        x_m = (centers - centers[0]) / span
        self.x_m = tf.constant(x_m, dtype=indata.dtype)

        # Reshape helpers built from channel axis ordering
        full_shape = [a.size for a in channel_axes]
        self.full_shape = full_shape
        self.cell_reshape = [
            a.size if a.name in self.cell_axis_names else 1 for a in channel_axes
        ]
        self.shape_reshape = [
            a.size if a.name == shape_axis else 1 for a in channel_axes
        ]

        # Non-negativity via softplus inside compute(); allowNegativeParam=True
        # so the fitter passes raw params through and squaring is not applied.
        # Default raw = softplus_inv(1) ≈ 0.5413 so c0=c1=1 (flat unit background).
        self.npou = 0
        self.allowNegativeParam = True
        self.is_linear = False
        if expectSignal is not None:
            raise ValueError(
                "AxisBernsteinModel does not support expectSignal; "
                "set initial Bernstein coefficients via --expectSignal on another model."
            )
        # nominal: softplus_inv(1) so c0=c1=1 at init (flat unit background)
        _softplus_inv_1 = float(np.log(np.exp(1.0) - 1.0))  # ≈ 0.5413
        self.xparamdefault = tf.constant(
            _softplus_inv_1 * np.ones(self.npoi), dtype=self.indata.dtype
        )
        # alternative (lnAmpl+frac): lnAmpl=0, frac=0 → flat unit background
        # self.xparamdefault = tf.constant(
        #     np.zeros(self.npoi), dtype=self.indata.dtype
        # )

    def compute(self, param, full=False):
        x_reshaped = tf.reshape(self.x_m, self.shape_reshape)

        rnorms = []
        for k, v in self.indata.channel_info.items():
            if v["masked"] and not full:
                continue
            nbins_channel = int(np.prod([a.size for a in v["axes"]]))
            irnorm = tf.ones(
                [nbins_channel, self.indata.nproc], dtype=self.indata.dtype
            )
            if k == self.channel:
                for i, proc_idx in enumerate(self.proc_idxs):
                    # nominal: independent softplus endpoints
                    c0_poi = param[
                        i * 2 * self.n_cell : i * 2 * self.n_cell + self.n_cell
                    ]
                    c1_poi = param[
                        i * 2 * self.n_cell + self.n_cell : (i + 1) * 2 * self.n_cell
                    ]
                    c0 = tf.reshape(tf.nn.softplus(c0_poi), self.cell_reshape)
                    c1 = tf.reshape(tf.nn.softplus(c1_poi), self.cell_reshape)
                    scaling = tf.reshape(
                        tf.broadcast_to(
                            c0 * (1.0 - x_reshaped) + c1 * x_reshaped,
                            self.full_shape,
                        ),
                        [-1, 1],
                    )
                    # alternative (lnAmpl+frac): amplitude decoupled from shape;
                    # 1D null space per empty cell instead of 2D, better-conditioned Hessian
                    # lnAmpl_poi = param[i * 2 * self.n_cell : i * 2 * self.n_cell + self.n_cell]
                    # frac_poi = param[i * 2 * self.n_cell + self.n_cell : (i + 1) * 2 * self.n_cell]
                    # lnAmpl = tf.reshape(lnAmpl_poi, self.cell_reshape)
                    # frac = tf.reshape(frac_poi, self.cell_reshape)
                    # scaling = tf.reshape(
                    #     tf.broadcast_to(
                    #         2.0 * tf.exp(lnAmpl) * (
                    #             tf.sigmoid(frac) * (1.0 - x_reshaped)
                    #             + tf.sigmoid(-frac) * x_reshaped
                    #         ),
                    #         self.full_shape,
                    #     ),
                    #     [-1, 1],
                    # )
                    proc_col = tf.one_hot(
                        proc_idx, self.indata.nproc, dtype=self.indata.dtype
                    )
                    irnorm = irnorm + (scaling - 1.0) * tf.reshape(proc_col, [1, -1])
            rnorms.append(irnorm)

        return tf.concat(rnorms, axis=0)
