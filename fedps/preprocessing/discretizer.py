import warnings
import numpy as np
from math import ceil
from sklearn.preprocessing import KBinsDiscretizer as SKL_KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample
from sklearn.utils.validation import validate_data
from .base import _PreprocessBase
from .util import validate_quantile_sketch_params
from ..stats import col_min_max
from ..sketch import (
    send_local_quantile_sketch,
    merge_local_quantile_sketch,
)


class KBinsDiscretizer(_PreprocessBase):
    def __init__(
        self,
        FL_type: str,
        role: str,
        n_bins=5,
        encode="onehot",
        strategy="quantile",
        dtype=None,
        subsample=200_000,
        random_state=None,
        sketch_name="KLL",
        k=200,
        is_hra=True,
        channel=None,
    ):
        super().__init__(FL_type, role, channel)
        if self.FL_type == "H":
            self.check_channel()
        self.sketch_name = sketch_name
        self.k = k
        self.is_hra = is_hra
        self.module = SKL_KBinsDiscretizer(
            n_bins=n_bins,
            encode=encode,
            strategy=strategy,
            dtype=dtype,
            subsample=subsample,
            random_state=random_state,
        )

    def _rm_bins_small_width(self, jj, bin_edges, n_bins):
        # Remove bins whose width are too small (i.e., <= 1e-8)
        mask = np.ediff1d(bin_edges[jj], to_begin=np.inf) > 1e-8
        bin_edges[jj] = bin_edges[jj][mask]
        if len(bin_edges[jj]) - 1 != n_bins[jj]:
            warnings.warn(
                "Bins whose width are too small (i.e., <= "
                "1e-8) in feature %d are removed. Consider "
                "decreasing the number of bins." % jj
            )
            n_bins[jj] = len(bin_edges[jj]) - 1

    def Hfit(self, X):
        self.module._validate_params()
        validate_quantile_sketch_params(self)

        if self.role == "client":
            X = validate_data(self.module, X, dtype="numeric")

            n_samples, n_features = X.shape
            n_bins = self.module._validate_n_bins(n_features)

            if self.module.subsample is not None:
                self.channel.send("n_samples", n_samples)
                subsample_ratio = self.channel.recv("subsample_ratio")
                if subsample_ratio is not None:
                    X = resample(
                        X,
                        replace=False,
                        n_samples=ceil(subsample_ratio * n_samples),
                        random_state=self.module.random_state,
                    )

        elif self.role == "server":
            subsample = self.module.subsample
            if subsample is not None:
                n_samples = sum(self.channel.recv_all("n_samples"))
                if n_samples > subsample:
                    subsample_ratio = subsample / n_samples
                else:
                    subsample_ratio = None
                self.channel.send_all("subsample_ratio", subsample_ratio)

        if self.module.dtype in (np.float64, np.float32):
            output_dtype = self.module.dtype
        else:  # self.dtype is None
            if self.role == "client":
                output_dtype = X.dtype
            elif self.role == "server":
                output_dtype = np.float64

        data_min, data_max = col_min_max(
            FL_type="H",
            role=self.role,
            X=X if self.role == "client" else None,
            ignore_nan=False,
            channel=self.channel,
        )

        if self.role == "server":
            n_features = data_max.shape[0]
            n_bins = self.module._validate_n_bins(n_features)

        bin_edges = np.zeros(n_features, dtype=object)
        for jj in range(n_features):
            col_max = data_max[jj]
            col_min = data_min[jj]

            if col_min == col_max:
                warnings.warn(
                    "Feature %d is constant and will be replaced with 0." % jj
                )
                n_bins[jj] = 1
                bin_edges[jj] = np.array([-np.inf, np.inf])
                continue

            if self.module.strategy == "uniform":
                bin_edges[jj] = np.linspace(col_min, col_max, n_bins[jj] + 1)
            elif self.module.strategy == "kmeans":
                from ..model import KMeans

                # Deterministic initialization with uniform spacing
                uniform_edges = np.linspace(col_min, col_max, n_bins[jj] + 1)
                init = (uniform_edges[1:] + uniform_edges[:-1])[:, None] * 0.5

                # 1D k-means procedure
                km = KMeans(
                    FL_type="H",
                    role=self.role,
                    channel=self.channel,
                    n_clusters=n_bins[jj],
                    init=init,
                    n_init=1,
                )
                centers = km.fit(
                    X[:, jj][:, None] if self.role == "client" else None
                ).module.cluster_centers_[:, 0]
                # Must sort, centers may be unsorted even with sorted init
                centers.sort()
                bin_edges[jj] = (centers[1:] + centers[:-1]) * 0.5
                bin_edges[jj] = np.r_[col_min, bin_edges[jj], col_max]
                self._rm_bins_small_width(jj, bin_edges, n_bins)

        if self.module.strategy == "quantile":
            mask = data_min != data_max
            if self.role == "client" and np.any(mask):
                send_local_quantile_sketch(
                    X[:, mask],
                    self.channel,
                    sketch_name=self.sketch_name,
                    k=self.k,
                    is_hra=self.is_hra,
                )
                bin_edges = self.channel.recv("bin_edges")

                for jj in range(n_features):
                    new_n_bins = len(bin_edges[jj]) - 1
                    if n_bins[jj] != new_n_bins:
                        warnings.warn(
                            "The number of bins of feature %d is changed"
                            "from %d to %d." % (jj, n_bins[jj], new_n_bins)
                        )
                        n_bins[jj] = new_n_bins

            elif self.role == "server" and np.any(mask):
                sketch = merge_local_quantile_sketch(
                    channel=self.channel,
                    sketch_name=self.sketch_name,
                    k=self.k,
                    is_hra=self.is_hra,
                )

                ii = 0
                for jj in range(n_features):
                    if not mask[jj]:
                        continue

                    quantiles = np.linspace(0, 1, n_bins[jj] + 1)[1:-1]
                    if self.sketch_name == "KLL":
                        bin_edges[jj] = sketch.get_quantiles(quantiles, isk=ii).reshape(
                            -1
                        )
                    elif self.sketch_name == "REQ":
                        bin_edges[jj] = np.array(sketch[ii].get_quantiles(quantiles))

                    bin_edges[jj] = np.r_[data_min[jj], bin_edges[jj], data_max[jj]]
                    ii += 1

                    self._rm_bins_small_width(jj, bin_edges, n_bins)

                self.channel.send_all("bin_edges", bin_edges)

        self.module.bin_edges_ = bin_edges
        self.module.n_bins_ = n_bins

        if "onehot" in self.module.encode:
            self.module._encoder = OneHotEncoder(
                categories=[np.arange(i) for i in self.module.n_bins_],
                sparse_output=self.module.encode == "onehot",
                dtype=output_dtype,
            )
            # Fit the OneHotEncoder with toy datasets
            # so that it's ready for use after the KBinsDiscretizer is fitted
            self.module._encoder.fit(np.zeros((1, len(self.module.n_bins_))))
        return self
