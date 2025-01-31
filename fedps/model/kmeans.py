from math import ceil
import numpy as np
import warnings
from sklearn.cluster import KMeans as SKL_KMeans
from sklearn.cluster._k_means_common import _is_same_clustering
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_random_state
from sklearn.utils.validation import (
    check_array,
    validate_data,
    _is_arraylike_not_scalar,
)
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from .base import _ModelBase
from ..stats.mean_var import col_mean, col_var_client, col_var_server


def _H_tolerance(role, channel, X, tol):
    if tol == 0:
        return 0
    if role == "client":
        col_var_client(
            X,
            ignore_nan=False,
            channel=channel,
            send_server=True,
            recv_server=False,
        )
        server_tol = channel.recv("server_tol")
    elif role == "server":
        server_col_var = col_var_server(
            ignore_nan=False, channel=channel, send_client=False, recv_client=True
        )
        server_tol = np.mean(server_col_var) * tol
        channel.send_all("server_tol", server_tol)
    return server_tol


class KMeans(_ModelBase):
    def __init__(
        self,
        FL_type: str,
        role: str,
        channel,
        n_clusters=8,
        init="random",
        n_init="auto",
        max_iter=300,
        tol=1e-4,
        verbose=0,
        random_state=None,
        copy_x=True,
    ):
        super().__init__(FL_type, role, channel)
        self.check_channel()
        self.module = SKL_KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            copy_x=copy_x,
        )

    def fit_predict(self, X):
        return self.fit(X).predict(X)

    def _check_n_init(self, default_n_init=None):
        # n-init
        if self.module.n_init == "auto":
            if isinstance(self.module.init, str) and self.module.init == "k-means++":
                self.module._n_init = 1
            elif isinstance(self.module.init, str) and self.module.init == "random":
                self.module._n_init = default_n_init
            else:  # array-like
                self.module._n_init = 1
        else:
            self.module._n_init = self.module.n_init

        if _is_arraylike_not_scalar(self.module.init) and self.module._n_init != 1:
            warnings.warn(
                (
                    "Explicit initial center position passed: performing only"
                    f" one init in {self.module.__class__.__name__} instead of "
                    f"n_init={self.module._n_init}."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            self.module._n_init = 1

    def _H_check_params_vs_input(self, X, n_samples, default_n_init=None):
        # n_clusters
        if self.role == "server" and n_samples < self.module.n_clusters:
            raise ValueError(
                f"n_samples={n_samples} should be >= n_clusters={self.module.n_clusters}."
            )

        # tol
        self.module._tol = _H_tolerance(
            role=self.role,
            channel=self.channel,
            X=X,
            tol=self.module.tol,
        )

        self._check_n_init(default_n_init)
        self.module._algorithm = self.module.algorithm

    def _H_init_centroids(
        self,
        X,
        n_samples,
        subsample_size,
        init,
        random_state,
    ):
        if isinstance(init, str) and init == "random":
            if self.role == "client":
                seeds = random_state.choice(
                    n_samples,
                    size=subsample_size,
                    replace=False,
                )
                centers = X[seeds]
                self.channel.send("centers", centers)
                centers = self.channel.recv("centers")

            elif self.role == "server":
                centers = self.channel.recv_all("centers")
                centers = np.squeeze(centers, axis=1)
                seeds = random_state.choice(
                    len(centers),
                    size=self.module.n_clusters,
                    replace=False,
                )
                centers = centers[seeds]
                self.channel.send_all("centers", centers)

        elif _is_arraylike_not_scalar(self.module.init):
            centers = init

        return centers

    def H_lloyd_iter_dense(
        self,
        X,
        centers_old,
        centers_new,
        weight_in_clusters,
        labels,
        update_centers=True,
    ):
        if self.role == "client":
            centers_squared_norms = np.sum(centers_old**2, axis=1)
            distance = -2 * X.dot(centers_old.T) + centers_squared_norms
            labels = np.argmin(distance, axis=1, out=labels)
            if update_centers:
                for j in range(centers_old.shape[0]):
                    mask = labels == j
                    centers_new[j] = np.sum(X[mask], axis=0)
                    weight_in_clusters[j] = np.sum(mask)

                self.channel.send("centers_new", centers_new)
                self.channel.send("weight_in_clusters", weight_in_clusters)
                centers_new = self.channel.recv("centers_new")

        elif self.role == "server":
            if update_centers:
                centers_new = self.channel.recv_all("centers_new")
                weight_in_clusters = self.channel.recv_all("weight_in_clusters")
                centers_new = np.sum(centers_new, axis=0)
                weight_in_clusters = np.sum(weight_in_clusters, axis=0)

                argmax_weight = np.argmax(weight_in_clusters)
                for j in range(centers_new.shape[0]):
                    if weight_in_clusters[j] > 0:
                        centers_new[j] /= weight_in_clusters[j]
                    else:
                        centers_new[j] = centers_new[argmax_weight]
                self.channel.send_all("centers_new", centers_new)
        return centers_new

    def _H_inertia_dense(
        self,
        X,
        centers,
        labels,
    ):
        if self.role == "client":
            inertia = 0
            for i in range(X.shape[0]):
                inertia += np.sum(np.square(X[i] - centers[labels[i]]))
            self.channel.send("inertia", inertia)
        elif self.role == "server":
            inertia = sum(self.channel.recv_all("inertia"))
        return inertia

    def _H_kmeans_single_lloyd(self, X, centers_init):
        n_clusters = centers_init.shape[0]

        # Buffers to avoid new allocations at each iteration.
        centers = centers_init
        centers_new = np.zeros_like(centers)

        if self.role == "client":
            labels = np.full(X.shape[0], -1, dtype=np.int32)
            labels_old = labels.copy()
            weight_in_clusters = np.zeros(n_clusters, dtype=X.dtype)
        elif self.role == "server":
            labels, weight_in_clusters = None, None

        lloyd_iter = self.H_lloyd_iter_dense
        _inertia = self._H_inertia_dense

        for i in range(self.module.max_iter):
            centers_new = lloyd_iter(
                X,
                centers,
                centers_new,
                weight_in_clusters,
                labels,
            )

            if self.module.verbose:
                inertia = _inertia(X, centers, labels)
                print(f"Iteration {i}, inertia {inertia}.")

            centers, centers_new = centers_new, centers

            if self.role == "client":
                label_equal = np.array_equal(labels, labels_old)
                self.channel.send("label_equal", label_equal)
                strict_convergence = self.channel.recv("strict_convergence")
            elif self.role == "server":
                label_equal = self.channel.recv_all("label_equal")
                strict_convergence = True if np.all(label_equal) else False
                self.channel.send_all("strict_convergence", strict_convergence)

            if strict_convergence:
                # First check the labels for strict convergence.
                if self.module.verbose:
                    print(f"Converged at iteration {i}: strict convergence.")
                break
            else:
                # No strict convergence, check for tol based convergence.
                center_shift_tot = ((centers_new - centers) ** 2).sum()
                if center_shift_tot <= self.module._tol:
                    if self.module.verbose:
                        print(
                            f"Converged at iteration {i}: center shift "
                            f"{center_shift_tot} within tolerance {self.module._tol}."
                        )
                    break
            if self.role == "client":
                labels_old[:] = labels

        if not strict_convergence:
            # rerun E-step so that predicted labels match cluster centers
            lloyd_iter(
                X,
                centers,
                centers,
                weight_in_clusters,
                labels,
                update_centers=False,
            )
        inertia = _inertia(X, centers, labels)
        return labels, inertia, centers, i + 1

    def Hfit(self, X):
        self.module._validate_params()
        if self.role == "client":
            X = validate_data(
                self.module,
                X,
                dtype=[np.float64, np.float32],
                order="C",
                copy=self.module.copy_x,
            )
            n_samples = X.shape[0]
            self.channel.send("n_samples", n_samples)
        elif self.role == "server":
            n_samples = sum(self.channel.recv_all("n_samples"))

        self._H_check_params_vs_input(X, n_samples, default_n_init=10)
        random_state = check_random_state(self.module.random_state)
        self.module._n_threads = _openmp_effective_n_threads()

        # Validate init array
        init = self.module.init
        init_is_array_like = _is_arraylike_not_scalar(init)
        if init_is_array_like and self.role == "client":
            init = check_array(init, dtype=X.dtype, copy=True, order="C")
            self.module._validate_center_shape(X, init)

        # subtract of mean of x for more accurate distance computations
        X_mean = col_mean(
            FL_type=self.FL_type,
            role=self.role,
            X=X if self.role == "client" else None,
            ignore_nan=False,
            channel=self.channel,
        )
        if self.role == "client":
            X -= X_mean
        if init_is_array_like:
            init -= X_mean

        if isinstance(init, str) and init == "random":
            if self.role == "client":
                subsample_ratio = self.channel.recv("subsample_ratio")
                subsample_size = ceil(subsample_ratio * n_samples)
            elif self.role == "server":
                subsample_ratio = self.module.n_clusters / n_samples
                self.channel.send_all("subsample_ratio", subsample_ratio)
        elif init_is_array_like:
            subsample_size = None

        best_inertia, best_labels = None, None

        for i in range(self.module._n_init):
            # Initialize centers
            centers_init = self._H_init_centroids(
                X,
                n_samples,
                subsample_size if self.role == "client" else None,
                init=init,
                random_state=random_state,
            )
            if self.module.verbose:
                print("Initialization complete")

            # run a k-means once
            labels, inertia, centers, n_iter_ = self._H_kmeans_single_lloyd(
                X,
                centers_init,
            )

            # determine if these results are the best so far
            # we chose a new run if it has a better inertia and the clustering is
            # different from the best so far (it's possible that the inertia is
            # slightly better even if the clustering is the same with potentially
            # permuted labels, due to rounding errors)
            if best_inertia is not None:
                if self.role == "client":
                    same_cluster = _is_same_clustering(
                        labels, best_labels, self.module.n_clusters
                    )
                    self.channel.send("same_cluster", same_cluster)
                    update = self.channel.recv("update")
                elif self.role == "server":
                    same_cluster = self.channel.recv_all("same_cluster")
                    update = inertia < best_inertia and not np.all(same_cluster)
                    self.channel.send_all("update", update)

            if best_inertia is None or update:
                best_labels = labels
                best_centers = centers
                best_inertia = inertia
                best_n_iter = n_iter_

        if not self.module.copy_x and self.role == "client":
            X += X_mean
        best_centers += X_mean

        if self.role == "client":
            self.channel.send("set_best_labels", set(best_labels))
        elif self.role == "server":
            client_set_best_labels = self.channel.recv_all("set_best_labels")
            server_set_best_labels = set()
            for set_best_labels in client_set_best_labels:
                server_set_best_labels.update(set_best_labels)

            distinct_clusters = len(server_set_best_labels)
            if distinct_clusters < self.module.n_clusters:
                warnings.warn(
                    "Number of distinct clusters ({}) found smaller than "
                    "n_clusters ({}). Possibly due to duplicate points "
                    "in X.".format(distinct_clusters, self.module.n_clusters),
                    ConvergenceWarning,
                    stacklevel=2,
                )

        self.module.cluster_centers_ = best_centers
        self.module._n_features_out = self.module.cluster_centers_.shape[0]
        self.module.labels_ = best_labels
        self.module.inertia_ = best_inertia
        self.module.n_iter_ = best_n_iter
        return self
