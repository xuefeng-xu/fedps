import numpy as np
from sklearn.linear_model import BayesianRidge as SKL_BayesianRidge
from sklearn.linear_model._base import _preprocess_data
from sklearn.utils import check_array
from sklearn.utils.validation import validate_data, FLOAT_DTYPES
from .base import _ModelBase


def _H_preprocess_data_client(
    channel,
    X,
    y,
    *,
    fit_intercept,
    copy=True,
    copy_y=True,
):
    X = check_array(X, copy=copy, accept_sparse=["csr", "csc"], dtype=FLOAT_DTYPES)
    y = check_array(y, dtype=X.dtype, copy=copy_y, ensure_2d=False)

    if fit_intercept:
        channel.send("col_sum", X.sum(axis=0))
        X_offset = channel.recv("X_offset")

        X_offset = X_offset.astype(X.dtype, copy=False)
        X -= X_offset

        channel.send("y_sum", y.sum(axis=0))
        y_offset = channel.recv("y_offset")
        y -= y_offset
    else:
        X_offset = np.zeros(X.shape[1], dtype=X.dtype)
        if y.ndim == 1:
            y_offset = X.dtype.type(0)
        else:
            y_offset = np.zeros(y.shape[1], dtype=X.dtype)

    return X, y, X_offset, y_offset


def _H_preprocess_data_server(
    n_samples,
    n_features,
    channel,
    fit_intercept,
):
    if fit_intercept:
        col_sum = channel.recv_all("col_sum")
        X_offset = np.sum(col_sum, axis=0) / n_samples
        channel.send_all("X_offset", X_offset)

        y_sum = channel.recv_all("y_sum")
        y_offset = np.sum(y_sum, axis=0) / n_samples
        channel.send_all("y_offset", y_offset)
    else:
        X_offset = np.zeros(n_features)
        y_offset = 0.0

    return X_offset, y_offset


def _V_preprocess_data_client_no_y(
    X,
    *,
    fit_intercept,
    copy=True,
):
    X = check_array(X, copy=copy, accept_sparse=["csr", "csc"], dtype=FLOAT_DTYPES)

    if fit_intercept:
        X_offset = np.mean(X, axis=0)
        X_offset = X_offset.astype(X.dtype, copy=False)
        X -= X_offset
    else:
        X_offset = np.zeros(X.shape[1], dtype=X.dtype)

    return X, X_offset


def _V_preprocess_data_client_no_X(
    y,
    *,
    fit_intercept,
    copy_y=True,
):
    y = check_array(y, dtype=FLOAT_DTYPES, copy=copy_y, ensure_2d=False)

    if fit_intercept:
        y_offset = np.mean(y, axis=0)
        y_offset = y_offset.astype(y.dtype, copy=False)
        y -= y_offset
    else:
        y_offset = np.zeros(1, dtype=y.dtype)

    return y, y_offset


class BayesianRidge(_ModelBase):
    def __init__(
        self,
        FL_type: str,
        role: str,
        max_iter=300,
        tol=1.0e-3,
        alpha_1=1.0e-6,
        alpha_2=1.0e-6,
        lambda_1=1.0e-6,
        lambda_2=1.0e-6,
        alpha_init=None,
        lambda_init=None,
        compute_score=False,
        fit_intercept=True,
        copy_X=True,
        verbose=False,
        channel=None,
    ):
        super().__init__(FL_type, role, channel)
        self.check_channel()
        self.module = SKL_BayesianRidge(
            max_iter=max_iter,
            tol=tol,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            alpha_init=alpha_init,
            lambda_init=lambda_init,
            compute_score=compute_score,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            verbose=verbose,
        )

    def fit(self, X=None, y=None):
        if self.FL_type == "H":
            return self.Hfit(X, y)
        else:
            return self.Vfit(X, y)

    def predict(self, X=None):
        if self.FL_type == "H":
            return self.module.predict(X)
        else:
            if self.role == "client":
                # Edge case: client only has y and has no features
                if X is not None:
                    y_mean = self.module.predict(X, return_std=False)
                else:
                    y_mean = 0
                self.channel.send("y_mean", y_mean)
                y_mean = self.channel.recv("y_mean")

            elif self.role == "server":
                y_mean = sum(self.channel.recv_all("y_mean"))
                self.channel.send_all("y_mean", y_mean)

            return y_mean

    def _H_update_coef_server(self, XT_y, eigen_vals_, eigen_vecs_, alpha_, lambda_):
        coef_ = np.linalg.multi_dot(
            [
                eigen_vecs_,
                eigen_vecs_.T / (eigen_vals_ + lambda_ / alpha_)[:, np.newaxis],
                XT_y,
            ]
        )
        self.channel.send_all("coef_", coef_)
        rmse_ = sum(self.channel.recv_all("rmse_"))
        return coef_, rmse_

    def _H_update_coef_client(self, X, y):
        coef_ = self.channel.recv("coef_")
        rmse_ = np.sum((y - np.dot(X, coef_)) ** 2)
        self.channel.send("rmse_", rmse_)
        return coef_

    def _V_update_coef_server(
        self, y, eigen_vals_, eigen_vecs_, alpha_, lambda_, client_mask
    ):
        sigma_y = np.linalg.multi_dot(
            [
                eigen_vecs_,
                eigen_vecs_.T / (eigen_vals_ + lambda_ / alpha_)[:, np.newaxis],
                y,
            ]
        )
        self.channel.send_selected("sigma_y", sigma_y, client_mask)
        X_coef = self.channel.recv_selected("X_coef", client_mask)
        X_coef = np.sum(X_coef, axis=0)
        rmse_ = np.sum((y - X_coef) ** 2)
        return rmse_

    def _V_update_coef_client(self, X):
        sigma_y = self.channel.recv("sigma_y")
        coef_ = np.dot(X.T, sigma_y)
        X_coef = np.dot(X, coef_)
        self.channel.send("X_coef", X_coef)
        return coef_

    def Hfit(self, X, y):
        if self.role == "client":
            X, y = validate_data(
                self.module, X, y, dtype=[np.float64, np.float32], y_numeric=True
            )

            n_samples, n_features = X.shape
            self.channel.send("n_samples", n_samples)
            self.channel.send("n_features", n_features)

            X, y, X_offset_, y_offset_ = _H_preprocess_data_client(
                self.channel,
                X,
                y,
                fit_intercept=self.module.fit_intercept,
                copy=self.module.copy_X,
            )

        elif self.role == "server":
            client_n_samples = self.channel.recv_all("n_samples")
            n_samples = sum(client_n_samples)

            client_n_features = self.channel.recv_all("n_features")
            if np.ptp(client_n_features) != 0:
                raise RuntimeError(
                    "The number of features are not equal for"
                    f" all clients: {client_n_features}"
                )
            n_features = client_n_features[0]

            X_offset_, y_offset_ = _H_preprocess_data_server(
                n_samples,
                n_features,
                self.channel,
                self.module.fit_intercept,
            )

        self.module.X_offset_ = X_offset_

        # Initialization of the values of the parameters
        eps = np.finfo(np.float64).eps
        # Add `eps` in the denominator to omit division by zero if `np.var(y)`
        # is zero
        alpha_ = self.module.alpha_init
        lambda_ = self.module.lambda_init

        if alpha_ is None:
            if self.role == "client":
                self.channel.send("y_square_sum", (y**2).sum(axis=0))
                if not self.module.fit_intercept:
                    self.channel.send("y_sum", y.sum(axis=0))

                alpha_ = self.channel.recv("alpha_")

            elif self.role == "server":
                y_square_sum = self.channel.recv_all("y_square_sum")
                y_square_sum = np.sum(y_square_sum, axis=0)
                y_var = y_square_sum / n_samples

                if not self.module.fit_intercept:
                    # y has been centered if fit_intercept is True
                    y_sum = self.channel.recv_all("y_sum")
                    y_sum = np.sum(y_sum, axis=0)
                    y_var -= (y_sum / n_samples) ** 2

                alpha_ = 1.0 / (y_var + eps)
                self.channel.send_all("alpha_", alpha_)

        if lambda_ is None:
            lambda_ = 1.0

        verbose = self.module.verbose
        lambda_1 = self.module.lambda_1
        lambda_2 = self.module.lambda_2
        alpha_1 = self.module.alpha_1
        alpha_2 = self.module.alpha_2

        if self.role == "client":
            if n_samples >= n_features:
                self.channel.send("XT_X", np.dot(X.T, X))
            else:
                _, S, Vh = np.linalg.svd(X, full_matrices=False)
                self.channel.send("XT_X", (Vh, S**2))

            XT_y = np.dot(X.T, y)
            self.channel.send("XT_y", XT_y)

        elif self.role == "server":
            XT_X = np.zeros((n_features, n_features))
            client_XT_X = self.channel.recv_all("XT_X")

            for cXT_X in client_XT_X:
                if isinstance(cXT_X, tuple):
                    Vh, S2 = cXT_X
                    XT_X += Vh.T * S2 @ Vh
                else:
                    XT_X += cXT_X

            eigen_vals_, eigen_vecs_ = np.linalg.eigh(XT_X)

            XT_y = sum(self.channel.recv_all("XT_y"))

        self.module.scores_ = list()
        coef_old_ = None

        # Convergence loop of the bayesian ridge regression
        for iter_ in range(self.module.max_iter):
            # update posterior mean coef_ based on alpha_ and lambda_ and
            # compute corresponding rmse
            if self.role == "client":
                coef_ = self._H_update_coef_client(X, y)

            elif self.role == "server":
                coef_, rmse_ = self._H_update_coef_server(
                    XT_y, eigen_vals_, eigen_vecs_, alpha_, lambda_
                )

                if self.module.compute_score:
                    # compute the log marginal likelihood
                    s = self.module._log_marginal_likelihood(
                        n_samples,
                        n_features,
                        eigen_vals_,
                        alpha_,
                        lambda_,
                        coef_,
                        rmse_,
                    )
                    self.module.scores_.append(s)

                # Update alpha and lambda according to (MacKay, 1992)
                gamma_ = np.sum(
                    (alpha_ * eigen_vals_) / (lambda_ + alpha_ * eigen_vals_)
                )
                lambda_ = (gamma_ + 2 * lambda_1) / (np.sum(coef_**2) + 2 * lambda_2)
                alpha_ = (n_samples - gamma_ + 2 * alpha_1) / (rmse_ + 2 * alpha_2)

            # Check for convergence
            if iter_ != 0 and np.sum(np.abs(coef_old_ - coef_)) < self.module.tol:
                if verbose:
                    print("Convergence after ", str(iter_), " iterations")
                break
            coef_old_ = np.copy(coef_)

        self.n_iter_ = iter_ + 1

        # return regularization parameters and corresponding posterior mean,
        # log marginal likelihood and posterior covariance
        if self.role == "client":
            alpha_, lambda_ = self.channel.recv("alpha_lambda_")

            coef_ = self._H_update_coef_client(X, y)

        elif self.role == "server":
            self.channel.send_all("alpha_lambda_", (alpha_, lambda_))

            coef_, rmse_ = self._H_update_coef_server(
                XT_y, eigen_vals_, eigen_vecs_, alpha_, lambda_
            )

            if self.module.compute_score:
                # compute the log marginal likelihood
                s = self.module._log_marginal_likelihood(
                    n_samples, n_features, eigen_vals_, alpha_, lambda_, coef_, rmse_
                )
                self.module.scores_.append(s)
                self.module.scores_ = np.array(self.module.scores_)

            # posterior covariance is given by 1/alpha_ * scaled_sigma_
            scaled_sigma_ = np.dot(
                eigen_vecs_,
                eigen_vecs_.T / (eigen_vals_ + lambda_ / alpha_)[:, np.newaxis],
            )
            sigma_ = (1.0 / alpha_) * scaled_sigma_
            self.module.sigma_ = sigma_

        self.module.coef_ = coef_
        self.module.alpha_ = alpha_
        self.module.lambda_ = lambda_

        self.module._set_intercept(X_offset_, y_offset_, X_offset_.dtype.type(1))

        return self

    def Vfit(self, X, y):
        if self.role == "client":
            if y is not None:
                # Edge case: client only has y and has no features
                if X is not None:
                    X, y = validate_data(
                        self.module,
                        X,
                        y,
                        dtype=[np.float64, np.float32],
                        y_numeric=True,
                    )

                    X, y, X_offset_, y_offset_, _ = _preprocess_data(
                        X,
                        y,
                        fit_intercept=self.module.fit_intercept,
                        copy=self.module.copy_X,
                    )
                else:
                    # X is None
                    X_offset_ = None
                    y, y_offset_ = _V_preprocess_data_client_no_X(
                        y,
                        fit_intercept=self.module.fit_intercept,
                    )

            else:
                # y is None, X is not None
                X = validate_data(
                    self.module, X, y="no_validation", dtype=[np.float64, np.float32]
                )

                X, X_offset_ = _V_preprocess_data_client_no_y(
                    X,
                    fit_intercept=self.module.fit_intercept,
                    copy=self.module.copy_X,
                )

            self.module.X_offset_ = X_offset_
            self.channel.send("y", y)

            if X is not None:
                n_samples, n_features = X.shape
            else:
                n_samples = len(y)
                n_features = 0

            self.channel.send("n_samples", n_samples)
            self.channel.send("n_features", n_features)

        elif self.role == "server":
            client_y = self.channel.recv_all("y")

            y_client_idx = [i for i, cy in enumerate(client_y) if cy is not None]
            if len(y_client_idx) == 1:
                y_client_idx = y_client_idx[0]
            else:
                raise RuntimeError(f"More than one client has `y`: {y_client_idx}")
            y = client_y[y_client_idx]

            client_n_samples = self.channel.recv_all("n_samples")
            if np.ptp(client_n_samples) != 0:
                raise RuntimeError(
                    "The number of samples are not equal for"
                    f" all clients: {client_n_samples}"
                )
            n_samples = client_n_samples[0]

            client_n_features = self.channel.recv_all("n_features")
            n_features = sum(client_n_features)

            # index of client has features
            client_mask = []
            for i, n in enumerate(client_n_features):
                if n > 0:
                    client_mask.append(i)

        # Initialization of the values of the parameters
        eps = np.finfo(np.float64).eps
        # Add `eps` in the denominator to omit division by zero if `np.var(y)`
        # is zero
        alpha_ = self.module.alpha_init
        lambda_ = self.module.lambda_init

        if alpha_ is None and self.role == "server":
            alpha_ = 1.0 / (np.var(y) + eps)

        if lambda_ is None:
            lambda_ = 1.0

        verbose = self.module.verbose
        lambda_1 = self.module.lambda_1
        lambda_2 = self.module.lambda_2
        alpha_1 = self.module.alpha_1
        alpha_2 = self.module.alpha_2

        if self.role == "client" and X is not None:
            if n_features >= n_samples:
                self.channel.send("X_XT", np.dot(X, X.T))
            else:
                U, S, _ = np.linalg.svd(X, full_matrices=False)
                self.channel.send("X_XT", (U, S**2))

        elif self.role == "server":
            X_XT = np.zeros((n_samples, n_samples))
            client_X_XT = self.channel.recv_selected("X_XT", client_mask)

            for cX_XT in client_X_XT:
                if isinstance(cX_XT, tuple):
                    U, S2 = cX_XT
                    X_XT += U * S2 @ U.T
                else:
                    X_XT += cX_XT

            eigen_vals_, eigen_vecs_ = np.linalg.eigh(X_XT)

        self.module.scores_ = list()
        coef_old_ = None

        # Convergence loop of the bayesian ridge regression
        for iter_ in range(self.module.max_iter):
            # update posterior mean coef_ based on alpha_ and lambda_ and
            # compute corresponding rmse
            if self.role == "client":
                # Edge case: client only has y and has no features
                if X is not None:
                    coef_ = self._V_update_coef_client(X)

                    sum_coef_square = np.sum(coef_**2)
                    self.channel.send("sum_coef_square", sum_coef_square)

                    if iter_ != 0:
                        coef_diff = np.sum(np.abs(coef_old_ - coef_))
                        self.channel.send("coef_diff", coef_diff)
                        convergence = self.channel.recv("convergence")
                else:
                    if iter_ != 0:
                        convergence = self.channel.recv("convergence")

            elif self.role == "server":
                rmse_ = self._V_update_coef_server(
                    y, eigen_vals_, eigen_vecs_, alpha_, lambda_, client_mask
                )

                sum_coef_square = sum(
                    self.channel.recv_selected("sum_coef_square", client_mask)
                )

                if self.module.compute_score:
                    # compute the log marginal likelihood
                    s = self.module._log_marginal_likelihood(
                        n_samples,
                        n_features,
                        eigen_vals_,
                        alpha_,
                        lambda_,
                        np.sqrt(sum_coef_square),
                        rmse_,
                    )
                    self.module.scores_.append(s)

                # Update alpha and lambda according to (MacKay, 1992)
                gamma_ = np.sum(
                    (alpha_ * eigen_vals_) / (lambda_ + alpha_ * eigen_vals_)
                )
                lambda_ = (gamma_ + 2 * lambda_1) / (sum_coef_square + 2 * lambda_2)
                alpha_ = (n_samples - gamma_ + 2 * alpha_1) / (rmse_ + 2 * alpha_2)

                if iter_ != 0:
                    coef_diff = sum(
                        self.channel.recv_selected("coef_diff", client_mask)
                    )
                    convergence = coef_diff < self.module.tol
                    self.channel.send_all("convergence", convergence)

            # Check for convergence
            if iter_ != 0 and convergence:
                if verbose:
                    print("Convergence after ", str(iter_), " iterations")
                break

            if self.role == "client" and X is not None:
                coef_old_ = np.copy(coef_)

        self.n_iter_ = iter_ + 1

        # return regularization parameters and corresponding posterior mean,
        # log marginal likelihood and posterior covariance
        if self.role == "client":
            alpha_, lambda_ = self.channel.recv("alpha_lambda_")

            if X is not None:
                coef_ = self._V_update_coef_client(X)
                self.module.coef_ = coef_.astype(X_offset_.dtype, copy=False)

                if self.module.compute_score:
                    sum_coef_square = np.sum(coef_**2)
                    self.channel.send("sum_coef_square", sum_coef_square)
            else:
                self.module.coef_ = None

        elif self.role == "server":
            self.channel.send_all("alpha_lambda_", (alpha_, lambda_))

            rmse_ = self._V_update_coef_server(
                y, eigen_vals_, eigen_vecs_, alpha_, lambda_, client_mask
            )

            if self.module.compute_score:
                # compute the log marginal likelihood
                sum_coef_square = sum(
                    self.channel.recv_selected("sum_coef_square", client_mask)
                )

                s = self.module._log_marginal_likelihood(
                    n_samples,
                    n_features,
                    eigen_vals_,
                    alpha_,
                    lambda_,
                    np.sqrt(sum_coef_square),
                    rmse_,
                )
                self.module.scores_.append(s)
                self.module.scores_ = np.array(self.module.scores_)

        self.module.alpha_ = alpha_
        self.module.lambda_ = lambda_

        self.module.intercept_ = 0.0
        if self.module.fit_intercept:
            if self.role == "client":
                if X is not None:
                    X_offset_coef_T = np.dot(X_offset_, self.module.coef_.T)
                    self.channel.send("X_offset_coef_T", X_offset_coef_T)

                if y is not None:
                    X_offset_coef_T = self.channel.recv("X_offset_coef_T")
                    self.module.intercept_ = y_offset_ - X_offset_coef_T

            elif self.role == "server":
                X_offset_coef_T = sum(
                    self.channel.recv_selected("X_offset_coef_T", client_mask)
                )
                self.channel.send_selected(
                    "X_offset_coef_T", X_offset_coef_T, y_client_idx
                )

        return self
