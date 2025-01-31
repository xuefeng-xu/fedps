import warnings
import numpy as np
from sklearn.utils.validation import check_array, FLOAT_DTYPES
from .util import check_channel, check_FL_type, check_role


def col_sum(FL_type: str, role: str, X=None, ignore_nan: bool = True, channel=None):
    FL_type = check_FL_type(FL_type)
    role = check_role(role)

    if FL_type == "H":
        if role == "client":
            return col_sum_client(X, ignore_nan, channel)
        else:
            return col_sum_server(ignore_nan, channel)
    elif role == "client":
        return col_sum_client(X, ignore_nan, send_server=False, recv_server=False)
    else:
        warnings.warn("Server doesn't have data", RuntimeWarning)


def row_sum(FL_type: str, role: str, X=None, ignore_nan: bool = True, channel=None):
    FL_type = check_FL_type(FL_type)
    role = check_role(role)

    if FL_type == "V":
        if role == "client":
            return row_sum_client(X, ignore_nan, channel)
        else:
            return row_sum_server(ignore_nan, channel)
    elif role == "client":
        return row_sum_client(X, ignore_nan, send_server=False, recv_server=False)
    else:
        warnings.warn("Server doesn't have data", RuntimeWarning)


def col_sum_client(
    X,
    ignore_nan: bool = True,
    channel=None,
    send_server: bool = True,
    recv_server: bool = True,
):
    check_channel(channel, send_server, recv_server)
    X = check_array(
        X, dtype=FLOAT_DTYPES, ensure_all_finite="allow-nan" if ignore_nan else True
    )

    if ignore_nan:
        client_col_sum = np.nansum(X, axis=0)
    else:
        client_col_sum = np.sum(X, axis=0)

    if send_server:
        channel.send("client_col_sum", client_col_sum)

    if recv_server:
        if not send_server:
            warnings.warn(
                "server_col_sum=None because send_server=False",
                RuntimeWarning,
            )
        server_col_sum = channel.recv("server_col_sum")
        return server_col_sum
    else:
        return client_col_sum


def col_sum_server(
    ignore_nan: bool = True,
    channel=None,
    send_client: bool = True,
    recv_client: bool = True,
):
    check_channel(channel, send_client, recv_client)

    if recv_client:
        client_col_sum = channel.recv_all("client_col_sum")

        if ignore_nan:
            server_col_sum = np.nansum(client_col_sum, axis=0)
        else:
            server_col_sum = np.sum(client_col_sum, axis=0)
    else:
        server_col_sum = None

    if send_client:
        if not recv_client:
            warnings.warn(
                "server_col_sum=None because recv_client=False",
                RuntimeWarning,
            )
        channel.send_all("server_col_sum", server_col_sum)
    return server_col_sum


def row_sum_client(
    X,
    ignore_nan: bool = True,
    channel=None,
    send_server: bool = True,
    recv_server: bool = True,
):
    check_channel(channel, send_server, recv_server)
    X = check_array(
        X, dtype=FLOAT_DTYPES, ensure_all_finite="allow-nan" if ignore_nan else True
    )

    if ignore_nan:
        client_row_sum = np.nansum(X, axis=1)
    else:
        client_row_sum = np.sum(X, axis=1)

    if send_server:
        channel.send("client_row_sum", client_row_sum)

    if recv_server:
        if not send_server:
            warnings.warn(
                "server_row_sum=None because send_server=False",
                RuntimeWarning,
            )
        server_row_sum = channel.recv("server_row_sum")
        return server_row_sum
    else:
        return client_row_sum


def row_sum_server(
    ignore_nan: bool = True,
    channel=None,
    send_client: bool = True,
    recv_client: bool = True,
):
    check_channel(channel, send_client, recv_client)

    if recv_client:
        client_row_sum = channel.recv_all("client_row_sum")

        if ignore_nan:
            server_row_sum = np.nansum(client_row_sum, axis=0)
        else:
            server_row_sum = np.sum(client_row_sum, axis=0)
    else:
        server_row_sum = None

    if send_client:
        if not recv_client:
            warnings.warn(
                "server_row_sum=None because recv_client=False",
                RuntimeWarning,
            )
        channel.send_all("server_row_sum", server_row_sum)
    return server_row_sum
