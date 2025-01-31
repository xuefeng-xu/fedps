import warnings
import numpy as np
from sklearn.utils.validation import check_array, FLOAT_DTYPES
from .util import check_channel, check_FL_type, check_role


def col_min(FL_type: str, role: str, X=None, ignore_nan: bool = True, channel=None):
    FL_type = check_FL_type(FL_type)
    role = check_role(role)

    if FL_type == "H":
        if role == "client":
            return col_min_client(X, ignore_nan, channel)
        else:
            return col_min_server(ignore_nan, channel)
    elif role == "client":
        return col_min_client(X, ignore_nan, send_server=False, recv_server=False)
    else:
        warnings.warn("Server doesn't have data", RuntimeWarning)


def row_min(FL_type: str, role: str, X=None, ignore_nan: bool = True, channel=None):
    FL_type = check_FL_type(FL_type)
    role = check_role(role)

    if FL_type == "V":
        if role == "client":
            return row_min_client(X, ignore_nan, channel)
        elif role == "server":
            return row_min_server(ignore_nan, channel)
    elif role == "client":
        return row_min_client(X, ignore_nan, send_server=False, recv_server=False)
    else:
        warnings.warn("Server doesn't have data", RuntimeWarning)


def col_max(FL_type: str, role: str, X=None, ignore_nan: bool = True, channel=None):
    FL_type = check_FL_type(FL_type)
    role = check_role(role)

    if FL_type == "H":
        if role == "client":
            return col_max_client(X, ignore_nan, channel)
        else:
            return col_max_server(ignore_nan, channel)
    elif role == "client":
        return col_max_client(X, ignore_nan, send_server=False, recv_server=False)
    else:
        warnings.warn("Server doesn't have data", RuntimeWarning)


def row_max(FL_type: str, role: str, X=None, ignore_nan: bool = True, channel=None):
    FL_type = check_FL_type(FL_type)
    role = check_role(role)

    if FL_type == "V":
        if role == "client":
            return row_max_client(X, ignore_nan, channel)
        else:
            return row_max_server(ignore_nan, channel)
    elif role == "client":
        return row_max_client(X, ignore_nan, send_server=False, recv_server=False)
    else:
        warnings.warn("Server doesn't have data", RuntimeWarning)


def col_min_max(FL_type: str, role: str, X=None, ignore_nan: bool = True, channel=None):
    FL_type = check_FL_type(FL_type)
    role = check_role(role)

    if FL_type == "H":
        if role == "client":
            return col_min_max_client(X, ignore_nan, channel)
        else:
            return col_min_max_server(ignore_nan, channel)
    elif role == "client":
        return col_min_max_client(X, ignore_nan, send_server=False, recv_server=False)
    else:
        warnings.warn("Server doesn't have data", RuntimeWarning)


def row_min_max(FL_type: str, role: str, X=None, ignore_nan: bool = True, channel=None):
    FL_type = check_FL_type(FL_type)
    role = check_role(role)

    if FL_type == "V":
        if role == "client":
            return row_min_max_client(X, ignore_nan, channel)
        else:
            return row_min_max_server(ignore_nan, channel)
    elif role == "client":
        return row_min_max_client(X, ignore_nan, send_server=False, recv_server=False)
    else:
        warnings.warn("Server doesn't have data", RuntimeWarning)


def col_min_client(
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
        client_col_min = np.nanmin(X, axis=0)
    else:
        client_col_min = np.min(X, axis=0)

    if send_server:
        channel.send("client_col_min", client_col_min)

    if recv_server:
        if not send_server:
            warnings.warn(
                "server_col_min=None because send_server=False",
                RuntimeWarning,
            )
        server_col_min = channel.recv("server_col_min")
        return server_col_min
    else:
        return client_col_min


def col_min_server(
    ignore_nan: bool = True,
    channel=None,
    send_client: bool = True,
    recv_client: bool = True,
):
    check_channel(channel, send_client, recv_client)

    if recv_client:
        client_col_min = channel.recv_all("client_col_min")

        if ignore_nan:
            server_col_min = np.nanmin(client_col_min, axis=0)
        else:
            server_col_min = np.min(client_col_min, axis=0)
    else:
        server_col_min = None

    if send_client:
        if not recv_client:
            warnings.warn(
                "server_col_min=None because recv_client=False",
                RuntimeWarning,
            )
        channel.send_all("server_col_min", server_col_min)
    return server_col_min


def col_max_client(
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
        client_col_max = np.nanmax(X, axis=0)
    else:
        client_col_max = np.max(X, axis=0)

    if send_server:
        channel.send("client_col_max", client_col_max)

    if recv_server:
        if not send_server:
            warnings.warn(
                "server_col_max=None because send_server=False",
                RuntimeWarning,
            )
        server_col_max = channel.recv("server_col_max")
        return server_col_max
    else:
        return client_col_max


def col_max_server(
    ignore_nan: bool = True,
    channel=None,
    send_client: bool = True,
    recv_client: bool = True,
):
    check_channel(channel, send_client, recv_client)

    if recv_client:
        client_col_max = channel.recv_all("client_col_max")

        if ignore_nan:
            server_col_max = np.nanmax(client_col_max, axis=0)
        else:
            server_col_max = np.max(client_col_max, axis=0)
    else:
        server_col_max = None

    if send_client:
        if not recv_client:
            warnings.warn(
                "server_col_max=None because recv_client=False",
                RuntimeWarning,
            )
        channel.send_all("server_col_max", server_col_max)
    return server_col_max


def col_min_max_client(
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
        client_col_min = np.nanmin(X, axis=0)
        client_col_max = np.nanmax(X, axis=0)
    else:
        client_col_min = np.min(X, axis=0)
        client_col_max = np.max(X, axis=0)

    if send_server:
        channel.send("client_col_min_max", [client_col_min, client_col_max])

    if recv_server:
        if not send_server:
            warnings.warn(
                "server_col_min_max=None because send_server=False",
                RuntimeWarning,
            )
        server_col_min, server_col_max = channel.recv("server_col_min_max")
        return server_col_min, server_col_max
    else:
        return client_col_min, client_col_max


def col_min_max_server(
    ignore_nan: bool = True,
    channel=None,
    send_client: bool = True,
    recv_client: bool = True,
):
    check_channel(channel, send_client, recv_client)

    if recv_client:
        client_col_min_max = channel.recv_all("client_col_min_max")
        client_col_min_max = np.array(client_col_min_max)

        # 0: client_col_min, 1: client_col_max
        if ignore_nan:
            server_col_min = np.nanmin(client_col_min_max[:, 0, :], axis=0)
            server_col_max = np.nanmax(client_col_min_max[:, 1, :], axis=0)
        else:
            server_col_min = np.min(client_col_min_max[:, 0, :], axis=0)
            server_col_max = np.max(client_col_min_max[:, 1, :], axis=0)
    else:
        server_col_min, server_col_max = None, None

    if send_client:
        if not recv_client:
            warnings.warn(
                "server_col_min_max=None because recv_client=False",
                RuntimeWarning,
            )
        channel.send_all("server_col_min_max", (server_col_min, server_col_max))
    return server_col_min, server_col_max


def row_min_client(
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
        client_row_min = np.nanmin(X, axis=1)
    else:
        client_row_min = np.min(X, axis=1)

    if send_server:
        channel.send("client_row_min", client_row_min)

    if recv_server:
        if not send_server:
            warnings.warn(
                "server_row_min=None because send_server=False",
                RuntimeWarning,
            )
        server_row_min = channel.recv("server_row_min")
        return server_row_min
    else:
        return client_row_min


def row_min_server(
    ignore_nan: bool = True,
    channel=None,
    send_client: bool = True,
    recv_client: bool = True,
):
    check_channel(channel, send_client, recv_client)

    if recv_client:
        client_row_min = channel.recv_all("client_row_min")

        if ignore_nan:
            server_row_min = np.nanmin(client_row_min, axis=0)
        else:
            server_row_min = np.min(client_row_min, axis=0)
    else:
        server_row_min = None

    if send_client:
        if not recv_client:
            warnings.warn(
                "server_row_min=None because recv_client=False",
                RuntimeWarning,
            )
        channel.send_all("server_row_min", server_row_min)
    return server_row_min


def row_max_client(
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
        client_row_max = np.nanmax(X, axis=1)
    else:
        client_row_max = np.max(X, axis=1)

    if send_server:
        channel.send("client_row_max", client_row_max)

    if recv_server:
        if not send_server:
            warnings.warn(
                "server_row_max=None because send_server=False",
                RuntimeWarning,
            )
        server_row_max = channel.recv("server_row_max")
        return server_row_max
    else:
        return client_row_max


def row_max_server(
    ignore_nan: bool = True,
    channel=None,
    send_client: bool = True,
    recv_client: bool = True,
):
    check_channel(channel, send_client, recv_client)

    if recv_client:
        client_row_max = channel.recv_all("client_row_max")

        if ignore_nan:
            server_row_max = np.nanmax(client_row_max, axis=0)
        else:
            server_row_max = np.max(client_row_max, axis=0)
    else:
        server_row_max = None

    if send_client:
        if not recv_client:
            warnings.warn(
                "server_row_max=None because recv_client=False",
                RuntimeWarning,
            )
        channel.send_all("server_row_max", server_row_max)
    return server_row_max


def row_min_max_client(
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
        client_row_min = np.nanmin(X, axis=1)
        client_row_max = np.nanmax(X, axis=1)
    else:
        client_row_min = np.min(X, axis=1)
        client_row_max = np.max(X, axis=1)

    if send_server:
        channel.send("client_row_min_max", [client_row_min, client_row_max])

    if recv_server:
        if not send_server:
            warnings.warn(
                "server_row_min_max=None because send_server=False",
                RuntimeWarning,
            )
        server_row_min, server_row_max = channel.recv("server_row_min_max")
        return server_row_min, server_row_max
    else:
        return client_row_min, client_row_max


def row_min_max_server(
    ignore_nan: bool = True,
    channel=None,
    send_client: bool = True,
    recv_client: bool = True,
):
    check_channel(channel, send_client, recv_client)

    if recv_client:
        client_row_min_max = channel.recv_all("client_row_min_max")
        client_row_min_max = np.array(client_row_min_max)

        # 0: client_row_min, 1: client_row_max
        if ignore_nan:
            server_row_min = np.nanmin(client_row_min_max[:, 0, :], axis=0)
            server_row_max = np.nanmax(client_row_min_max[:, 1, :], axis=0)
        else:
            server_row_min = np.min(client_row_min_max[:, 0, :], axis=0)
            server_row_max = np.max(client_row_min_max[:, 1, :], axis=0)
    else:
        server_row_min, server_row_max = None, None

    if send_client:
        if not recv_client:
            warnings.warn(
                "server_row_min_max=None because recv_client=False",
                RuntimeWarning,
            )
        channel.send_all("server_row_min_max", (server_row_min, server_row_max))
    return server_row_min, server_row_max
