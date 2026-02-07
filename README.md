# FedPS: Federated data Preprocessing via aggregated Statistics

The workflow consists of five steps:
① Compute local statistics;
② Share and aggregate statistics;
③ Derive preprocessing parameters;
④ Broadcast parameters to clients;
⑤ Apply preprocessing locally.

<div align=center>
    <img src="doc/overview.svg", alt="Overview", width="60%">
</div>

## Installation

### Dependencies

- Python (>= 3.10)
- Scikit-learn (~= 1.7)
- NumPy (>= 1.20)
- DataSketches (<= 4.1.0)
- PyZMQ

### Building from source

1. Create a Python env

```bash
conda create --name fedps python=3.10
conda activate fedps
```

2. Clone this project

```bash
git clone https://github.com/xuefeng-xu/fedps.git && cd fedps
```

3. Build the project

```bash
pip install -e .
```

## Usage

1. Set up communication channels

```python
# Client1 channel
from fedps.channel import ClientChannel

channel = ClientChannel(
    local_ip="127.0.0.1", local_port=5556,
    remote_ip="127.0.0.1", remote_port=5555,
)
```

```python
# Client2 channel
from fedps.channel import ClientChannel

channel = ClientChannel(
    local_ip="127.0.0.1", local_port=5557,
    remote_ip="127.0.0.1", remote_port=5555,
)
```

```python
# Server channel
from fedps.channel import ServerChannel

channel = ServerChannel(
    local_ip="127.0.0.1", local_port=5555,
    remote_ip=["127.0.0.1", "127.0.0.1"],
    remote_port=[5556, 5557],
)
```

2. Specify `FL_type` and `role` in the preprocessor

- `FL_type`: "H" (Horizontal) or "V" (Vertical)

- `role`: "client" or "server"

```python
# Client1 code example
from fedps.preprocessing import MinMaxScaler

X = [[-1, 2], [-0.5, 6]]
est = MinMaxScaler(FL_type="H", role="client", channel=channel)
Xt = est.fit_transform(X)
print(Xt)
```

```python
# Client2 code example
from fedps.preprocessing import MinMaxScaler

X = [[0, 10], [1, 18]]
est = MinMaxScaler(FL_type="H", role="client", channel=channel)
Xt = est.fit_transform(X)
print(Xt)
```

```python
# Server code example
from fedps.preprocessing import MinMaxScaler

est = MinMaxScaler(FL_type="H", role="server", channel=channel)
est.fit()
```

3. Run the script

```bash
# Run in three terminals
python client1.py
python client2.py
python server.py
```

PS: See more cases in the [example](example) folder.

## Available preprocessing modules

- Discretization
  - [`KBinsDiscretizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html)

- Encoding
  - [`LabelEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
  - [`LabelBinarizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html)
  - [`MultiLabelBinarizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html)
  - [`OneHotEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
  - [`OrdinalEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html)
  - [`TargetEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html)

- Scaling
  - [`MaxAbsScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html)
  - [`MinMaxScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
  - [`Normalizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html)
  - [`RobustScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)
  - [`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

- Transformation
  - [`PowerTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html)
  - [`QuantileTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html)
  - [`SplineTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html)

- Imputation
  - [`IterativeImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html) (experimental)
  - [`KNNImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html)
  - [`SimpleImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)

## Differences from Scikit-learn

- Currently, this library does not support sparse data.
- [`KBinsDiscretizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html), [`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html), and [`SplineTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html) cannot set the `sample_weight` parameter in their fit methods.
- [`KBinsDiscretizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html) does not support the `quantile_method` parameter.
- [`IterativeImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html) does not support the `sample_posterior` and `n_nearest_features` parameters.
- [`KNNImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html) does not support custom weight funtion and distance metric.

## Acknowledgement

This project is build on [Scikit-learn](https://github.com/scikit-learn/scikit-learn).
