# NAF


## Prerequisites

It is recommended to use an isolated **Python 3.10** environment, like [miniconda](https://docs.conda.io/en/latest/miniconda.html).

For example:

```{bash}
conda create -n NAF python=3.10
```

The NAF package can be installed using `setup.py`:

```{bash}
conda activate NAF

python setup.py develop
```

## Usage

You can find some usage examples in the [notebooks](notebooks/) directory.

Basically, a NAF model can be instantiated with the following code:

```{python}
from naf.forests import ForestKind, TaskType
from naf.naf_model import NeuralAttentionForest, NAFParams

params = NAFParams(
    kind=ForestKind.EXTRA,
    task=TaskType.REGRESSION,
    mode='end_to_end',
    n_epochs=100,
    lr=0.01,
    lam=0.0,
    target_loss_weight=1.0,
    hidden_size=16,
    n_layers=1,
    forest=dict(
        n_estimators=100,
        min_samples_leaf=1
    ),
    random_state=12345
)
model = NeuralAttentionForest(params)
```

Parameter description could be found in the [Parameters](#Parameters) section.

**Make sure**, that the input data features are standardized: it is not necessary for classical tree-based models, but improve the neural network performance much.

For training the underlying classical forest run:
```{python}
# X_train is standardized
model.fit(X_train, y_train)
```

For neural network weights optimization run:

```{python}
# X_train is the same as at the previous stage.
model.optimize_weights(X_train, y_train)
```

Another *experimental* option is to optimize the neural network on unlabeled data (just reconstruction target):

```{python}
model.optimize_weights_unlabeled(X_unlabeled)
```

Predictions can be obrained with the `predict` method:

```{python}
preds = model.predict(pt)
```

Additionally, the `need_attention_weights=True` can be passed to the `predict`
to obtain reconstructed features and attention weights:

```{python}
preds, recons, alphas, betas = model.predict(inputs, need_attention_weights=True)
```

Here:

- `inputs` is of shape `(n_samples, n_features)`;
- `alphas` is of shape `(n_samples, n_background, n_trees)`;
- `betas` is of shape `(n_samples, n_trees)`;
- `n_background` is a number of samples in the original training data set (`X_train` in this case).

Sample-to-background attention weights can be calculated by multiplying `alphas` and `betas` along *tree* dimension:

```{python}
sample_attention_weights = np.einsum('nbt,nt->nb', alphas, betas)
```


### Parameters

- `n_epochs` – number of epochs for neural network training;
- `lr` – neural network learning rate;
- `lam` – reconstruction loss weight (typically $0 \le \lambda \le 1$, $0$ means no reconstruction loss);
- `target_loss_weight` – target estimation loss weight (typically is $1$);
- `hidden_size` – size of each neural network layer;
- `n_layers` – number of neural network layers;
- `forest` – parameters of an underlying forest.
