import jax
import jax.numpy as jnp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def l2_normalize(x, eps=1e-8):
    norms = jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x / jnp.maximum(norms, eps)

def predict_logits(act_A, W, add_bias=True):
    if add_bias:
        W_w = W[:-1]   # (D, C)
        b   = W[-1]    # (C,)
        return act_A @ W_w + b
    else:
        return act_A @ W

def predict_classes(act_A, W, add_bias=True):
    logits = predict_logits(act_A, W, add_bias=add_bias)
    return jnp.argmax(logits, axis=-1)

def accuracy(act_A, labels, W, add_bias=True):
    preds = predict_classes(act_A, W, add_bias=add_bias)
    return jnp.mean(preds == labels)

def fit_linear_classifier_closed_form(act_A, training_labels, num_classes=None,
                                      l2_reg=0.0, add_bias=True):
    X = act_A
    N, D = X.shape

    if num_classes is None:
        num_classes = int(jnp.max(training_labels)) + 1

    Y = jax.nn.one_hot(training_labels, num_classes=num_classes)

    if add_bias:
        X = jnp.concatenate([X, jnp.ones((N, 1), X.dtype)], axis=1)
        D = D + 1

    XtX = X.T @ X
    XtY = X.T @ Y

    if l2_reg > 0.0:
        XtX = XtX + l2_reg * jnp.eye(D, dtype=X.dtype)

    W = jnp.linalg.solve(XtX, XtY)
    return W


def sweep_l2_regularization(act_A, 
                            labels,
                            l2_values,
                            add_bias=True):
    """
    Sweep L2 regularization for a SINGLE dataset (act_A, labels).

    Args:
        act_A:      (N, D) features
        labels:     (N,) integer labels
        l2_values:  1D jnp.array or Python list of lambdas
        add_bias:   whether to append a bias term in the linear model

    Returns:
        l2_values:  (L,) jnp.array of lambdas (same as input, but jnp)
        accs:       (L,) jnp.array of accuracies for each lambda
        Ws:         (L, D_eff, C) jnp.array of weight matrices
                    where D_eff = D + 1 if add_bias else D
    """
    X = act_A
    y = labels

    l2_values = jnp.asarray(l2_values)
    num_l2 = l2_values.shape[0]

    num_classes = int(jnp.max(y)) + 1
    N, D = X.shape
    D_eff = D + 1 if add_bias else D

    # we'll collect results in Python lists, then stack
    accs_list = []
    Ws_list = []

    for lam in l2_values:
        W = fit_linear_classifier_closed_form(
            X,
            y,
            num_classes=num_classes,
            l2_reg=float(lam),
            add_bias=add_bias,
        )
        acc = accuracy(X, y, W, add_bias=add_bias)

        accs_list.append(acc)
        Ws_list.append(W)

    accs = jnp.stack(accs_list)                     # (L,)
    Ws = jnp.stack(Ws_list)                         # (L, D_eff, C)

    return accs, Ws


def accuracies_from_Ws(act, labels, Ws, add_bias=True):
    """
    act:   (N, D)    features of *any* dataset (train/val/test/other space)
    labels:(N,)      corresponding labels
    Ws:    (L, D_eff, C) weights from sweep_l2_regularization
           where D_eff = D + 1 if add_bias else D

    Returns:
        accs: (L,) accuracy for each W in Ws on this dataset
    """
    accs = []
    for W in Ws:
        accs.append(accuracy(act, labels, W, add_bias=add_bias))
    accs = jnp.stack(accs)
    return accs


def per_class_accuracy(act_A, labels, W, add_bias=True, num_classes=None):
    """
    act_A:  (N, D)
    labels: (N,) int32/int64, values in [0, C-1]
    W:      (D(+1), C)
    Returns:
        per_class_acc: (C,) array with accuracy for each class
        counts:        (C,) number of examples per class
    """
    preds = predict_classes(act_A, W, add_bias=add_bias).astype(jnp.int32)
    labels = labels.astype(jnp.int32)

    if num_classes is None:
        num_classes = int(jnp.max(labels)) + 1

    correct = (preds == labels).astype(jnp.float32)

    # How many samples of each class?
    counts = jnp.bincount(labels, length=num_classes)

    # How many *correct* in each class?
    correct_per_class = jnp.bincount(labels, weights=correct, length=num_classes)

    # Avoid division by zero if some class is absent
    per_class_acc = jnp.where(counts > 0,
                              correct_per_class / counts,
                              0.0)

    return per_class_acc, counts
