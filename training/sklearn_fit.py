"""Fit scikit-learn / XGBoost models so that Stop can interrupt them.

``estimator.fit()`` has no callback hook, so a long forest or boosting fit
used to run to the end after Stop. The slow families are fitted in chunks
instead, checking :mod:`training.cancel` between chunks:

- forests (RandomForest, ExtraTrees): ``warm_start`` adds trees chunk by chunk;
- HistGradientBoosting: ``warm_start`` adds boosting iterations chunk by chunk;
- XGBoost: a training callback stops between boosting rounds.

Chunked forests and boosting produce the same model as one ``fit()`` call
(sklearn advances the same random state across warm starts). Other models
(linear models, k-NN) fit quickly and run in one call.
"""
from __future__ import annotations

from training import cancel

_CHUNKS = 20


def fit_with_stop(model, X, y):
    """Fit ``model`` on ``X, y`` in place and return it.

    Raises :class:`training.cancel.TrainingStopped` if Stop is pressed before
    or during the fit.
    """
    cancel.check()
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )
    from sklearn.ensemble._forest import BaseForest

    if isinstance(model, BaseForest):
        _fit_in_chunks(model, X, y, "n_estimators")
    elif isinstance(model, (HistGradientBoostingClassifier, HistGradientBoostingRegressor)):
        _fit_in_chunks(model, X, y, "max_iter", finished=_hgb_stopped_early)
    elif _is_xgboost(model):
        _fit_xgboost(model, X, y)
    else:
        model.fit(X, y)
    cancel.check()
    return model


def _fit_in_chunks(model, X, y, size_param: str, finished=None) -> None:
    total = int(model.get_params()[size_param])
    warm_start = model.get_params()["warm_start"]
    step = max(1, -(-total // _CHUNKS))  # ceil(total / _CHUNKS)
    try:
        model.set_params(warm_start=True)
        size = 0
        while size < total:
            size = min(size + step, total)
            model.set_params(**{size_param: size})
            model.fit(X, y)
            if finished and finished(model, size):
                break  # early stopping ended training; another chunk would resume it
            cancel.check()
    finally:
        model.set_params(warm_start=warm_start, **{size_param: total})


def _hgb_stopped_early(model, size: int) -> bool:
    """Whether a single fit() would have early-stopped by iteration ``size``.

    ``n_iter_ < size`` covers a stop inside the chunk. A stop due exactly at
    the chunk's last iteration isn't visible that way (the chunk ended on
    max_iter), so ask sklearn's own test on the recorded scores.
    """
    if model.n_iter_ < size:
        return True
    if not model.do_early_stopping_:
        return False
    scores = model.validation_score_ if model._use_validation_data else model.train_score_
    return bool(model._should_stop(list(scores)))


def _is_xgboost(model) -> bool:
    return type(model).__module__.split(".")[0] == "xgboost"


def _fit_xgboost(model, X, y) -> None:
    from xgboost.callback import TrainingCallback

    class _StopOnRequest(TrainingCallback):
        def after_iteration(self, model, epoch, evals_log) -> bool:
            return cancel.stop_requested()  # True ends training after this round

    callbacks = model.get_params().get("callbacks")
    model.set_params(callbacks=[*(callbacks or []), _StopOnRequest()])
    try:
        model.fit(X, y)
    finally:
        model.set_params(callbacks=callbacks)
