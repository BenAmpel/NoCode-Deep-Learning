"""Train tab tools that compare models: k-fold cross-validation, the
significance test, random hyperparameter search and the model sweep."""
from __future__ import annotations

import numpy as np

from models.registry import is_sklearn
from pipeline import RUN_CONFIG_FIELDS, RunConfig
from pipeline.context import RunResult
from pipeline.runner import PreflightError, run as run_stages
from training import cancel as _cancel
from training.sklearn_fit import fit_with_stop
from ui.callbacks.train import _STOPPED_MESSAGE, _format_run_error
from ui.helpers import _get_pyplot, _metric_display_value, _slug_part
from ui.layout import Components


# ── Cross-validation callback ─────────────────────────────────────────────

def run_cv(modality_val, data_path_val, text_col_val, label_col_val,
           time_col_val, window_size_val, n_frames_val, sample_rate_val,
           augmentation_val, val_split_val, subset_enabled_val, subset_percent_val, subset_seed_val, training_mode_val, model_name_val,
           task_val, epochs_val, lr_val, batch_size_val, dropout_val,
           optimizer_val, scheduler_val, use_amp_val,
           timeseries_forecast_mode_val, timeseries_forecast_horizon_val, timeseries_lag_steps_val, timeseries_rolling_window_val,
           hidden_size_val, num_layers_val, cv_k_val):
    if not data_path_val or not model_name_val:
        yield "❌ Set Data path and Model before running cross-validation."
        return

    _cancel.reset()
    yield f"Starting {int(cv_k_val)}-fold cross-validation…\n"
    lines = []

    try:
        from training.cross_val import cross_validate, format_cv_results

        # Build a model factory that captures the current UI settings
        def _model_factory(num_classes: int, input_size: int = 1):
            if is_sklearn(model_name_val):
                from models.tabular_models import get_tabular_model
                return get_tabular_model(model_name_val, num_classes=num_classes,
                                         input_size=input_size,
                                         task=task_val,
                                         dropout=float(dropout_val))
            elif modality_val == "text":
                from models.text_models import get_text_model
                return get_text_model(model_name_val, num_classes=num_classes,
                                      mode=training_mode_val,
                                      hidden_size=int(hidden_size_val),
                                      num_layers=int(num_layers_val),
                                      dropout=float(dropout_val))
            elif modality_val == "tabular":
                from models.tabular_models import get_tabular_model
                return get_tabular_model(model_name_val, num_classes=num_classes,
                                         input_size=input_size,
                                         dropout=float(dropout_val))
            elif modality_val == "timeseries":
                from models.timeseries_models import get_timeseries_model
                return get_timeseries_model(
                    model_name_val,
                    num_classes=num_classes,
                    input_size=input_size,
                    hidden_size=int(hidden_size_val),
                    num_layers=int(num_layers_val),
                    dropout=float(dropout_val),
                )
            else:
                from models.image_models import get_image_model
                return get_image_model(model_name_val, num_classes=num_classes,
                                       mode=training_mode_val,
                                       dropout=float(dropout_val))

        # Load the full dataset for cross-validation
        if modality_val == "tabular":
            import pandas as pd, numpy as np
            import torch
            from torch.utils.data import TensorDataset
            from modalities.tabular import load_tabular_data
            # We need the full dataset; load with a tiny val_split and combine
            # augmentation="none": noise must not be applied before the data
            # is split into folds, or validation folds see augmented rows.
            tl, vl, classes, prep, input_size, Xtr, ytr, Xv, yv = load_tabular_data(
                data_path_val, label_col=label_col_val, batch_size=int(batch_size_val),
                val_split=0.01, augmentation="none",
                subset_percent=float(subset_percent_val) if bool(subset_enabled_val) else 100.0,
                subset_seed=int(subset_seed_val),
                task=task_val)
            X_all = np.concatenate([Xtr, Xv])
            y_all = np.concatenate([ytr, yv])
            label_dtype = "float32" if task_val == "regression" else "int64"
            full_dataset = TensorDataset(
                torch.tensor(X_all), torch.tensor(y_all.astype(label_dtype)))
            num_classes  = len(classes)

            for prog in cross_validate(
                full_dataset,
                model_factory=lambda: _model_factory(num_classes, input_size),
                k=int(cv_k_val),
                epochs=int(epochs_val), lr=float(lr_val),
                optimizer_name=optimizer_val, task=task_val,
                batch_size=int(batch_size_val),
            ):
                if prog.get("done"):
                    result = format_cv_results(
                        prog["fold_accs"],
                        prog["mean_acc"],
                        prog["std_acc"],
                        fold_metrics=prog.get("fold_metrics"),
                        aggregate_metrics=prog.get("aggregate_metrics"),
                        task=task_val,
                    )
                    if prog.get("warning"):
                        result = f"> ⚠️ {prog['warning']}\n\n" + result
                    lines.append(result)
                else:
                    fold_line = (
                        f"Fold {prog['fold']}/{prog['total_folds']}  "
                        f"epoch {prog['epoch']}  val_metric={prog.get('val_acc', 0):.2f}"
                        + ("%" if task_val == "classification" else "")
                    )
                    lines.append(fold_line)
                yield "\n".join(lines)
        elif modality_val == "timeseries":
            import numpy as np
            import torch
            from torch.utils.data import TensorDataset
            from modalities.timeseries import prepare_timeseries_windows

            windows, labels, classes, prep, input_size = prepare_timeseries_windows(
                data_path=data_path_val,
                label_col=label_col_val,
                time_col=time_col_val if time_col_val else None,
                window_size=int(window_size_val),
                task=task_val,
                subset_percent=float(subset_percent_val) if bool(subset_enabled_val) else 100.0,
                subset_seed=int(subset_seed_val),
                forecast_mode=bool(timeseries_forecast_mode_val),
                forecast_horizon=int(timeseries_forecast_horizon_val),
                lag_steps=int(timeseries_lag_steps_val),
                rolling_window=int(timeseries_rolling_window_val),
            )
            num_classes = len(classes) if task_val == "classification" else 1
            from sklearn.model_selection import TimeSeriesSplit
            temporal_splits = list(TimeSeriesSplit(n_splits=int(cv_k_val)).split(windows))

            if is_sklearn(model_name_val):
                X_all = windows.reshape(len(windows), -1).astype(np.float32)
                y_all = labels
                from eval.metrics import classification_metric_summary, regression_metric_summary

                fold_scores = []
                fold_metrics = []
                for fold_idx, (train_idx, val_idx) in enumerate(temporal_splits, start=1):
                    model = _model_factory(num_classes=num_classes, input_size=X_all.shape[1])
                    X_train = X_all[train_idx]
                    y_train = y_all[train_idx]
                    X_val = X_all[val_idx]
                    y_val = y_all[val_idx]
                    lines.append(f"Fold {fold_idx}/{int(cv_k_val)}  fitting classical baseline…")
                    yield "\n".join(lines)
                    fit_with_stop(model, X_train, y_train)
                    y_pred = model.predict(X_val)
                    y_prob = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
                    if task_val == "classification":
                        metric_summary = classification_metric_summary(y_val, y_pred, y_prob, n_classes=num_classes)
                        score = float(metric_summary["accuracy"]) * 100.0
                    else:
                        metric_summary = regression_metric_summary(y_val, y_pred)
                        score = float(metric_summary["mae"])
                    fold_scores.append(round(score, 4))
                    fold_metrics.append(metric_summary)
                from training.cross_val import _aggregate_metric_dicts, format_cv_results
                result = format_cv_results(
                    fold_scores,
                    round(float(np.mean(fold_scores)), 4),
                    round(float(np.std(fold_scores)), 4),
                    fold_metrics=fold_metrics,
                    aggregate_metrics=_aggregate_metric_dicts(fold_metrics),
                    task=task_val,
                )
                result = "> ℹ️ Time-series cross-validation uses walk-forward splits to preserve chronology.\n\n" + result
                lines.append(result)
                yield "\n".join(lines)
            else:
                label_dtype = torch.int64 if task_val == "classification" else torch.float32
                full_dataset = TensorDataset(
                    torch.tensor(windows, dtype=torch.float32),
                    torch.tensor(labels, dtype=label_dtype),
                )
                for prog in cross_validate(
                    full_dataset,
                    model_factory=lambda: _model_factory(num_classes, input_size),
                    k=int(cv_k_val),
                    epochs=int(epochs_val), lr=float(lr_val),
                    optimizer_name=optimizer_val, task=task_val,
                    batch_size=int(batch_size_val),
                    splits=temporal_splits,
                    split_strategy_name="timeseries_split",
                ):
                    if prog.get("done"):
                        result = format_cv_results(
                            prog["fold_accs"],
                            prog["mean_acc"],
                            prog["std_acc"],
                            fold_metrics=prog.get("fold_metrics"),
                            aggregate_metrics=prog.get("aggregate_metrics"),
                            task=task_val,
                        )
                        result = "> ℹ️ Time-series cross-validation uses walk-forward splits to preserve chronology.\n\n" + result
                        lines.append(result)
                    else:
                        fold_line = (
                            f"Fold {prog['fold']}/{prog['total_folds']}  "
                            f"epoch {prog['epoch']}  val_metric={prog.get('val_acc', 0):.2f}"
                            + ("%" if task_val == "classification" else "")
                        )
                        lines.append(fold_line)
                    yield "\n".join(lines)
        else:
            yield "ℹ️ Cross-validation is currently supported for **tabular** modality only."
            return

    except _cancel.TrainingStopped:
        yield "\n".join([*lines, "⏹ Cross-validation stopped."])
    except Exception as exc:
        yield f"❌ Cross-validation failed: {exc}"


def run_cv_significance(*args):
    """Paired 10-fold significance test; Stop ends it between fits."""
    _cancel.reset()
    try:
        return _cv_significance(*args)
    except _cancel.TrainingStopped:
        return "### ⏹ Significance test stopped.", None


def _cv_significance(modality_val, data_path_val, label_col_val, time_col_val, window_size_val, subset_enabled_val, subset_percent_val, subset_seed_val, training_mode_val,
                        timeseries_forecast_mode_val, timeseries_forecast_horizon_val, timeseries_lag_steps_val, timeseries_rolling_window_val,
                        base_model_name, task_val, batch_size_val, dropout_val,
                        optimizer_val, hidden_size_val, num_layers_val,
                        epochs_val, lr_val, n_estimators_val, max_depth_val,
                        c_param_val, max_iter_val, lr_xgb_val, cv_k_val,
                        metric_name, compare_models):
    if modality_val not in {"tabular", "timeseries"}:
        return "### ⚠️ Significance testing is currently available for tabular and time-series models only.", None
    if task_val not in {"classification", "regression"}:
        return "### ⚠️ Significance testing currently supports classification or regression tasks.", None
    if int(cv_k_val) != 10:
        return "### ⚠️ Set cross-validation to **10 folds** before running the significance test.", None
    selected_models = [m for m in (compare_models or []) if m and m != base_model_name]
    if not selected_models:
        return "### ⚠️ Choose at least one additional model family to compare against the active model.", None

    import numpy as np
    import torch
    from scipy.stats import ttest_rel, wilcoxon
    from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
    from torch.utils.data import DataLoader, TensorDataset

    from eval.metrics import classification_metric_summary, regression_metric_summary
    from models.registry import is_sklearn
    from models.tabular_models import get_tabular_model
    from modalities.tabular import load_tabular_data
    from modalities.timeseries import prepare_timeseries_windows
    from training.cross_val import _collect_fold_predictions, _train_one_fold

    if modality_val == "tabular":
        _tl, _vl, classes, _prep, input_size, Xtr, ytr, Xv, yv = load_tabular_data(
            data_path_val,
            label_col=label_col_val,
            batch_size=int(batch_size_val),
            val_split=0.01,
            augmentation="none",
            subset_percent=float(subset_percent_val) if bool(subset_enabled_val) else 100.0,
            subset_seed=int(subset_seed_val),
            task=task_val,
        )
        X_all = np.concatenate([Xtr, Xv])
        y_all = np.concatenate([ytr, yv])
        ts_input_size = None
        windows_all = None
    else:
        windows_all, y_all, classes, prep_ts, ts_input_size = prepare_timeseries_windows(
            data_path=data_path_val,
            label_col=label_col_val,
            time_col=time_col_val if time_col_val else None,
            window_size=int(window_size_val),
            task=task_val,
            subset_percent=float(subset_percent_val) if bool(subset_enabled_val) else 100.0,
            subset_seed=int(subset_seed_val),
            forecast_mode=bool(timeseries_forecast_mode_val),
            forecast_horizon=int(timeseries_forecast_horizon_val),
            lag_steps=int(timeseries_lag_steps_val),
            rolling_window=int(timeseries_rolling_window_val),
        )
        X_all = windows_all.reshape(len(windows_all), -1).astype(np.float32)
        input_size = int(prep_ts.get("sklearn_input_size", X_all.shape[1]))

    split_warning = None
    if modality_val == "timeseries":
        splitter = TimeSeriesSplit(n_splits=10)
        splits = list(splitter.split(X_all))
        higher_is_better = metric_name not in {"mae", "rmse"}
    elif task_val == "classification":
        _, counts = np.unique(y_all, return_counts=True)
        min_class_size = int(counts.min()) if len(counts) else 0
        if min_class_size >= 10:
            splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            splits = list(splitter.split(X_all, y_all))
        else:
            splitter = KFold(n_splits=10, shuffle=True, random_state=42)
            splits = list(splitter.split(X_all))
            split_warning = (
                f"Used plain K-Fold because the smallest class has only {min_class_size} samples."
            )
        higher_is_better = metric_name not in {"mae", "rmse"}
    else:
        splitter = KFold(n_splits=10, shuffle=True, random_state=42)
        splits = list(splitter.split(X_all))
        higher_is_better = metric_name == "r2"

    def build_family(model_name_local):
        if modality_val == "timeseries" and not is_sklearn(model_name_local):
            from models.timeseries_models import get_timeseries_model

            return get_timeseries_model(
                model_name_local,
                num_classes=(len(classes) if task_val == "classification" else 1),
                input_size=ts_input_size,
                hidden_size=int(hidden_size_val),
                num_layers=int(num_layers_val),
                dropout=float(dropout_val),
            )
        return get_tabular_model(
            model_name_local,
            num_classes=len(classes) if task_val == "classification" else 1,
            input_size=input_size,
            task=task_val,
            dropout=float(dropout_val),
            n_estimators=int(n_estimators_val),
            max_depth=int(max_depth_val),
            C=float(c_param_val),
            max_iter=int(max_iter_val),
            learning_rate=float(lr_xgb_val),
        )

    def score_family(model_name_local):
        fold_scores = []
        for train_idx, val_idx in splits:
            if is_sklearn(model_name_local):
                X_train = X_all[train_idx]
                y_train = y_all[train_idx]
                X_val = X_all[val_idx]
                y_val = y_all[val_idx]
                model = build_family(model_name_local)
                fit_with_stop(model, X_train, y_train)
                y_pred = model.predict(X_val)
                y_prob = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
            else:
                model = build_family(model_name_local)
                if modality_val == "timeseries":
                    label_dtype = torch.int64 if task_val == "classification" else torch.float32
                    train_loader = DataLoader(
                        TensorDataset(
                            torch.tensor(windows_all[train_idx], dtype=torch.float32),
                            torch.tensor(y_all[train_idx], dtype=label_dtype),
                        ),
                        batch_size=int(batch_size_val),
                        shuffle=True,
                    )
                    val_loader = DataLoader(
                        TensorDataset(
                            torch.tensor(windows_all[val_idx], dtype=torch.float32),
                            torch.tensor(y_all[val_idx], dtype=label_dtype),
                        ),
                        batch_size=int(batch_size_val),
                        shuffle=False,
                    )
                else:
                    X_train = X_all[train_idx]
                    y_train = y_all[train_idx]
                    X_val = X_all[val_idx]
                    y_val = y_all[val_idx]
                    label_dtype = torch.int64 if task_val == "classification" else torch.float32
                    train_loader = DataLoader(
                        TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=label_dtype)),
                        batch_size=int(batch_size_val),
                        shuffle=True,
                    )
                    val_loader = DataLoader(
                        TensorDataset(torch.tensor(X_val), torch.tensor(y_val, dtype=label_dtype)),
                        batch_size=int(batch_size_val),
                        shuffle=False,
                    )
                for _ in _train_one_fold(
                    model,
                    train_loader,
                    val_loader,
                    epochs=int(epochs_val),
                    lr=float(lr_val),
                    optimizer_name=optimizer_val,
                    task=task_val,
                ):
                    pass
                y_true, y_pred, y_prob = _collect_fold_predictions(model, val_loader, task_val)
                y_val = y_true

            if task_val == "classification":
                metric_summary = classification_metric_summary(y_val, y_pred, y_prob, n_classes=len(classes))
            else:
                metric_summary = regression_metric_summary(y_val, y_pred)
            score = metric_summary.get(metric_name)
            if score is None:
                raise ValueError(f"Metric '{metric_name}' is unavailable for model '{model_name_local}'.")
            fold_scores.append(float(score))
        return np.asarray(fold_scores, dtype=float)

    baseline_scores = score_family(base_model_name)
    challenger_curves = {}
    lines = [
        "### Significance testing",
        "",
        f"- **Baseline family**: `{base_model_name}`",
        f"- **Metric**: `{metric_name}`",
        f"- **Folds**: `10` paired folds",
        f"- **Split strategy**: `{'TimeSeriesSplit' if modality_val == 'timeseries' else ('StratifiedKFold' if split_warning is None and task_val == 'classification' else 'KFold')}`",
        "",
        "| Challenger | Baseline mean | Challenger mean | Delta | Paired t-test p | Wilcoxon p | Result |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    if split_warning:
        lines.insert(5, f"- **Note**: {split_warning}")

    for challenger in selected_models:
        challenger_scores = score_family(challenger)
        challenger_curves[challenger] = challenger_scores
        if higher_is_better:
            delta = challenger_scores.mean() - baseline_scores.mean()
        else:
            delta = baseline_scores.mean() - challenger_scores.mean()
        ttest_p = float(ttest_rel(challenger_scores, baseline_scores).pvalue)
        try:
            wilcoxon_p = float(wilcoxon(challenger_scores, baseline_scores).pvalue)
        except ValueError:
            wilcoxon_p = float("nan")
        result = "Different" if (ttest_p < 0.05 or (not np.isnan(wilcoxon_p) and wilcoxon_p < 0.05)) else "No clear difference"
        lines.append(
            f"| {challenger} | {baseline_scores.mean():.4f} | {challenger_scores.mean():.4f} | {delta:+.4f} | {ttest_p:.4f} | "
            f"{('n/a' if np.isnan(wilcoxon_p) else f'{wilcoxon_p:.4f}')} | {result} |"
        )

    lines.extend([
        "",
        "Paired tests use the same 10 folds for each family. Positive delta means the challenger performed better on the selected metric.",
    ])
    plot = _plot_cv_significance_scores(metric_name, base_model_name, baseline_scores, challenger_curves)
    return "\n".join(lines), plot


# inspect.signature (Gradio, the UI wiring test) then sees the real parameters.
run_cv_significance.__wrapped__ = _cv_significance

# ── Hyperparameter search callback ────────────────────────────────────────


def run_hparam_search(modality_val, data_path_val, text_col_val, label_col_val,
                      time_col_val, window_size_val, n_frames_val, sample_rate_val,
                      augmentation_val, val_split_val, subset_enabled_val, subset_percent_val, subset_seed_val, training_mode_val,
                      model_name_val, task_val, batch_size_val, dropout_val,
                      optimizer_val, hidden_size_val, num_layers_val,
                      lr_vals_str, dropout_vals_str, n_trials_val, hparam_epochs_val):
    if not data_path_val or not model_name_val:
        yield "❌ Set Data path and Model before running hyperparameter search."
        return

    # Parse candidate values
    try:
        lr_candidates      = [float(v.strip()) for v in lr_vals_str.split(",") if v.strip()]
        dropout_candidates = [float(v.strip()) for v in dropout_vals_str.split(",") if v.strip()]
    except ValueError as e:
        yield f"❌ Could not parse candidates: {e}"
        return

    param_space = {"lr": lr_candidates, "dropout": dropout_candidates}
    _cancel.reset()
    yield f"Starting random hyperparameter search over {len(lr_candidates)}×{len(dropout_candidates)} grid…\n"
    lines = []

    try:
        if modality_val != "tabular":
            yield "ℹ️ Hyperparameter search is currently supported for **tabular** modality only."
            return

        import pandas as pd, numpy as np, torch
        from torch.utils.data import TensorDataset, DataLoader
        from modalities.tabular import load_tabular_data
        from training.hparam_search import random_search, format_search_results

        tl, vl, classes, prep, input_size, Xtr, ytr, Xv, yv = load_tabular_data(
            data_path_val, label_col=label_col_val, batch_size=int(batch_size_val),
            val_split=float(val_split_val),
            subset_percent=float(subset_percent_val) if bool(subset_enabled_val) else 100.0,
            subset_seed=int(subset_seed_val),
            task=task_val)
        num_classes = len(classes)

        def _factory(params):
            from models.tabular_models import get_tabular_model
            return get_tabular_model(
                model_name_val, num_classes=num_classes,
                input_size=input_size,
                dropout=float(params.get("dropout", dropout_val)))

        all_results = []
        for prog in random_search(
            _factory, tl, vl,
            param_space=param_space,
            n_trials=int(n_trials_val),
            epochs_per_trial=int(hparam_epochs_val),
            task=task_val,
        ):
            if prog.get("done"):
                lines.append("\n" + format_search_results(prog["all_results"]))
                lines.append(f"\n✅ Best params: {prog['best_params']}  "
                             f"val_acc={prog['best_val_acc']:.2f}%")
            else:
                lines.append(f"Trial {prog['trial']}/{prog['total_trials']}  "
                             f"params={prog['params']}  val_acc={prog['val_acc']:.2f}%")
            yield "\n".join(lines)

    except _cancel.TrainingStopped:
        yield "\n".join([*lines, "⏹ Hyperparameter search stopped."])
    except Exception as exc:
        yield f"❌ Hyperparameter search failed: {exc}"


def _train_sweep_model(run_config):
    """Train one sweep model; return its bundle path or raise RuntimeError(reason).

        The reason is the first ❌ line of the status text the Train tab would
        have shown for the same failure.
        """
    result = None
    try:
        for item in run_stages(run_config):
            if isinstance(item, RunResult):
                result = item
    except PreflightError as exc:
        failure_text = "\n".join(exc.errors)
    except _cancel.TrainingStopped:
        failure_text = _STOPPED_MESSAGE
    except Exception as exc:
        failure_text = _format_run_error(exc)
    else:
        if result is not None and result.bundle_path:
            return result.bundle_path
        failure_text = ""
    reason = next(
        (line.strip().lstrip("❌").replace("**", "").strip()
         for line in failure_text.splitlines() if line.strip().startswith("❌")),
        "the run saved no model bundle",
    )
    raise RuntimeError(reason)


def run_model_sweep(config, sweep_models_val, sweep_metrics_val, sweep_sort_metric_val, sweep_sort_order_val):
    """Train ``config`` once per model in ``sweep_models_val`` and tabulate the metrics."""
    modality_val, task_val, bundle_name_val = config.modality, config.task, config.bundle_name
    import pandas as pd
    from training.run_comparison import load_history

    selected_models = [m for m in (sweep_models_val or []) if m]
    selected_metrics = [m for m in (sweep_metrics_val or []) if m]
    if len(selected_models) < 2:
        yield (
            "### ⚠️ Model sweep\n\nChoose at least two models to compare.",
            pd.DataFrame(),
        )
        return
    if task_val not in {"classification", "regression"}:
        yield (
            "### ⚠️ Model sweep\n\nModel sweep currently supports classification and regression tasks.",
            pd.DataFrame(),
        )
        return
    if not selected_metrics:
        yield (
            "### ⚠️ Model sweep\n\nChoose at least one metric to show in the table.",
            pd.DataFrame(),
        )
        return

    status_lines = ["### Model sweep", "", f"Preparing {len(selected_models)} runs for `{modality_val}` / `{task_val}`."]
    rows = []

    def _sorted_frame(items):
        if not items:
            return pd.DataFrame(columns=["model", *selected_metrics, "status", "bundle"])
        df = pd.DataFrame(items)
        sort_metric = sweep_sort_metric_val if sweep_sort_metric_val in df.columns else None
        if sort_metric:
            df = df.sort_values(
                by=sort_metric,
                ascending=(str(sweep_sort_order_val) == "ascending"),
                na_position="last",
                kind="stable",
            )
        visible_cols = ["model", *selected_metrics, "status", "bundle"]
        return df[[c for c in visible_cols if c in df.columns]]

    _cancel.reset()
    for idx, model_choice in enumerate(selected_models, start=1):
        if _cancel.stop_requested():
            status_lines.append("⏹ Sweep stopped.")
            break
        status_lines.append(f"- Running `{model_choice}` ({idx}/{len(selected_models)})…")
        yield "\n".join(status_lines), _sorted_frame(rows)

        local_bundle = f"{bundle_name_val}_{_slug_part(model_choice)}"
        try:
            bundle_path_val = _train_sweep_model(
                config.replace(model_name=model_choice, bundle_name=local_bundle))
            run_record = next((r for r in reversed(load_history()) if r.get("bundle_path") == bundle_path_val), None)
            metric_payload = (run_record or {}).get("metrics", {})
            row = {
                "model": model_choice,
                "status": "complete",
                "bundle": bundle_path_val,
            }
            for metric_name in selected_metrics:
                row[metric_name] = _metric_display_value(metric_name, metric_payload.get(metric_name))
            rows.append(row)
            status_lines.append(f"  ✓ `{model_choice}` complete")
        except Exception as exc:
            row = {
                "model": model_choice,
                "status": f"failed: {exc}",
                "bundle": "",
            }
            for metric_name in selected_metrics:
                row[metric_name] = None
            rows.append(row)
            status_lines.append(f"  ⚠️ `{model_choice}` failed")

        yield "\n".join(status_lines), _sorted_frame(rows)

    status_lines.extend([
        "",
        f"Sorted by `{sweep_sort_metric_val}` in **{sweep_sort_order_val}** order.",
    ])
    yield "\n".join(status_lines), _sorted_frame(rows)


def _plot_cv_significance_scores(metric_name, baseline_name, baseline_scores, challenger_scores_by_name):
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=130)
    folds = np.arange(1, len(baseline_scores) + 1)
    ax.plot(folds, baseline_scores, marker="o", linewidth=2.2, label=baseline_name, color="#115d52")
    palette = ["#d97706", "#2563eb", "#7c3aed", "#dc2626", "#0f766e"]
    for idx, (name, scores) in enumerate(challenger_scores_by_name.items()):
        ax.plot(
            folds,
            scores,
            marker="o",
            linewidth=1.8,
            linestyle="--",
            label=name,
            color=palette[idx % len(palette)],
            alpha=0.95,
        )
    ax.set_title(f"Fold-by-fold comparison for {metric_name}")
    ax.set_xlabel("Fold")
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_xticks(folds)
    ax.grid(alpha=0.22, linestyle=":")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    plt.close(fig)  # rendered by Gradio after return; frees pyplot's reference
    return fig


def register(c: Components) -> dict:
    """Wire this area's events to the components in ``c``.

    Returns its long-running events by name, for the Stop button.
    """
    cv_event = c.cv_run_btn.click(
        run_cv,
        inputs=[
            c.modality, c.data_path, c.text_col, c.label_col, c.time_col,
            c.window_size, c.n_frames, c.sample_rate, c.augmentation,
            c.val_split, c.use_random_subset, c.subset_percent, c.subset_seed, c.training_mode, c.model_name, c.task,
            c.epochs, c.lr, c.batch_size, c.dropout, c.optimizer, c.scheduler_name, c.use_amp,
            c.timeseries_forecast_mode, c.timeseries_forecast_horizon, c.timeseries_lag_steps, c.timeseries_rolling_window,
            c.hidden_size, c.num_layers, c.cv_k,
        ],
        outputs=c.cv_results_box,
        concurrency_id="heavy_jobs",
        concurrency_limit=1,
    )

    cv_sig_event = c.cv_significance_btn.click(
        run_cv_significance,
        inputs=[
            c.modality, c.data_path, c.label_col, c.time_col, c.window_size, c.use_random_subset, c.subset_percent, c.subset_seed, c.training_mode,
            c.timeseries_forecast_mode, c.timeseries_forecast_horizon, c.timeseries_lag_steps, c.timeseries_rolling_window,
            c.model_name, c.task, c.batch_size, c.dropout,
            c.optimizer, c.hidden_size, c.num_layers,
            c.epochs, c.lr, c.n_estimators, c.max_depth,
            c.C_param, c.max_iter, c.lr_xgb, c.cv_k,
            c.cv_significance_metric, c.cv_significance_models,
        ],
        outputs=[c.cv_significance_box, c.cv_significance_plot],
        concurrency_id="heavy_jobs",
        concurrency_limit=1,
    )

    hparam_event = c.hparam_run_btn.click(
        run_hparam_search,
        inputs=[
            c.modality, c.data_path, c.text_col, c.label_col, c.time_col,
            c.window_size, c.n_frames, c.sample_rate, c.augmentation,
            c.val_split, c.use_random_subset, c.subset_percent, c.subset_seed, c.training_mode, c.model_name, c.task,
            c.batch_size, c.dropout, c.optimizer,
            c.hidden_size, c.num_layers,
            c.hparam_lr_vals, c.hparam_dropout_vals, c.hparam_n_trials, c.hparam_epochs,
        ],
        outputs=c.hparam_results_box,
        concurrency_id="heavy_jobs",
        concurrency_limit=1,
    )

    # The sweep picks the model itself, so it takes every RunConfig input but model_name.
    sweep_config_components = {name: c[name] for name in RUN_CONFIG_FIELDS if name != "model_name"}

    def sweep_from_ui(values):
        yield from run_model_sweep(
            RunConfig.from_component_values(sweep_config_components, values, model_name=""),
            values[c.sweep_models], values[c.sweep_metrics],
            values[c.sweep_sort_metric], values[c.sweep_sort_order],
        )

    sweep_event = c.sweep_run_btn.click(
        sweep_from_ui,
        inputs={*sweep_config_components.values(),
                c.sweep_models, c.sweep_metrics, c.sweep_sort_metric, c.sweep_sort_order},
        outputs=[c.sweep_status_box, c.sweep_table],
        concurrency_id="heavy_jobs",
        concurrency_limit=1,
    )

    return {"cv_event": cv_event, "cv_sig_event": cv_sig_event, "hparam_event": hparam_event, "sweep_event": sweep_event}
