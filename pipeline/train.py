"""Stage 3: train the model, reporting each epoch."""
from __future__ import annotations

import logging
import os
import time
from typing import Iterator

from config import DEFAULTS
from pipeline.context import RunContext
from pipeline.progress import LIVE_PLOT_UPDATE_EVERY, Update, fmt_eta

logger = logging.getLogger("nocode_dl")


def train(ctx: RunContext) -> Iterator[Update]:
    cfg, log = ctx.cfg, ctx.log
    extra_data = ctx.extra_data
    train_loader, val_loader = ctx.train_loader, ctx.val_loader
    num_classes = ctx.num_classes
    model = ctx.model
    sklearn_model, graph_node2vec = ctx.sklearn_model, ctx.graph_node2vec
    runtime_device = ctx.runtime_device
    safe_bundle_name = ctx.bundle_name
    final_model = None
    history: list = []
    stopped_early_flag = False

    log.set_phase("training", eta_text="ETA: estimating…")
    ckpt_dir = None
    if int(cfg.checkpoint_every) > 0:
        ckpt_dir = os.path.join("outputs", "checkpoints", safe_bundle_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        log.log(f"  Checkpoints → {ckpt_dir}")
        yield Update()

    if cfg.model_name == "Autoencoder":
        import torch, torch.nn as nn
        model = model.to(runtime_device)
        ae_optim      = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))
        ae_criterion  = nn.MSELoss()
        ae_history    = []
        ae_epoch_times = []
        log.log("Training — autoencoder optimisation started")
        yield Update()
        for epoch in range(1, int(cfg.epochs) + 1):
            epoch_t0 = time.time()
            # ── Train pass ──────────────────────────────────────
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                inputs, _ = batch
                inputs    = inputs.to(runtime_device)
                ae_optim.zero_grad()
                recon = model(inputs)
                loss  = ae_criterion(recon, inputs)
                loss.backward()
                ae_optim.step()
                train_loss += loss.item()
            train_loss /= max(len(train_loader), 1)

            # ── Validation pass ─────────────────────────────────
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    inputs, _ = batch
                    inputs    = inputs.to(runtime_device)
                    recon     = model(inputs)
                    val_loss += ae_criterion(recon, inputs).item()
            val_loss /= max(len(val_loader), 1)

            rec = {"epoch": epoch, "train_loss": round(train_loss, 4),
                   "val_loss": round(val_loss, 4), "val_acc": 0.0,
                   "eta_seconds": 0, "lr": float(cfg.lr)}
            epoch_elapsed = time.time() - epoch_t0
            ae_epoch_times.append(epoch_elapsed)
            avg_epoch_time = sum(ae_epoch_times) / len(ae_epoch_times)
            eta_seconds = avg_epoch_time * (int(cfg.epochs) - epoch)
            rec["eta_seconds"] = round(eta_seconds, 1)
            ae_history.append(rec)
            pct = 0.15 + 0.65 * epoch / int(cfg.epochs)
            ctx.progress(pct, desc=f"Autoencoder epoch {epoch}/{int(cfg.epochs)}")
            eta_str = fmt_eta(eta_seconds) if epoch < int(cfg.epochs) else "ETA: completing training"
            log.set_eta(eta_str)
            log.update_epoch_metrics(
                epoch_text=f"{epoch}/{int(cfg.epochs)}",
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{val_loss:.4f}",
                metric="reconstruction",
                lr_value=f"{float(cfg.lr):.6f}",
            )
            line = (f"Epoch {epoch}/{int(cfg.epochs)} — "
                    f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  {eta_str}")
            log.log(line)
            live_plot = list(ae_history) if (
                epoch == int(cfg.epochs) or epoch % LIVE_PLOT_UPDATE_EVERY == 0
            ) else None
            yield Update(loss_history=live_plot, eta=eta_str)
        final_model = model
        history     = ae_history

    elif graph_node2vec:
        from training.graph_trainer import train_node2vec_classifier

        log.set_phase("training", eta_text="ETA: learning graph embeddings")
        log.update_epoch_metrics(
            epoch_text="1/1",
            train_loss="—",
            val_loss="—",
            metric="embedding baseline",
            lr_value="—",
        )
        log.log("Training — learning Node2Vec graph embeddings")
        yield Update()
        node2vec_result = train_node2vec_classifier(
            extra_data,
            embedding_dim=int(cfg.hidden_size),
            max_iter=int(cfg.max_iter),
        )
        final_model = node2vec_result["model"]
        history = node2vec_result["history"]
        extra_data["graph_embeddings"] = node2vec_result["embeddings"]
        extra_data["graph_eval"] = {
            "y_true": node2vec_result["y_true"],
            "y_pred": node2vec_result["y_pred"],
            "y_prob": node2vec_result["y_prob"],
        }
        log.set_phase("training", eta_text="ETA: finalizing results")
        log.update_epoch_metrics(
            epoch_text="1/1",
            train_loss="—",
            val_loss="0.0000",
            metric=f"acc={history[-1].get('val_acc', 0)}",
            lr_value="—",
        )
        log.log("Training — Node2Vec baseline fit complete")
        yield Update()

    elif sklearn_model:
        from training.trainer import train_sklearn
        X_train = extra_data["X_train"]; y_train = extra_data["y_train"]
        X_val   = extra_data.get("X_val"); y_val   = extra_data.get("y_val")
        log.set_phase("training", eta_text="ETA: fitting baseline")
        log.update_epoch_metrics(
            epoch_text="Baseline",
            train_loss="—",
            val_loss="—",
            metric="fitting",
            lr_value="—",
        )
        log.log("Training — fitting baseline model")
        yield Update()
        for prog in train_sklearn(model, X_train, y_train, X_val, y_val):
            if prog["done"]:
                final_model = prog["model"]
                history     = prog["history"]
                log.set_phase("training", eta_text="ETA: finalizing results")
                if history:
                    last = history[-1]
                    log.update_epoch_metrics(
                        epoch_text=str(last.get("epoch", "1/1")),
                        train_loss=last.get("train_loss"),
                        val_loss=last.get("val_loss"),
                        metric=last.get("val_acc"),
                        lr_value=last.get("lr"),
                    )
                log.log("Training — baseline fit complete, finalizing metrics")
                yield Update()
            else:
                log.set_phase("training", eta_text="ETA: fitting baseline")
                log.log(prog.get("status", "Fitting model…"))
                yield Update()

    elif cfg.modality == "graph":
        from training.graph_trainer import train_graph_pytorch
        cw = None
        if cfg.use_class_weights and cfg.task == "classification":
            try:
                train_labels = extra_data["y"][extra_data["train_idx"]].cpu().tolist()
                from training.trainer import compute_class_weights
                cw = compute_class_weights(train_labels, num_classes=num_classes)
                log.log("  ✓ Class weights applied (imbalance correction)")
                yield Update()
            except Exception as e:
                log.log(f"  ⚠️ Class weights skipped: {e}")
                yield Update()

        _running_history: list[dict] = []
        log.log("Training — full-graph node classification started")
        yield Update()
        for prog in train_graph_pytorch(
            model,
            extra_data,
            epochs=int(cfg.epochs),
            lr=float(cfg.lr),
            optimizer_name=cfg.optimizer,
            patience=DEFAULTS["early_stopping_patience"],
            class_weights=cw,
        ):
            if prog["done"]:
                final_model = prog["model"]
                history = prog["history"]
            else:
                pct = 0.15 + 0.65 * prog["epoch"] / prog["epochs"]
                ctx.progress(pct, desc=f"Epoch {prog['epoch']}/{prog['epochs']}")
                eta_str = fmt_eta(prog.get("eta_seconds", 0))
                log.set_eta(eta_str)
                log.update_epoch_metrics(
                    epoch_text=f"{prog['epoch']}/{prog['epochs']}",
                    train_loss=prog["train_loss"],
                    val_loss=prog["val_loss"],
                    metric=f"acc={prog['val_acc']}",
                    lr_value=prog.get("lr", ""),
                )
                line = (
                    f"Epoch {prog['epoch']}/{prog['epochs']}  "
                    f"train={prog['train_loss']}  val={prog['val_loss']}  "
                    f"acc={prog['val_acc']}  lr={prog.get('lr','')}  {eta_str}"
                    + ("  ⏹ early stop" if prog["stopped_early"] else "")
                )
                log.log(line)
                _running_history.append({k: prog[k] for k in ("epoch", "train_loss", "val_loss", "val_acc")})
                live_plot = list(_running_history) if (
                    prog["epoch"] == prog["epochs"]
                    or prog["stopped_early"]
                    or prog["epoch"] % LIVE_PLOT_UPDATE_EVERY == 0
                ) else None
                yield Update(loss_history=live_plot, eta=eta_str if not prog["stopped_early"] else "ETA: completing training")

    else:
        from training.trainer import train_pytorch, compute_class_weights
        # Optional class weighting for imbalanced datasets
        cw = None
        if cfg.use_class_weights and cfg.task in ("classification",):
            try:
                # Fast path: pull labels directly from the dataset without
                # iterating the full DataLoader (avoids an extra O(n) pass).
                ds = train_loader.dataset
                if hasattr(ds, "tensors"):
                    # TensorDataset — labels are the second tensor
                    all_labels = ds.tensors[1].tolist()
                elif hasattr(ds, "samples"):
                    # ImageDataset / AudioDataset style
                    all_labels = [s[1] for s in ds.samples]
                else:
                    # Fallback: iterate (safe but slower)
                    all_labels = [int(lbl) for _, lbls in train_loader
                                  for lbl in lbls.tolist()]
                cw = compute_class_weights(all_labels, num_classes=num_classes)
                log.log("  ✓ Class weights applied (imbalance correction)")
                yield Update()
            except Exception as e:
                log.log(f"  ⚠️ Class weights skipped: {e}")
                yield Update()

        # Accumulate epoch records so the live loss plot shows the full curve,
        # not just the most-recent single epoch.
        _running_history: list[dict] = []
        log.log("Training — optimiser warmup complete, starting epochs")
        yield Update()

        for prog in train_pytorch(
            model, train_loader, val_loader,
            epochs=int(cfg.epochs), lr=float(cfg.lr),
            optimizer_name=cfg.optimizer,
            patience=DEFAULTS["early_stopping_patience"],
            scheduler_name=cfg.scheduler_name,
            use_amp=bool(cfg.use_amp),
            task=cfg.task,
            class_weights=cw,
            checkpoint_dir=ckpt_dir,
            checkpoint_every=int(cfg.checkpoint_every),
            device=runtime_device,
        ):
            if prog["done"]:
                final_model = prog["model"]
                history     = prog["history"]
            else:
                stopped_early_flag = prog.get("stopped_early", False)
                pct = 0.15 + 0.65 * prog["epoch"] / prog["epochs"]
                ctx.progress(pct, desc=f"Epoch {prog['epoch']}/{prog['epochs']}")
                eta_str = fmt_eta(prog.get("eta_seconds", 0))
                log.set_eta(eta_str)
                metric_label = "acc" if cfg.task == "classification" else (
                    "exact-match" if cfg.task == "multi-label" else "MAE")
                log.update_epoch_metrics(
                    epoch_text=f"{prog['epoch']}/{prog['epochs']}",
                    train_loss=prog["train_loss"],
                    val_loss=prog["val_loss"],
                    metric=f"{metric_label}={prog['val_acc']}",
                    lr_value=prog.get("lr", ""),
                )
                line = (
                    f"Epoch {prog['epoch']}/{prog['epochs']}  "
                    f"train={prog['train_loss']}  val={prog['val_loss']}  "
                    f"{metric_label}={prog['val_acc']}  "
                    f"lr={prog.get('lr','')}  {eta_str}"
                    + ("  ⏹ early stop" if prog["stopped_early"] else "")
                )
                log.log(line)
                # Append to running history so the plot shows the full curve
                _running_history.append({k: prog[k] for k in
                                         ("epoch", "train_loss", "val_loss", "val_acc")})
                live_plot = list(_running_history) if (
                    prog["epoch"] == prog["epochs"]
                    or prog["stopped_early"]
                    or prog["epoch"] % LIVE_PLOT_UPDATE_EVERY == 0
                ) else None
                yield Update(loss_history=live_plot,
                             eta=eta_str if not prog["stopped_early"] else "ETA: completing training")

    log.set_phase("complete", eta_text="ETA: Done")
    if stopped_early_flag:
        log.log("Training — early stopping triggered, best checkpoint restored")
        yield Update()
    log.log("Training complete ✓")
    loss_history_final = history if history else None
    yield Update(loss_history=loss_history_final, eta="ETA: Done")

    # ── 3b. Plain-English training summary ───────────────────────────────
    training_summary_text = ""
    if history and not sklearn_model and cfg.model_name != "Autoencoder":
        try:
            from ui.training_summary import summarise_run
            training_summary_text = summarise_run(
                history, cfg.task, cfg.model_name, cfg.modality,
                stopped_early=stopped_early_flag)
            yield Update(loss_history=loss_history_final, eta="ETA: Done",
                         train_summary=training_summary_text)
        except Exception as e:
            logger.warning("Training summary generation failed: %s", e)

    ctx.final_model = final_model
    ctx.stopped_early = stopped_early_flag
    ctx.result.history = history
    ctx.result.training_summary = training_summary_text
