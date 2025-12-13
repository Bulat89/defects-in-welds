import hydra
import logging
import os
import json

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger

from defects_in_welds.data.data_module import DefectDataModule
from defects_in_welds.training.lit_module import DefectLitModule

# Настройка логирования
log = logging.getLogger(__name__)


# ====================================================================
# 1. Основная функция обучения, управляемая Hydra
# ====================================================================


@hydra.main(config_path="../../configs", config_name="params", version_base="1.3")
def run_training(cfg):
    """
    Основная функция, управляющая стадией 'train' в DVC Pipeline.

    cfg - словарь конфигурации, загруженный из params.yaml.
    """
    log.info("--- DVC STAGE: TRAIN ---")
    log.info(f"Configuration loaded: {cfg}")

    # 1. Инициализация MLflow Logger
    # MLflowLogger автоматически сохранит все гиперпараметры (cfg)
    mlflow_logger = MLFlowLogger(
        experiment_name="welding_defect_detection",
        run_name=f"run_e{cfg.train.max_epochs}_b{cfg.train.batch_size}",  # ИСПРАВЛЕНО: cfg.train
        tracking_uri="file:./mlruns",  # Локальное хранилище MLflow
    )
    log.info(f"MLflow Logger initialized. Experiment: {mlflow_logger.experiment_id}")

    # 2. Callbacks
    # Сохраняем лучшую модель по метрике val_mAP
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="best_model",
        monitor="val_mAP",
        mode="max",
        save_top_k=1,
        verbose=True,
    )
    # Отслеживание скорости обучения
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # 3. DataModule
    dm = DefectDataModule(
        data_root="data",
        batch_size=cfg.train.batch_size,  # ИСПРАВЛЕНО: cfg.train
        image_size=cfg.prepare.img_size,
        augmentations=cfg.prepare.augmentations,
    )
    dm.setup(stage="fit")

    # 4. LightningModule (Модель)
    # num_classes = число дефектов + 1 (фон)
    # Внимание: cfg.model.num_classes может вызвать следующую ошибку,
    # если num_classes не находится в разделе 'model' в params.yaml.
    num_classes_total = cfg.model.num_classes + 1
    model = DefectLitModule(
        num_classes=num_classes_total,
        learning_rate=cfg.train.learning_rate,  # ИСПРАВЛЕНО: cfg.train
    )

    # 5. Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,  # ИСПРАВЛЕНО: cfg.train
        logger=mlflow_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        accelerator="auto",  # Использует GPU, если доступен
        log_every_n_steps=10,
    )

    # 6. Запуск обучения
    trainer.fit(model, dm)

    # 7. Фиксация истории обучения (для DVC metrics)
    # Извлекаем метрики из MLflow (или просто создаем заглушку)
    history = {
        "max_epochs": cfg.train.max_epochs,  # ИСПРАВЛЕНО: cfg.train
        "final_val_mAP": trainer.callback_metrics.get("val_mAP").item()
        if trainer.callback_metrics.get("val_mAP")
        else None,
        "best_model_path": checkpoint_callback.best_model_path,
    }

    # Записываем историю в файл, который DVC будет отслеживать
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/history.json", "w") as f:
        json.dump(history, f, indent=4)

    log.info(
        f"Training finished. Best model saved to: {checkpoint_callback.best_model_path}"
    )


# ====================================================================
# 2. Запуск через CLI
# ====================================================================

if __name__ == "__main__":
    # Установим базовый путь для Hydra (важно из-за перемещения params.yaml)
    # hydra.main ищет config_path относительно места запуска скрипта
    # Скрипт запускается из корня проекта: python defects_in_welds/training/run_training.py
    # Поэтому config_path = "configs" (если бы params.yaml был в configs/params.yaml)
    # Но мы переместили params.yaml в корень!
    # Hydra.main может не найти его, если мы используем @hydra.main.
    # Поэтому мы используем явный вызов Hydra Compose.

    run_training()
