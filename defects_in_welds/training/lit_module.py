import torchmetrics
import pytorch_lightning as pl
from torch.optim import Adam

from defects_in_welds.model.model import DefectDetectionModel


# ====================================================================
# PyTorch Lightning Module
# ====================================================================


class DefectLitModule(pl.LightningModule):
    """
    Lightning Module для обучения модели обнаружения объектов.
    """

    def __init__(self, num_classes: int, learning_rate: float = 0.001):
        super().__init__()
        # Автоматическое сохранение гиперпараметров в self.hparams
        self.save_hyperparameters()

        # 1. Модель
        self.model = DefectDetectionModel(num_classes=num_classes)

        # 2. Метрика (Intersection over Union - IoU)
        # BoundingBoxMeanAveragePrecision — стандартная метрика для обнаружения объектов.
        # В TorchMetrics она возвращает словарь метрик.
        self.val_map = torchmetrics.detection.MeanAveragePrecision(box_format="xyxy")

    def forward(self, images, targets=None):
        """Прямой проход (используется для инференса)."""
        return self.model(images, targets)

    # --- ЛОГИКА ОБУЧЕНИЯ ---

    def training_step(self, batch, batch_idx):
        """
        Шаг обучения. Модель возвращает словарь потерь.
        """
        images, targets = batch

        # Модель в режиме обучения возвращает словарь потерь
        loss_dict = self.model(images, targets)

        # Суммируем все потери
        total_loss = sum(loss for loss in loss_dict.values())

        # Логирование
        self.log(
            "train_loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        for key, loss in loss_dict.items():
            self.log(
                f"train_loss/{key}", loss, on_step=False, on_epoch=True, logger=True
            )

        return total_loss

    # --- ЛОГИКА ВАЛИДАЦИИ ---

    def validation_step(self, batch, batch_idx):
        """
        Шаг валидации. Модель возвращает предсказания.
        """
        images, targets = batch

        # В режиме обучения/валидации модель возвращает словарь потерь
        loss_dict = self.model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())
        self.log(
            "val_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # В режиме оценки модель возвращает предсказания
        self.model.eval()
        preds = self.model(images)
        self.model.train()

        # Обновление метрики mAP
        self.val_map.update(preds, targets)

        return {"val_loss": total_loss, "val_map": self.val_map}

    def on_validation_epoch_end(self):
        """
        Вычисление mAP в конце эпохи валидации.
        """
        map_results = self.val_map.compute()

        # Логирование основных метрик (mAP и mAP_50)
        self.log(
            "val_mAP", map_results["map"], on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "val_mAP_50",
            map_results["map_50"],
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        # Очищаем метрику для следующей эпохи
        self.val_map.reset()

    # --- ОПТИМИЗАТОР ---

    def configure_optimizers(self):
        """
        Настройка оптимизатора (Adam).
        """
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)

        # Возвращаем только оптимизатор (без планировщика)
        return optimizer


# ====================================================================
# ТЕСТ
# ====================================================================

if __name__ == "__main__":
    # Тест: 5 классов дефектов + 1 класс фона = 6
    lit_model = DefectLitModule(num_classes=6)
    print("Lightning Module created successfully.")

    # Проверка конфигурации оптимизатора
    optimizer = lit_model.configure_optimizers()
    print(f"Optimizer: {type(optimizer)}")
