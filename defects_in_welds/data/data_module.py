import os
import logging
from typing import Optional, List, Any

import torch
import numpy as np
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader, Dataset
import fire

# Настройка логирования
log = logging.getLogger(__name__)


# ====================================================================
# 1. CUSTOM DATASET CLASS
# ====================================================================


class WeldingDefectsDataset(Dataset):
    """
    Класс датасета для обнаружения дефектов сварки.
    Предполагает структуру данных YOLO:
    - images/ (изображения)
    - labels/ (соответствующие .txt файлы с аннотациями YOLO)
    """

    def __init__(
        self,
        data_dir: str,
        image_size: List[int],
        transforms: Optional[A.Compose] = None,
    ):
        self.data_dir = data_dir
        self.transforms = transforms
        self.image_files = [
            os.path.join(data_dir, "images", f)
            for f in os.listdir(os.path.join(data_dir, "images"))
            if f.endswith((".jpg", ".png"))
        ]
        self.label_files = [
            os.path.join(
                data_dir, "labels", os.path.splitext(os.path.basename(f))[0] + ".txt"
            )
            for f in self.image_files
        ]
        self.image_size = tuple(image_size)
        log.info(
            f"Initialized Dataset from {data_dir} with {len(self.image_files)} samples."
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, Any]]:
        # 1. Загрузка изображения
        image_path = self.image_files[idx]
        image = np.array(A.augmentations.functional.read_rgb_image(image_path))

        # 2. Загрузка аннотаций YOLO (class_id x_center y_center w h)
        boxes = []
        labels = []
        label_path = self.label_files[idx]

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    try:
                        # Парсинг: class_id, x_c, y_c, w, h (в нормализованных координатах 0-1)
                        class_id, x_c, y_c, w, h = map(float, line.split())
                        boxes.append([x_c, y_c, w, h])  # Нормализованные
                        labels.append(int(class_id))
                    except ValueError:
                        # Пропускаем некорректные строки
                        continue

        if not boxes:
            # Если аннотаций нет, добавляем фиктивные данные для Albumentations
            boxes = np.array([[0, 0, 1, 1]])  # Нормализованные координаты
            labels = np.array([-1])  # Фиктивный класс

        # 3. Применение трансформ Albumentations
        transformed = self.transforms(image=image, bboxes=boxes, class_labels=labels)

        image = transformed["image"]
        # Albumentations возвращает bboxes в PASCAL_VOC (x_min, y_min, x_max, y_max)
        bboxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
        labels = torch.tensor(transformed["class_labels"], dtype=torch.int64)

        # 4. Формирование целевого словаря для PyTorch/Torchvision
        target = {}
        # Если фиктивный класс, делаем его пустым
        if (labels == -1).all():
            target["boxes"] = torch.empty((0, 4), dtype=torch.float32)
            target["labels"] = torch.empty((0,), dtype=torch.int64)
        else:
            target["boxes"] = bboxes
            target["labels"] = labels

        return image, target


# ====================================================================
# 2. PYTORCH LIGHTNING DATA MODULE
# ====================================================================


class DefectDataModule(pl.LightningDataModule):
    """
    Lightning DataModule для управления загрузкой данных.
    """

    def __init__(
        self,
        data_root: str = "data",
        batch_size: int = 16,
        image_size: List[int] = [640, 640],
        augmentations: bool = True,
    ):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.image_size = image_size
        self.augmentations = augmentations

    def setup(self, stage: Optional[str] = None):
        """Создание наборов данных (Dataset) для train/val/test."""

        # --- 1. Определение трансформ (как в configs/main.yaml) ---
        base_transforms = [
            A.Resize(self.image_size[0], self.image_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # ImageNet Norm
            ToTensorV2(),
        ]

        if self.augmentations and stage == "fit":
            train_transforms = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                    ),
                    A.RandomBrightnessContrast(p=0.2),
                    *base_transforms,
                ],
                bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
            )  # Вход YOLO
        else:
            # Валидация/Тестирование без аугментаций
            train_transforms = A.Compose(
                base_transforms,
                bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
            )

        val_test_transforms = A.Compose(
            base_transforms,
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
        )

        # --- 2. Создание Dataset ---

        if stage == "fit" or stage is None:
            self.train_ds = WeldingDefectsDataset(
                data_dir=os.path.join(self.data_root, "train"),
                image_size=self.image_size,
                transforms=train_transforms,
            )
            self.val_ds = WeldingDefectsDataset(
                data_dir=os.path.join(
                    self.data_root, "valid"
                ),  # Используем 'valid' как в DVC
                image_size=self.image_size,
                transforms=val_test_transforms,
            )

        if stage == "test" or stage is None:
            self.test_ds = WeldingDefectsDataset(
                data_dir=os.path.join(self.data_root, "test"),
                image_size=self.image_size,
                transforms=val_test_transforms,
            )

    # Функция, которая объединяет результаты Dataset в один батч
    def collate_fn(self, batch):
        return tuple(zip(*batch))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=os.cpu_count() // 2,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=os.cpu_count() // 2,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=os.cpu_count() // 2,
        )


# ====================================================================
# 3. CLI для DVC Pipeline (стадия 'prepare')
# ====================================================================


def prepare(img_size: int = 640, augmentations: bool = True):
    """
    DVC-стадия 'prepare'. Просто проверяет, что данные доступны и
    соответствуют требуемым параметрам.

    Вызывается DVC: cmd: python defects_in_welds/data/data_module.py prepare
    """
    log.info("--- DVC STAGE: PREPARE ---")
    log.info(f"Image Size Parameter: {img_size}x{img_size}")
    log.info(f"Augmentations Parameter: {augmentations}")

    # Инициализируем DataModule (проверяет наличие папок)
    try:
        dm = DefectDataModule(
            image_size=[img_size, img_size], augmentations=augmentations
        )
        dm.setup(stage="fit")
        log.info(
            f"DataModule initialized successfully. Found {len(dm.train_ds)} training samples."
        )

        # Создаем пустой файл-заглушку для DVC, чтобы зафиксировать успешное выполнение
        with open("data_prepared.txt", "w") as f:
            f.write("Data preparation finished.")
        log.info("Preparation successful. Artifact data_prepared.txt created.")

    except FileNotFoundError as e:
        log.error(f"FATAL: One of the required data directories is missing. {e}")
        raise


if __name__ == "__main__":
    # Fire позволяет легко превратить функции в CLI-интерфейс
    fire.Fire({"prepare": prepare, "datamodule": DefectDataModule})
