import os
import logging
from typing import Optional, List, Any
import cv2
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
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Получаем высоту и ширину исходного изображения
        H, W, _ = image.shape

        # 2. Загрузка аннотаций YOLO (class_id x_center y_center w h)
        boxes_denormalized = []  # Формат: PASCAL_VOC [xmin, ymin, xmax, ymax] (в ПИКСЕЛЯХ)
        labels = []
        has_boxes = False
        label_path = self.label_files[idx]

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    try:
                        # Парсинг: class_id, x_c, y_c, w, h (нормализованные 0-1)
                        class_id, x_c, y_c, w, h = map(float, line.split())

                        # КЛЮЧЕВОЙ ШАГ: Конвертация YOLO (normalized) -> PASCAL_VOC (denormalized, ПИКСЕЛИ)
                        xmin = (x_c - w / 2) * W
                        ymin = (y_c - h / 2) * H
                        xmax = (x_c + w / 2) * W
                        ymax = (y_c + h / 2) * H

                        # Гарантируем корректность (x_min < x_max и т.д.) перед подачей в Albumentations
                        # и удаляем рамки с нулевой или отрицательной площадью
                        if xmax > xmin and ymax > ymin:
                            boxes_denormalized.append([xmin, ymin, xmax, ymax])
                            labels.append(int(class_id))
                            has_boxes = True

                    except ValueError:
                        continue

        # 3. Применение трансформ Albumentations

        if not has_boxes:
            # Если аннотаций нет, применяем только image-трансформации и возвращаем пустой таргет
            # Это позволяет избежать передачи пустых bboxes в A.Compose,
            # где bboxes_params = PASCAL_VOC
            transformed = self.transforms(image=image)
            image_tensor = transformed["image"]
            target = {
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "labels": torch.empty((0,), dtype=torch.int64),
            }
            return image_tensor, target

        boxes_for_augment = np.array(boxes_denormalized)
        labels_for_augment = np.array(labels)

        # Albumentations принимает PASCAL_VOC в пикселях (если BboxParams настроен правильно)
        transformed = self.transforms(
            image=image, bboxes=boxes_for_augment, class_labels=labels_for_augment
        )

        image = transformed["image"]
        bboxes = transformed["bboxes"]
        labels = transformed["class_labels"]

        # 4. Финальная обработка для Torchvision

        final_boxes = []
        final_labels = []

        for box, label in zip(bboxes, labels):
            # Albumentations уже должен был отфильтровать и обрезать bboxes, но
            # всегда стоит добавить проверку на всякий случай
            xmin, ymin, xmax, ymax = box[:4]

            # Torchvision требует int64 метки
            int_label = int(label)

            # Гарантируем положительную ширину/высоту (требование Torchvision)
            # Albumentations обрезает рамки до границ изображения, но мы
            # еще раз проверяем на минимальность
            if xmax > xmin and ymax > ymin:
                final_boxes.append([xmin, ymin, xmax, ymax])
                final_labels.append(int_label)

        # 5. Формирование целевого словаря для PyTorch/Torchvision
        target = {}

        if not final_boxes:
            target["boxes"] = torch.empty((0, 4), dtype=torch.float32)
            target["labels"] = torch.empty((0,), dtype=torch.int64)
        else:
            # bboxes уже в пикселях, что требуется Torchvision
            target["boxes"] = torch.tensor(final_boxes, dtype=torch.float32)
            target["labels"] = torch.tensor(final_labels, dtype=torch.int64)

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

    # КОРРЕКТИРОВКА: Настройка для работы с ПИКСЕЛЯМИ (PASCAL_VOC)
    def _get_bbox_params(self):
        return A.BboxParams(
            # Используем PASCAL_VOC, который ожидает ПИКСЕЛИ в старых версиях Albumentations
            format="pascal_voc",
            label_fields=["class_labels"],
            # min_area и min_visibility используются для удаления невалидных рамок
            min_area=1.0,
            min_visibility=0.1,
        )

    def setup(self, stage: Optional[str] = None):
        """Создание наборов данных (Dataset) для train/val/test."""
        bbox_params = self._get_bbox_params()

        # --- 1. Определение трансформ ---
        # NOTE: always_apply=True для Resize, Normalize и ToTensorV2
        # является избыточным, но не вредит
        base_transforms = [
            A.Resize(self.image_size[0], self.image_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # ImageNet Norm
            ToTensorV2(),
        ]

        # Для Augmentations используем A.BboxParams
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
                bbox_params=bbox_params,
            )
        else:
            # Если нет аугментаций, A.Compose все равно необходим для применения base_transforms
            train_transforms = A.Compose(
                base_transforms,
                bbox_params=bbox_params,
            )

        val_test_transforms = A.Compose(
            base_transforms,
            bbox_params=bbox_params,
        )

        # --- 2. Создание Dataset ---

        if stage == "fit" or stage is None:
            self.train_ds = WeldingDefectsDataset(
                data_dir=os.path.join(self.data_root, "train"),
                image_size=self.image_size,
                transforms=train_transforms,
            )
            self.val_ds = WeldingDefectsDataset(
                data_dir=os.path.join(self.data_root, "valid"),
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

    # num_workers = 0 для Windows
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=0,
        )


# ====================================================================
# 3. CLI для DVC Pipeline (стадия 'prepare')
# ====================================================================


def prepare(img_size: int = 640, augmentations: bool = True):
    """
    DVC-стадия 'prepare'. Просто проверяет, что данные доступны и
    соответствуют требуемым параметрам.
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
    fire.Fire({"prepare": prepare, "datamodule": DefectDataModule})
