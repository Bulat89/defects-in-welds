import torch
from torch import nn
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)


# ====================================================================
# PyTorch Model Class (Faster R-CNN)
# ====================================================================


class DefectDetectionModel(nn.Module):
    """
    Класс для создания модели обнаружения объектов (Faster R-CNN)
    на основе архитектуры ResNet-50-FPN-V2.
    """

    def __init__(self, num_classes: int, architecture: str = "fasterrcnn"):
        super().__init__()
        self.num_classes = num_classes
        self.architecture = architecture

        # Загружаем предобученную модель Faster R-CNN (на COCO)
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        model = fasterrcnn_resnet50_fpn_v2(weights=weights)

        # 1. Заменяем классификатор (RoIHeads)
        # Получаем количество входных признаков для классификатора
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # Заменяем боксовый предиктор на новый, с учетом num_classes (включая фон)
        # num_classes = число дефектов + 1 (фон)
        model.roi_heads.box_predictor = model.roi_heads.box_predictor.__class__(
            in_features, num_classes
        )

        self.model = model

    def forward(self, images: list[torch.Tensor], targets: list = None):
        """
        Прямой проход.
        Во время обучения возвращает потери (loss).
        Во время инференса возвращает предсказания.
        """
        # Torchvision detection models ожидают вход: list[Tensor], list[dict]
        # Pytorch Lightning подает вход: list[Tensor], list[dict] (после collate_fn)

        # Если targets передан (режим обучения/валидации)
        if targets is not None:
            # Pytorch Detection Models возвращают словарь потерь
            return self.model(images, targets)

        # Режим инференса (возвращает список предсказаний)
        return self.model(images)


if __name__ == "__main__":
    # Тест: 5 классов дефектов + 1 класс фона = 6
    test_model = DefectDetectionModel(num_classes=6)
    print(f"Model architecture: {test_model.architecture}")
    print(
        f"Number of classes (including background): {test_model.model.roi_heads.box_predictor.cls_score.out_features}"
    )

    # Проверка forward pass (требуется список тензоров для изображений)
    dummy_input = [torch.rand(3, 640, 640) for _ in range(4)]
    dummy_targets = [
        {
            "boxes": torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
        },
        {
            "boxes": torch.empty((0, 4), dtype=torch.float32),
            "labels": torch.empty((0,), dtype=torch.int64),
        },
        {
            "boxes": torch.tensor([[50, 50, 150, 150]], dtype=torch.float32),
            "labels": torch.tensor([3], dtype=torch.int64),
        },
        {
            "boxes": torch.empty((0, 4), dtype=torch.float32),
            "labels": torch.empty((0,), dtype=torch.int64),
        },
    ]

    # Режим обучения (возвращает потери)
    losses = test_model(dummy_input, dummy_targets)
    print(f"Losses (Training Mode): {losses}")

    # Режим инференса (возвращает предсказания)
    predictions = test_model(dummy_input)
    print(f"Predictions (Inference Mode): {len(predictions)} batches")
