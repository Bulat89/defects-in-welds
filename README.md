Поскольку запретов не было описание делал при помощи LLM!!!!!!!!!!!!!!!!!


Обнаружение дефектов сварных швов (Welding Defect Detection)Этот проект реализует конвейер машинного обучения (MLOps Pipeline) для обнаружения дефектов на изображениях сварных швов. В качестве модели обнаружения используется архитектура Faster R-CNN на базе PyTorch/Torchvision.Конвейер MLOps реализован с помощью Data Version Control (DVC) для управления данными и моделями, а конфигурация — с помощью Hydra.Технологический стекФреймворк: PyTorch LightningМодель: Faster R-CNN (с предобученным бэкбоном)Управление данными/моделями: DVC (Data Version Control)Конфигурация: HydraЛогирование экспериментов: MLflowВиртуальное окружение: Conda/venv Структура проектаdefects-in-welds/


├── configs/
│   └── params.yaml           # Конфигурация параметров DVC/Hydra
├── data/                     # Исходные данные (dvc-tracked)
├── defects_in_welds/
│   ├── data/                 # Модули для работы с данными (Dataset, DataModule)
│   ├── model/                # Определение модели (Faster R-CNN)
│   └── training/             # Логика обучения (LightningModule, run_training)
├── DVC.yaml                  # Определение DVC-конвейера (stages: prepare, train)
└── README.md                 # Настоящий файл


Инструкции по запуску (Воспроизведение)Для воспроизведения проекта вам понадобится Python 3.8+ и установленные зависимости.1. Настройка окруженияСоздайте и активируйте виртуальное окружение:Bash# Используя venv
python -m venv venv
source venv/bin/activate

2. Установка зависимостейУстановите все необходимые библиотеки:Bashpip install -r requirements.txt
3. Загрузка данных (DVC Pull)Проект использует DVC для управления набором данных. Чтобы получить данные, выполните:Bashdvc pull
Ожидаемая структура данных: Каталог data/ должен содержать подкаталоги train, valid и test, а внутри каждого — images и labels (аннотации в формате YOLO).4. Запуск полного конвейера DVCКонвейер DVC определен в файле DVC.yaml и включает этапы подготовки данных (prepare) и обучения модели (train).Запустите полный цикл обучения:Bashdvc repro

Эта команда выполнит следующее:prepare: Проверит и подготовит данные.train: Запустит обучение модели PyTorch Lightning, используя параметры из configs/params.yaml.5. Запуск обучения вручную (для отладки/GPU в Colab)Если вы хотите запустить только этап обучения с определенными параметрами (например, если вы используете Colab и установили batch_size=8 и precision='16-mixed'):Bash# Примечание: Hydra автоматически ищет params.yaml
python -m defects_in_welds.training.run_training train.batch_size=8 trainer.precision=16-mixed
Результаты и воспроизводимостьОтслеживание экспериментов (MLflow)Все метрики, гиперпараметры и артефакты (чекпоинты) сохраняются с помощью MLflow.Для просмотра результатов в графическом интерфейсе запустите сервер MLflow:Bashmlflow ui
Примечание для Colab: В Colab вам может потребоваться использовать nohup или Ngrok для доступа к UI.МетрикиКлючевые метрики сохраняются в файле metrics/history.json и в логах MLflow. Основная метрика для оценки — val_mAP (Mean Average Precision) на валидационном наборе.АртефактыОбученная модель с лучшим значением val_mAP сохраняется в:Локально: checkpoints/best_model.ckptВ MLflow: В соответствующем Run'е (каталог artifacts/models). Решение частых проблем (Troubleshooting)ПроблемаПричинаРешениеtorch.OutOfMemoryErrorСлишком большой batch_size или image_size для GPU.Уменьшите batch_size (например, до 4) в params.yaml и используйте смешанную точность (precision='16-mixed').Ошибка размерности тензоров в TorchvisionНеправильный формат тензоров для пустых рамок (boxes).Проблема решена в lit_module.py путем обеспечения размера torch.Size([0, 4]) для пустых боксов.FileNotFoundError (data)DVC не загрузил данные.Убедитесь, что вы выполнили dvc pull и данные находятся в data/.Сгенерировано AI-помощником.
