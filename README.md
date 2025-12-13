Setup (Инструкции по настройке окружения).

Train (Инструкции по запуску обучения).

S3_BUCKET_NAME="dvc-welds-cache-yandex"
AWS_ACCESS_KEY="YCAJEOUSQsEIPVecZpiMZ-BuU"
AWS_SECRET_KEY="YCPQUI16mNnObYE5lW0XaAZ0p9hQJfNvXw-d9Pvo"
YANDEX_ENDPOINT="https://storage.yandexcloud.net"




# =========================================================================
# Шаг 1: ЗАМЕНА ЗНАЧЕНИЙ (Выполняется в вашем терминале)
# =========================================================================

S3_BUCKET_NAME="dvc-welds-cache-yandex"
AWS_ACCESS_KEY="YCAJEOUSQsEIPVecZpiMZ-BuU"
AWS_SECRET_KEY="YCPQUI16mNnObYE5lW0XaAZ0p9hQJfNvXw-d9Pvo"
YANDEX_ENDPOINT="https://storage.yandexcloud.net"

# =========================================================================
# Шаг 2: Настройка DVC Remote
# =========================================================================

# 1. Устанавливаем плагин S3
poetry add dvc-s3

# 2. Настраиваем DVC Remote с именем 'yandex_s3'
poetry run dvc remote add -d yandex_s3 s3://$S3_BUCKET_NAME --local -f

# 3. Указываем кастомный Endpoint Яндекса
poetry run dvc remote modify yandex_s3 endpointurl $YANDEX_ENDPOINT

# 4. Указываем DVC использовать Path Style Addressing
poetry run dvc remote modify yandex_s3 path_style true

# 5. Сохраняем ключи доступа (DVC сохранит их в .dvc/config)
poetry run dvc remote modify yandex_s3 access_key_id $AWS_ACCESS_KEY
poetry run dvc remote modify yandex_s3 secret_access_key $AWS_SECRET_KEY

# =========================================================================
# Шаг 3: Проверка и запуск обучения
# =========================================================================

# 1. Отправляем данные в облако (если они там еще не лежат)
poetry run dvc push

# 2. Скачиваем данные обратно для проверки
poetry run dvc pull

# 3. Запускаем весь пайплайн (prepare, train)
poetry run dvc repro
