# Прогноз цены квартир — Астана (krisha.kz)

Проект по предсказанию цены за м² квартир в Астане на основе данных с krisha.kz.
Ноутбук `astana.ipynb` — EDA, очистка, feature engineering и сравнение моделей
(Decision Tree / Random Forest / XGBoost). Интерфейс: **Telegram-бот**. API: **FastAPI**.

---

## Требования

- Python 3.11+
- pip
- На macOS: [Homebrew](https://brew.sh) (для `libomp`, нужен XGBoost)
- Токен Telegram-бота (получить у [@BotFather](https://t.me/BotFather))

---

## Установка

### 1. Клонировать репозиторий

```bash
git clone https://github.com/eloquenttbota/krisha_elo.git
cd krisha_elo
```

### 2. Настроить окружение

```bash
./setup.sh          # окружение для бота/бэкенда
./setup.sh --dev     # + Jupyter и всё для ноутбука (matplotlib, scipy, jupyterlab...)
```

Скрипт пересоздаёт `venv/` с нуля и на macOS сам ставит `libomp` через Homebrew,
если его не хватает — это системная (не pip) зависимость XGBoost.

> ⚠️ **`venv/` нельзя переносить между компьютерами** (архивом, копированием) —
> внутри абсолютные пути и бинарники, собранные под конкретную машину/архитектуру.
> При переезде на новый компьютер удалите `venv/` и запустите `./setup.sh` заново.

Если предпочитаете вручную:

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
pip install --upgrade pip
pip install -r requirements-dev.txt   # или requirements.txt без ноутбука
```

### 3. Настроить переменные окружения

```bash
cp .env.example .env
```

Откройте `.env` и вставьте токен вашего Telegram-бота:

```
TELEGRAM_BOT_TOKEN=ваш_токен
API_URL=http://localhost:8000
```

---

## Подготовка модели

```bash
source venv/bin/activate
jupyter lab
```

Откройте `astana.ipynb` и выполните все ячейки (Run All). Ноутбук читает
`krysha_astana_160726.csv` напрямую из корня проекта, весь код — в пакете
`src/`. По завершении в корне появятся `model.pkl` и `feature_names.pkl` —
их использует бэкенд.

---

## Запуск

Нужно открыть **два терминала**. В каждом сначала активировать venv:

```bash
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### Терминал 1 — Backend (FastAPI)

Важно: команду нужно запускать **из папки `krisha_elo`**, не из `backend/`.

```bash
# Убедитесь что вы в папке krisha_elo
pwd   # должно показать .../krisha_elo

uvicorn backend.main:app --reload
```

Дождитесь сообщения:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Терминал 2 — Telegram бот

```bash
cd bot
python bot.py
```

Дождитесь сообщения:
```
Бот запущен...
```

Откройте Telegram, найдите вашего бота и отправьте `/start`.

> ⚠️ Backend должен быть запущен **до** того как бот попытается посчитать цену.

---

## Деплой (Render)

Бэкенд и бот запускаются одним процессом через `start.py` (FastAPI + Telegram
бот в одном asyncio-event loop) — это укладывается в один Render Web Service.

1. Зарегистрируйтесь на [render.com](https://render.com) и подключите этот
   GitHub-репозиторий.
2. **New → Blueprint** — Render подхватит `render.yaml` из корня репозитория
   автоматически (build: `pip install -r requirements.txt`, start: `python start.py`).
3. В настройках сервиса задайте переменную окружения `TELEGRAM_BOT_TOKEN`
   (сама переменная не хранится в репозитории — `render.yaml` только просит
   Render запросить её значение).
4. Деплой запустится сам; Render передаёт порт через `$PORT`, `start.py` его
   уже читает.

Если предпочитаете настраивать вручную (без Blueprint): **New → Web Service**,
Build Command `pip install -r requirements.txt`, Start Command `python start.py`,
Python 3.11.

---

## Структура проекта

```
krisha_elo/
├── src/                              # Пайплайн: EDA, очистка, feature engineering, модели
│   ├── setup_env.py
│   ├── data_loading.py
│   ├── eda.py
│   ├── cleaning.py
│   ├── feature_engineering.py
│   ├── feature_selection.py
│   ├── modeling.py
│   ├── viz_utils.py
│   └── geo.py
├── backend/
│   ├── main.py                      # FastAPI приложение
│   └── preprocess.py                # Feature engineering для одного объявления (зеркалит src/)
├── bot/
│   └── bot.py                       # Telegram бот
├── astana.ipynb                     # Ноутбук-презентация: EDA → модель → продакшн
├── krysha_astana_160726.csv         # Исходные данные с krisha.kz
├── krisha.kz.png                    # Логотип для ноутбука
├── model.pkl                        # Обученная модель (после запуска ноутбука)
├── feature_names.pkl                # Список признаков модели (после запуска ноутбука)
├── setup.sh                         # Скрипт настройки окружения
├── start.py                         # Точка входа для деплоя (backend + бот в одном процессе)
├── render.yaml                      # Конфигурация деплоя на Render
├── requirements.txt                 # Зависимости для бота/бэкенда
├── requirements-dev.txt             # + зависимости для ноутбука
├── .env.example                     # Шаблон переменных окружения
├── .gitignore
└── README.md
```

---

## Как работает бот

1. `/start` — бот задаёт вопросы о квартире шаг за шагом
2. Пользователь вводит: площадь, комнаты, этаж, этажность, год постройки,
   высоту потолков, район, тип дома, продавца и наличие ЖК
3. Бот отправляет данные в FastAPI
4. В ответ приходит прогноз: **цена за м²** и **общая стоимость** в тенге
