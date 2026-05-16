# Прогноз цены квартир — Астана (krisha.kz)

Проект по предсказанию цены за м² квартир в Астане на основе данных с krisha.kz.  
Модель: **Decision Tree Regressor** (sklearn). Интерфейс: **Telegram бот**. API: **FastAPI**.

---

## Требования

- Python 3.9+
- pip
- Токен Telegram бота (получить у [@BotFather](https://t.me/BotFather))

---

## Установка

### 1. Клонировать репозиторий

```bash
git clone https://github.com/eloquenttbota/krisha_elo.git
cd krisha_elo
```

### 2. Создать виртуальное окружение

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Установить зависимости

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Настроить переменные окружения

```bash
cp .env.example .env
```

Откройте `.env` и вставьте токен вашего Telegram бота:

```
TELEGRAM_BOT_TOKEN=ваш_токен
API_URL=http://localhost:8000
```

---

## Подготовка модели

Откройте `astana_botagoz_git.ipynb` в Jupyter и запустите все ячейки:

```bash
jupyter lab
```

После запуска в папке появятся `dt_model.pkl` и `feature_names.pkl`.

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

## Структура проекта

```
krisha_elo/
├── backend/
│   ├── main.py                      # FastAPI приложение
│   └── preprocess.py                # Feature engineering
├── bot/
│   └── bot.py                       # Telegram бот
├── astana_botagoz_git.ipynb         # Ноутбук с ML моделью
├── krisha_data.csv                  # Исходные данные с krisha.kz
├── new_features_of_krisha_data.csv  # Данные после feature engineering
├── dt_model.pkl                     # Обученная модель (после запуска ноутбука)
├── feature_names.pkl                # Список признаков модели (после запуска ноутбука)
├── requirements.txt                 # Зависимости
├── .env.example                     # Шаблон переменных окружения
├── .gitignore
└── README.md
```

---

## Как работает бот

1. `/start` — бот задаёт вопросы о квартире шаг за шагом
2. Пользователь вводит: площадь, комнаты, этаж, год постройки, район и т.д.
3. Бот отправляет данные в FastAPI
4. В ответ приходит прогноз: **цена за м²** и **общая стоимость** в тенге
