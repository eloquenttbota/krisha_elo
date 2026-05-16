# Прогноз цены квартир — Астана (krisha.kz)

Проект по предсказанию цены за м² квартир в Астане на основе данных с krisha.kz.  
Модель: **Decision Tree Regressor** (sklearn).

---

## Требования

- Python 3.9+
- pip

---

## Установка и запуск

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

### 4. Запустить Jupyter

```bash
jupyter lab
```

Откройте файл `astana_botagoz_git.ipynb` и запустите все ячейки: **Run → Run All Cells**.

---

## Структура проекта

```
krisha_elo/
├── astana_botagoz_git.ipynb         # основной ноутбук
├── krisha_data.csv                  # исходные данные с krisha.kz
├── new_features_of_krisha_data.csv  # данные после feature engineering
├── dt_model.pkl                     # сохранённая модель (после запуска ноутбука)
├── requirements.txt                 # зависимости
├── .gitignore                       # исключения для git
└── README.md                        # этот файл
```

---

## Результат

После запуска ноутбука в папке появится файл `dt_model.pkl` — обученная модель, готовая к использованию.

```python
import pickle

with open('dt_model.pkl', 'rb') as f:
    model = pickle.load(f)

# model.predict(X)
```
