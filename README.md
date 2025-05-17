

# Predictive Maintenance Machine Learning Application

## Описание проекта

Приложение для прогнозирования технического обслуживания с использованием машинного обучения. Позволяет анализировать данные о работе оборудования и предсказывать вероятность его отказа.

## Требования к системе

- Python 3.8+
- Visual Studio Code
- Интернет-соединение для загрузки данных

## Подготовка к работе

### 1. Установка Python

1. Скачайте Python с официального сайта: https://www.python.org/
2. При установке обязательно отметьте "Add Python to PATH"
3. Проверьте установку в командной строке:
   ```bash
   python --version
   pip --version
 ### 2. Установка Visual Studio Code

Скачайте VS Code: https://code.visualstudio.com/
Установите расширения:
Python
Pylance

 ### 3. Клонирование репозитория

# Клонируйте репозиторий
git clone [ссылка_на_ваш_репозиторий]
cd predictive_maintenance

4. Создание виртуального окружения
# Создание окружения
python -m venv venv

# Активация окружения
# Для Windows:
venv\Scripts\activate
# Для macOS/Linux:
source venv/bin/activate

5. Установка зависимостей
# Установка библиотек
pip install -r requirements.txt

Запуск приложения
Способ 1: Через терминал VS Code
Откройте терминал в VS Code
Активируйте виртуальное окружение
Запустите приложение:
streamlit run main.py

Способ 2: Через конфигурацию запуска
Создайте файл .vscode/launch.json:
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Streamlit",
            "type": "python",
            "request": "launch",
            "module": "streamlit",
            "args": ["run", "main.py"]
        }
    ]
}

Нажмите F5 или выберите "Run and Debug"
Выберите конфигурацию "Streamlit"

Структура проекта

predictive_maintenance/
│
├── venv/                # Виртуальное окружение
├── main.py              # Основной файл приложения
├── analysis_and_model.py # Страница анализа и моделирования
├── presentation.py       # Страница презентации
├── requirements.txt      # Список зависимостей
└── README.md            # Документация проекта

Функциональность
Загрузка данных о техническом обслуживании
Предобработка и очистка данных
Визуализация статистики
Обучение моделей машинного обучения
Прогнозирование отказов оборудования

Используемые технологии
Python
Streamlit
Scikit-learn
Pandas
Matplotlib
Seaborn

Возможные проблемы и решения

Ошибки при установке

Убедитесь, что Python добавлен в PATH
Обновите pip: pip install --upgrade pip
Используйте python -m pip install ... если возникают проблемы
Ошибки при запуске
Проверьте активацию виртуального окружения
Установите все зависимости из requirements.txt
Проверьте совместимость версий библиотек

Обновление зависимостей

# Обновление списка зависимостей
pip freeze > requirements.txt