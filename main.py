import streamlit as st

# Установка конфигурации страницы
st.set_page_config(
    page_title="Приложение для прогнозирования технического обслуживания",
    page_icon="🔧",
    layout="wide"
)

# Импорт страниц
import analysis_and_model
import presentation

# Настройка навигации по страницам
pages = {
    "Анализ и Модель": analysis_and_model,
    "Презентация Проекта": presentation
}

# Боковая панель для навигации
st.sidebar.title("Навигация")
selection = st.sidebar.radio("Перейти", list(pages.keys()))

# Отображение выбранной страницы
pages[selection].show()