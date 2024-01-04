""" Задание:
  - страница 1 с информацией о разработчике моделей ML:
    @ ФИО, 
    @ номер учебной группы
    @ цветная фотография, 
    @ тема РГР)
"""
# source venv/Scripts/activate
# streamlit run Notebooks/Streamlit_Rgr/Autor_info.py

import streamlit as stm 
  
stm.set_page_config(page_title = "Autor Info") 
stm.title("Информация об авторе")
stm.markdown('''
  ### Имя: Портнягин Алексей
  ### Группа: ФИТ-222
  ### Тема РГР: Разработка дашборда \
  для вывода моделей ML и анализа данных
''')
