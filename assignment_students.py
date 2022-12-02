#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 12:17:43 2022

@author: Ernesto Evgeniy Sanches Shayda
"""

# Для работы следует установить (используя pip или conda);
# - Python 3.9
# - Текущая версия scipy 1.9.3 для математических алгоритмов
# - pandas + openpyxl + xlsxwriter для работы с Excel файлами
# - json для загрузки текстовых данных

import json
import numpy as np
import pandas as pd
from scipy.optimize import linprog

def describe_preferences(preferences, student_names, department_names):
    ''' Вывод информации о таблице предпочтений '''
    number_of_students, number_of_departments = preferences.shape
    print("Проверка сгенерированных данных:")
    print("Количество предпочтений у кафедр (должно соответствовать таблице):")
    for department_id in range(number_of_departments):
        p1 = np.count_nonzero(preferences[:, department_id] == 1)
        p2 = np.count_nonzero(preferences[:, department_id] == 2)
        print(f"{department_names[department_id]} - {p1} / {p2}")
    '''
    print("Количество предпочтений у студентов (должно быть не больше 1):")
    for student_id in range(number_of_students):
        p1 = np.count_nonzero(preferences[student_id] == 1)
        p2 = np.count_nonzero(preferences[student_id] == 2)
        print(f"{student_names[student_id]} - {p1} / {p2}")
    '''

def load_data_from_files(table_path, limits_path):
    ''' Код для считывания тестовых данных из файлов '''
    data = np.array(pd.read_excel(
        table_path, header=None, usecols="A:N",
        sheet_name="Текущая").drop(columns=[10,]).fillna(0), dtype=int)
    student_ids = data[:, 0]
    preferences = data[:, 1:]

    n, m = preferences.shape
    limits = np.loadtxt(limits_path, delimiter=' ')
    student_names = [f"Студент {x}" for x in student_ids]
    # Названия кафедр - по столбцам матрицы предпочтений
    department_names = [f"Кафедра {x + 1}" for x in range(m)]
    SHOW_DEBUG = True # выводить дополнительную информацию
    if SHOW_DEBUG:
        describe_preferences(preferences, student_names, department_names)
    return preferences, limits, student_names, department_names

def generate_real_data(number_of_students=48, max_students_in_department=4):
    '''
        Код для генерации тестовых данных по присланной таблице предметов.
        Сам код, решающий задачу, находится ниже в __main__.
    '''
    # задано количество студентов с предпочтением 1-го и 2-го приоритета на
    # каждый предмет. По этим данным программа генерирует индивидуальных
    # студентов с заданым распределением
    # Оставщиеся кафедры и студенты заполняются нулевыми/безразличными
    # предпочтениями
    '''
    departments = {
        "1. Экосистема ИНГО" : [5, 4],
        "2. Оф - Он" : [1, 6],
        "3. Кросс-прод" : [0, 2],
        "4. Self-made" : [8, 5],
        "5. NPS" : [3, 0],
        "6. Лекция" : [2, 5],
        "7. Будь здоров" : [1, 3],
        "8. Лизинг п. и т." : [2, 1],
        "9. Лизинг цифр. и плат." : [5, 2],
        "10. Инвест для Z" : [13, 10],
        "11. Стратегия СВО" : [3, 3],
        "12. Банк СОЮЗ" : [3, 4]
        }
    '''
    # Обновленный до 48 студентов список
    departments = {
        "1. Экосистема ИНГО" : [5, 4],
        "2. Оф - Он" : [1, 7],
        "3. Кросс-прод" : [0, 2],
        "4. Self-made" : [8, 5],
        "5. NPS" : [3, 0],
        "6. Лекция" : [2, 5],
        "7. Будь здоров" : [1, 4],
        "8. Лизинг п. и т." : [3, 1],
        "9. Лизинг цифр. и плат." : [5, 2],
        "10. Инвест для Z" : [14, 11],
        "11. Стратегия СВО" : [3, 3],
        "12. Банк СОЮЗ" : [3, 4]
        }

    # максимальное количество студентов на кафедрах
    number_of_departments = len(departments)
    limits = np.full((number_of_departments,), max_students_in_department)
    # имена студентов - по строкам матрицы предпочтений
    student_names = [f"Студент {x + 1}" for x in range(number_of_students)]
    # названия кафедр - по столбцам матрицы предпочтений
    department_names = list(departments.keys())

    # генерация предпочтений студентов
    # Матрица предпочтений (nxm): по строкам студенты (n)
    #                             по столбцам кафедры (m)
    # Предпочтения в каждой строке задаются числами 1, 2, ...
    # Значение в матрице - приоритет предпочтения студента - кафедре
    preferences = np.zeros((number_of_students, number_of_departments))

    # По столбцам заполняем случайно предпочтения студентов
    # так, чтобы соблюдались ограничения из опроса и не было
    # повторяющихся предпочтений
    while(True):
        is_updated = False
        department_ids = np.arange(number_of_departments)
        for department_id in department_ids:
            department = department_names[department_id]
            first_priority, second_priority = departments[department]
            for priority, remaining in [(1, first_priority), (2, second_priority)]:
                zero_preferences_ids = np.where(np.logical_and(
                    preferences[:, department_id] == 0,
                    np.sum(preferences == priority, axis=1) == 0
                    ))[0]
                if len(zero_preferences_ids) > 0 and remaining > 0:
                    sample_count = min(len(zero_preferences_ids), remaining)
                    zero_preferences_ids_sample = np.random.choice(
                        zero_preferences_ids, sample_count, replace=False)
                    preferences[zero_preferences_ids_sample, department_id] = priority
                    updated_count = len(zero_preferences_ids_sample)
                    departments[department][priority - 1] -= updated_count
                    is_updated = True
        if not is_updated:
            break

    SHOW_DEBUG = True # выводить дополнительную информацию
    if SHOW_DEBUG:
        describe_preferences(preferences, student_names, department_names)
    return preferences, limits, student_names, department_names


def generate_simple_data():
    # небольшие примеры для тестирования
    '''
    # Простой пример: 5х3
    preferences = np.array(
        [[1,2,3],preferences
         [1,2,3],
         [1,2,3],
         [2,1,3],
         [3,1,2]])
    '''
    # Больший пример: 10х5;
    # Смысл значений 1..5, 6, 7 - см. в описании словаря values ниже
    preferences = np.array(
        [
         [1,2,3,4,5],
         [1,2,3,4,5],
         [1,2,3,4,5],
         [4,5,2,1,3],
         [4,3,1,2,5],
         [1,4,2,5,3],
         [1,2,4,3,5],
         [2,3,4,5,1],
         [5,1,4,3,2],
         [4,3,5,2,1],
         ])
    n, m = preferences.shape
    # имена студентов - по строкам матрицы предпочтений
    student_names = [f"Student {x + 1}" for x in range(n)]
    # Названия кафедр - по столбцам матрицы предпочтений
    department_names = [f"Department {x + 1}" for x in range(m)]
    # массив размером m, максимальное количество мест.
    limits = np.full((m,), 3) # [3, 3, 3, 3, 3] - m раз число 3
    #limits = [2, 2, 2] # пример задания вручную, для задачи 5 студ. х 3 каф.
    return preferences, limits, student_names, department_names



if __name__ == "__main__":
    # Раскомментируйте np.random.seed если хотите получать неслучайные
    # результаты для тестирования алгоритма
    # np.random.seed(42)

    # количество повторений эксперимента
    N_EXPERIMENT_REPEATS = 3

    # добавлять ли колонки с результатами
    # это необязательно, так как результат виден из окрашивания
    ADD_RESULT_COLUMNS = True

    # значения ценности каждого предпочтения
    # Ненулевые значения - значимые предпочтения;
    # Нулевые значения - кафедра не интересует, не важен приоритет между
    #                    нулевыми значениями
    # Например: {1:0, 2:3, 3:6, 4:0, 5:0} - при назначении студента
    #   на самую предпочитаемую кафедру (приоритет 1) - получаем 0 баллов,
    #   на вторую по порядку предпочтения (приоритет 2) - 3 баллов,
    #   на третью - 6 балла,
    #   при назначении же на две оставшиеся кафедры - 0 баллов
    # Примечание: если мы хотим добавить "специальные предпочтения",
    # а именно - кто-то не хочет совсем на определенную кафедру,
    # или обязательно хочет точно попасть на некоторую кафедру,
    # можно добавить специальные предпочтения с большим (например, 100)
    # или сильно отрицательным (например, -100) значением, и указывать
    # индексы этих предпочтений в таблице preferences (6, 7 в примере выше)

    '''
    values = {1: 0,  # хорошие варианты ++
              2: 3,  # хорошие варианты +
              3: 10, # средний вариант  -
              4: 20, # плохие варианты  --
              5: 20, # плохие варианты  --
              6: 100,# самый неподходящий вариант
              7: -100, # хорошее значение чтобы точно выбрать кафедру
              0: 20 # по умолчанию, кафедры без приоритета
              }
    '''
    # загрузка приоритетов из файла
    priorities_name = "priorities.txt"
    # object_hook конвертирует строковые ключи из файла в числа
    values = json.load(open(priorities_name, "r"), object_hook=lambda d:
                       {int(k) if k.lstrip('-').isdigit() else k: v for k, v in d.items()})

    # Для всех вариантов ограничения мест на кафедрах
    projects_name = "проекты"
    # "limits_all_4",
    for limits_name in ["limits_some_7"]:
        # сохраняемые результаты для организации по листам в Excel
        all_results = []
        all_experiments_departments = []
        all_experiments_priorities = []
        print(f"\nДанные для эксперимента: {projects_name}.xlsx, {limits_name}.txt\n")
        # Генерация основных данных - чтение из файла
        preferences, limits, student_names, department_names = (
            load_data_from_files(f"{projects_name}.xlsx",
                                 f"{limits_name}.txt"))
        # Предыдущий пример - генерация случайных распределений по заданной таблице
        #preferences, limits, student_names, department_names = (
        #    generate_real_data())

        # Повторяем эксперимент много раз. Будут найдены разные решения,
        # каждое из которых одинаково оптимально
        for experiment_idx in range(N_EXPERIMENT_REPEATS):
            # конечный результат данной итерации цикла экспериментов
            result_df = pd.DataFrame(preferences, index=student_names,
                                     columns=department_names)
            # Важно: так как может существовать много оптимальных назначений,
            # алгоритм выдаст одно из них, возможно, первое найденное.
            # Чтобы гарантировать случайность назначения студентов в рамках
            # выбранных предпочтений, следует случайно перемешать их заранее.
            # Можно на примерах проверить справедливость назначений,
            # предлагаемых алгоритмом
            n, m = preferences.shape
            idx_shuffle = np.arange(n)
            np.random.shuffle(idx_shuffle)
            preferences = preferences[idx_shuffle]
            student_names = [student_names[i] for i in idx_shuffle]

            # Решаем вариант общей задачи о назначениях, когда есть ограничение мест
            # на кафедрах (на назначаемых объектах)

            # В библиотеке scipy есть функция для решения простой задачи о назначениях,
            # когда ограничение мест всегда равно 1. Этот вариант подходит в случае,
            # напрммер, назначения работников на задачи, когда каждый работник
            # выполняет 1 задачу и каждая задача выполняется 1 работником
            # Эту задачу решает функция scipy.optimize.linear_sum_assignment

            # Однако у нас - усложненный вариант, являющийся частным случаем общей
            # задачи о назначениях. Есть библиотеки на github, решающие такую задачу,
            # но найденные мной варианты не являются официально признанными.
            # Поэтому мы сформулируем наш вариант задачи в виде общей задачи
            # "линейного проргаммирования" (общая задача о назначениях является частным
            # случаем линейного программирования) и решим ее стандартным способом
            # из библиотеки scipy: scipy.optimize.linprog.

            # Определим матрицы и векторы, используемые в задаче линейного
            # программирования (см. документацию scipy.optimize.linprog),
            # преобразовав данные из нашей изначальной задачи.
            # Минус в "c" нужен для преобразования задачи максимизации в минимизацию
            c = np.array([values[x] for x in preferences.flatten()])
            A_ub = np.zeros((m, n * m))
            b_ub = np.zeros(m)
            A_eq = np.zeros((n, n * m))
            b_eq = np.zeros(n)
            integrality = np.ones(len(c)) # все назначения целочисленные (0/1)
            bounds = (0, 1) # 0 - не назначен, 1 - назначен
            # матрица и вектор ограничений задают ограничения на 1 кафедру на студента
            # и limits студентов на кафедры. Заполняем ограничения используя циклы
            # по строкам и столбцам матрицы preferences
            for i in range(n):
                preferences_constraint = np.zeros((n, m))
                preferences_constraint[i] = 1 # сумма назначений по строке i --->
                A_eq[i, :] = preferences_constraint.flatten()
                b_eq[i] = 1 # ---> равна 1
            for j in range(m):
                preferences_constraint = np.zeros((n, m))
                preferences_constraint[:, j] = 1 # сумма назначений по столбцу j --->
                A_ub[j, :] = preferences_constraint.flatten()
                b_ub[j] = limits[j] # ---> меньше либо равна значению из limits

            # Запускаем алгоритм решения задачи линейного программирования!
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                             bounds=bounds, integrality=integrality)

            final_assignment = np.asarray(result.x.reshape((n, m)), dtype=int)
            print(f"Оптимизация завершена: {result.message}")
            # Возвращаем порядок студентов в исходный после перемешивания
            inverted_shuffle = np.empty_like(idx_shuffle)
            inverted_shuffle[idx_shuffle] = np.arange(len(idx_shuffle))
            # Ответ готов
            SHOW_DEBUG = True # выводить дополнительную информацию
            if SHOW_DEBUG:
                print(f"\nПредпочтения (приоритет):\n{preferences[inverted_shuffle]}")
                print(f"Баллы (численные предпочтения):\n{c.reshape(preferences.shape)[inverted_shuffle]}")
                print(f"Студенты по строкам: {[student_names[i] for i in inverted_shuffle]}")
                print(f"Кафедры по столбцам: {department_names}")
                print(f"Результат (таблица назначений):\n{final_assignment[inverted_shuffle]}")
                print(f"Количество баллов в соответствии с предпочтениями: {result.fun}")

                print("\nПроверка решения на соответствие ограничениям:")

                print("1. Каждый студент назначен ровно на одну кафедру")
                print(f"Количество назначений каждого студента: \n{(A_eq @ result.x)[inverted_shuffle]}")
                print(f"Ожидаемое количество назначений студентов (точно): \n{b_eq}")

                print("2. Каждая кафедра приняла не более заданного числа студентов")
                print(f"Количество назначений на каждую кафедру: \n{A_ub @ result.x}")
                print(f"Ожидаемое количество назначений на кафедры (не более): \n{b_ub}")

            print("\nНазначения:\n")

            current_results_assignments = []
            current_result_departments = []
            current_results_priority = []
            current_results_good_or_bad = []

            fst_prior_num, snd_prior_num, trd_prior_num, low_prior_num = [0,0,0,0]
            for student_idx in inverted_shuffle:
                department_idx = np.argmax(final_assignment[student_idx])
                priority = preferences[student_idx, department_idx]
                priority_to_sign = {1 : "++",
                                    2 : " +",
                                    3 : " -", 
                                    4 : " --",
                                    0 : " 0"}
                if priority == 1: fst_prior_num +=1
                elif priority == 2: snd_prior_num +=1
                elif priority == 3: trd_prior_num +=1
                else: low_prior_num +=1                    
                good_or_bad = (priority_to_sign[priority]
                               if priority in priority_to_sign else "--")

                current_results_assignments.append(department_names[department_idx])
                current_result_departments.append(department_idx)
                current_results_priority.append(priority)
                current_results_good_or_bad.append(good_or_bad)

                print(f"[{good_or_bad}] {student_names[student_idx]} --> {department_names[department_idx]} -")

            # сохранение для использования в окрашивании
            all_experiments_departments.append(current_result_departments)
            all_experiments_priorities.append(current_results_priority)

            if ADD_RESULT_COLUMNS:
                # добавление текущего результата виде дополнительных колонок
                result_df[f"Назн. #{experiment_idx}"] = current_results_assignments
                result_df[f"Приор. #{experiment_idx}"] = current_results_priority
                result_df[f"Кач. #{experiment_idx}"] = current_results_good_or_bad
            # отмена перемешивания, чтобы с начала начать следующий повтор
            preferences = preferences[inverted_shuffle]
            student_names = [student_names[i] for i in inverted_shuffle]

            # сохранение основных результатов
            all_results.append(result_df)

            print("Количество назначений по первому приоритету:", fst_prior_num)
            print("Количество назначений по второму приоритету:", snd_prior_num)
            print("Количество назначений по третьему приоритету:", trd_prior_num+low_prior_num)
        

        # Сохранение конечного результата по листам
        results_file_name = f"{projects_name}_{limits_name}_результат.xlsx"
        writer = pd.ExcelWriter(results_file_name)
        for experiment_idx in range(N_EXPERIMENT_REPEATS):
            result_df = all_results[experiment_idx]
            result_departments = all_experiments_departments[experiment_idx]
            result_priorities = all_experiments_priorities[experiment_idx]
            result_sheet_name = f'Результат {experiment_idx}'
            # добавление цветов (stackoverflow questions/52887800)
            def style_all_cells(df):
                green = 'background-color: lightgreen'
                yellow = 'background-color: yellow'
                red = 'background-color: red'
                def priority_to_color(priority):
                    if priority == 1:
                        return green
                    elif priority == 2:
                        return yellow
                    else:
                        return red
                df_style = pd.DataFrame('', index=df.index, columns=df.columns)
                for student_idx, (department_idx, priority) in enumerate(
                        zip(result_departments, result_priorities)):
                    color = priority_to_color(priority)
                    df_style.iloc[student_idx, department_idx] = color
                return df_style

            # style.apply вычисляет и применяет цвета с сохранением в Excel
            result_df.style.apply(style_all_cells, axis=None).to_excel(
                writer, sheet_name=result_sheet_name)
            for column in result_df:
                # Автоматическся установка ширины столбцов под размер текста
                # (stackoverflow questions/17326973)
                column_length = max(result_df[column].astype(str).map(len).max(),
                                    len(column)) * 4//3
                col_idx = result_df.columns.get_loc(column)
                writer.sheets[result_sheet_name].set_column(
                    col_idx, col_idx + 1, column_length)
        writer.save()
