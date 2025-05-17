import math

# Задача 3: Усовершенствованный метод ломаных (метод Эйлера-Коши или Хейна)
# y' = 0.263 * (x^2 + cos(1.2*x)) + 0.453 * y
# Отрезок [0.2, 1.2], шаг dx = 0.1
# Начальное условие y(0.2) = 0.25
# Вычисления с четырьмя десятичными знаками

# Правая часть дифференциального уравнения dy/dx = f(x, y)
def diff_eq_func(x, y):
    term1 = 0.263 * (x**2 + math.cos(1.2 * x))
    term2 = 0.453 * y
    # Округление промежуточных вычислений, если это строго требуется
    # Например, term1 = round(term1, 5 или больше для промежуточных)
    # result = round(term1 + term2, 5 или больше)
    # Но в задаче сказано "Все вычисления выполнять с четырьмя десятичными знаками".
    # Это может означать округление КАЖДОГО результата операции до 4 знаков,
    # или округление f(x,y) до 4 знаков, или только конечных y_k.
    # Будем округлять результат функции f(x,y) и промежуточные y_{k+1/2}
    
    # Расчет с промежуточным округлением (строго следуя "все вычисления")
    # x_sq_rounded = round(x**2, 4) # Пример, если даже x^2 надо округлять
    # cos_val = math.cos(1.2 * x)
    # cos_val_rounded = round(cos_val, 4) # Округление значения косинуса
    # sum_terms_rounded = round(x_sq_rounded + cos_val_rounded, 4)
    # term1_calc = round(0.263 * sum_terms_rounded, 4)
    # term2_calc = round(0.453 * y, 4)
    # result = round(term1_calc + term2_calc, 4)
    # return result
    # Для большей точности обычно округляют только конечный результат шага y_k.
    # Здесь будем округлять результат f(x,y) и y_{k+1/2} до большего числа знаков (например, 6-7),
    # а y_k до 4 знаков.
    # Однако, если "все вычисления" значит строгое округление, то:
    
    val_cos = math.cos(1.2 * x)
    # val_cos_rounded = round(val_cos, 4) # Строгое округление значения cos
    
    # Промежуточные вычисления с большей точностью, финальный y_k округляется до 4.
    # Либо, если требуется округлять каждый шаг:
    # x_squared = round(x*x, 7)
    # cos_term = round(math.cos(round(1.2*x, 7)), 7)
    # term_in_brackets = round(x_squared + cos_term, 7)
    # part1 = round(0.263 * term_in_brackets, 7)
    # part2 = round(0.453 * y, 7)
    # return round(part1 + part2, 7) # Возвращаем с большей точностью для промежуточных шагов метода

    # Согласно условию "Все вычисления выполнять с четырьмя десятичными знаками",
    # будем стараться округлять результаты операций, которые могут дать много знаков.
    # y передается уже округленным до 4 знаков.
    
    x_sq = x*x 
    cos_val = math.cos(1.2*x)
    
    # Пример строгого округления каждого шага:
    # x_sq_r = round(x*x, 4)
    # cos_val_r = round(math.cos(round(1.2*x,4)), 4) # cos(1.2x)
    # sum_r = round(x_sq_r + cos_val_r, 4)
    # term1_r = round(0.263 * sum_r, 4)
    # term2_r = round(0.453 * y, 4) # y уже y_k (округлен до 4)
    # return round(term1_r + term2_r, 4)
    
    # Менее агрессивное округление: сохраняем больше знаков внутри f(x,y)
    # и округляем y_k+1/2 и y_k+1 до 4 знаков.
    return 0.263 * (x*x + math.cos(1.2*x)) + 0.453 * y


# Усовершенствованный метод ломаных
def improved_euler_method(f_func, x0, y0, h, x_end):
    """
    Решает ОДУ y'=f(x,y) усовершенствованным методом ломаных.
    f_func: функция f(x,y)
    x0, y0: начальные условия
    h: шаг
    x_end: конечная точка x
    Все вычисления с четырьмя десятичными знаками.
    """
    print(f"\nЗадача 3: Усовершенствованный метод ломаных")
    print(f"y' = 0.263(x^2 + cos(1.2x)) + 0.453y")
    print(f"x0 = {x0}, y0 = {y0}, h = {h}, x_end = {x_end}")
    print("------------------------------------")
    print("k  |   x_k   |   y_k   | f(x_k,y_k) | x_{k+1/2} | y_{k+1/2} | f(x_{k+1/2},y_{k+1/2}) | y_{k+1}")
    print("------------------------------------")

    x_k = x0
    y_k = round(y0, 4) # Начальное y округлено
    
    k_count = 0
    results = []
    results.append((k_count, x_k, y_k))

    while x_k < x_end: # Идем до x_end - h, чтобы последний x_k был x_end
        # Вычисляем f(x_k, y_k)
        f_xk_yk = diff_eq_func(x_k, y_k) 
        # f_xk_yk_rounded = round(f_xk_yk, 4) # Округляем значение производной
        f_xk_yk_intermediate_prec = f_xk_yk # Используем без раннего округления для промежуточных

        # Промежуточные значения
        x_k_half = round(x_k + h / 2.0, 5) # x_k + 0.05, точно
        
        # y_{k+1/2} = y_k + (h/2) * f(x_k, y_k)
        y_k_half_val = y_k + (h / 2.0) * f_xk_yk_intermediate_prec
        y_k_half_rounded = round(y_k_half_val, 4) # Округляем y_{k+1/2}

        # Вычисляем f(x_{k+1/2}, y_{k+1/2})
        f_xk_half_yk_half = diff_eq_func(x_k_half, y_k_half_rounded)
        # f_xk_half_yk_half_rounded = round(f_xk_half_yk_half, 4) # Округляем значение производной на промежуточном шаге
        f_xk_half_yk_half_intermediate_prec = f_xk_half_yk_half

        # Новое значение y_{k+1}
        # y_{k+1} = y_k + h * f(x_{k+1/2}, y_{k+1/2})
        y_k_plus_1_val = y_k + h * f_xk_half_yk_half_intermediate_prec
        y_k_plus_1_rounded = round(y_k_plus_1_val, 4) # Округляем y_{k+1} до 4 знаков

        # Печать промежуточных значений для таблицы (округленных до 4 знаков где требуется)
        # Для f(x,y) в таблице лучше показать значение, которое использовалось для расчета y_k+1/2 или y_k+1
        print(f"{k_count:2d} | {x_k:7.4f} | {y_k:7.4f} | {round(f_xk_yk_intermediate_prec,4):8.4f} |  {x_k_half:7.4f}  |  {y_k_half_rounded:7.4f}  |   {round(f_xk_half_yk_half_intermediate_prec,4):15.4f}   | {y_k_plus_1_rounded:7.4f}")

        x_k = round(x_k + h, 2) # Новый x_k, округление чтобы избежать ошибок float типа 0.300000000004
        y_k = y_k_plus_1_rounded
        k_count += 1
        results.append((k_count, x_k, y_k))
        
        if x_k > x_end + h/2: # Небольшой допуск, чтобы точно дойти до x_end
            break
            
    print("------------------------------------")
    return results

# Начальные параметры для Задачи 3
x_initial = 0.2
y_initial = 0.25
step_h = 0.1
x_final = 1.2 # Конечная точка x

solution_points = improved_euler_method(diff_eq_func, x_initial, y_initial, step_h, x_final)

print("\nИтоговые значения y(x):")
for point in solution_points:
    k, x_val, y_val = point
    print(f"x = {x_val:.1f}, y = {y_val:.4f}")

