import math
import matplotlib.pyplot as plt # Импорт для построения графиков
import numpy as np # Импорт для создания массива значений x

# Задача 1: Комбинированный метод хорд и касательных
# Уравнение: x^3 - 12x + 10 = 0
# Точность: 0.001

# Определяем функцию и ее производные
def f(x):
    return x**3 - 12*x + 10

def f_prime(x): # Первая производная
    return 3*x**2 - 12

def f_double_prime(x): # Вторая производная
    return 6*x

# Комбинированный метод хорд и касательных
def combined_method(a, b, epsilon):
    """
    Реализация комбинированного метода хорд и касательных.
    a, b: границы интервала
    epsilon: требуемая точность
    """
    print(f"\nПоиск корня на интервале [{a}, {b}]")
    
    x_n = 0
    xt_n = 0 
    iterations = 0

    # Убедимся, что f(a) и f(b) имеют разные знаки
    if f(a) * f(b) >= 0:
        print(f"Ошибка: Значения функции на концах интервала [{a}, {b}] одного знака или один из них ноль.")
        print(f"f({a}) = {f(a)}, f({b}) = {f(b)}")
        return None

    # Определение случая (а) или (б) для выбора начальных приближений и формул
    # Это упрощенное определение, основанное на знаках f(endpoint) и f''(endpoint)
    # Предполагаем, что f''(x) не меняет знак слишком часто внутри узких интервалов локализации.
    use_case_a = False
    # Проверяем условие f(a) * f''(x) > 0 (для случая а)
    # или f(b) * f''(x) > 0 (для случая б)
    # Знак f''(x) на интервале [a,b] можно оценить по f''((a+b)/2) или на концах.
    
    # Оцениваем знак f''(x) на интервале. Если он меняется, метод может быть менее стабилен.
    # Для f''(x) = 6x, знак меняется в x=0.
    f_double_prime_sign_at_a = f_double_prime(a)
    f_double_prime_sign_at_b = f_double_prime(b)
    
    # Упрощенное определение случая:
    # Если f(a) и f''(a) одного знака И f''(x) не меняет знак на [a,b] (или f''(a) и f''(b) одного знака)
    # -> Случай (а): x0=a (Ньютон), xt0=b (Хорды)
    # Если f(b) и f''(b) одного знака И f''(x) не меняет знак на [a,b] (или f''(a) и f''(b) одного знака)
    # -> Случай (б): x0=a (Хорды), xt0=b (Ньютон)

    # Более надежный подход из методички:
    # 1. Выбираем начальное приближение x0 так, чтобы f(x0) * f''(x0) > 0.
    #    Если f(a) * f_double_prime(a) > 0, то x0 = a (Ньютон), xt0 = b (Хорды) - это наш случай (а)
    #    Если f(b) * f_double_prime(b) > 0, то x0 = b (Ньютон), xt0 = a (Хорды) - это соответствует случаю (б) с заменой ролей a и b в формулах.
    #    Для простоты, мы будем придерживаться обозначений x_n и xt_n из лекций.

    if f(a) * f_double_prime(a) > 0: # f(a) и f''(a) одного знака
        # Это условие для начальной точки метода Ньютона x_n = a
        # Тогда для метода хорд xt_n = b
        x_n = a
        xt_n = b
        use_case_a = True # Ньютон с x_n, Хорды с xt_n (x_n уточняется Ньютоном, xt_n хордами от x_n)
        print(f"Условие f(a)*f''(a) > 0 выполнено (f({a})={f(a):.3f}, f''({a})={f_double_prime(a):.3f}). Случай (а).")
    elif f(b) * f_double_prime(b) > 0: # f(b) и f''(b) одного знака
        # Это условие для начальной точки метода Ньютона xt_n = b
        # Тогда для метода хорд x_n = a
        x_n = a  # Хорды с x_n
        xt_n = b # Ньютон с xt_n
        use_case_a = False # Хорды с x_n, Ньютон с xt_n (xt_n уточняется Ньютоном, x_n хордами от xt_n)
        print(f"Условие f(b)*f''(b) > 0 выполнено (f({b})={f(b):.3f}, f''({b})={f_double_prime(b):.3f}). Случай (б).")
    else:
        # Если ни одно из условий не выполнено строго, возможно, f''(x) меняет знак или близка к нулю.
        # Это может потребовать более детального анализа или разбиения интервала.
        # Для учебной задачи, если условия не ясны, можно попробовать один из вариантов или выдать предупреждение.
        print(f"Предупреждение: Не удалось однозначно определить случай (а) или (б) по знакам f и f'' на концах интервала [{a}, {b}].")
        print(f"f({a})={f(a):.3f}, f''({a})={f_double_prime(a):.3f}")
        print(f"f({b})={f(b):.3f}, f''({b})={f_double_prime(b):.3f}")
        # По умолчанию, если f(a) < 0 и f(b) > 0:
        if f(a) < f(b): # f(a) ниже, f(b) выше
            if f_double_prime((a+b)/2) > 0 : # Вогнутая, f(b)f''>0
                 x_n = a; xt_n = b; use_case_a = False; print("Выбран случай (б) эвристически.")
            else: # Выпуклая, f(a)f''>0
                 x_n = a; xt_n = b; use_case_a = True; print("Выбран случай (а) эвристически.")
        else: # f(a) > 0 и f(b) < 0
            if f_double_prime((a+b)/2) < 0 : # Выпуклая, f(a)f''>0
                 x_n = a; xt_n = b; use_case_a = True; print("Выбран случай (а) эвристически.")
            else: # Вогнутая, f(b)f''>0
                 x_n = a; xt_n = b; use_case_a = False; print("Выбран случай (б) эвристически.")


    print(f"Начальные значения: x_n = {x_n}, xt_n = {xt_n}")
    if use_case_a:
        print("Используется случай (а): x_n уточняется Ньютоном, xt_n - методом хорд.")
    else:
        print("Используется случай (б): x_n уточняется методом хорд, xt_n - Ньютоном.")

    while abs(xt_n - x_n) > epsilon:
        iterations += 1
        if iterations > 100: # Предохранитель от бесконечного цикла
            print("Превышено максимальное количество итераций.")
            return None

        fx_n = f(x_n)
        fxt_n = f(xt_n)

        # Проверка деления на ноль (упрощенная, без try-except)
        denominator_newton_xn = f_prime(x_n)
        denominator_newton_xtn = f_prime(xt_n)
        denominator_chord = (fxt_n - fx_n)

        if use_case_a: # x_n - Ньютон, xt_n - Хорды
            if denominator_newton_xn == 0:
                print(f"Ошибка: Производная f'({x_n:.4f}) равна нулю. Невозможно применить метод Ньютона.")
                return None
            if denominator_chord == 0:
                print(f"Ошибка: f(xt_n) - f(x_n) равно нулю ({fxt_n:.4f} - {fx_n:.4f}). Невозможно применить метод хорд.")
                return None
            
            x_n_new = x_n - fx_n / denominator_newton_xn
            # Хорда проводится между (x_n, f(x_n)) и (xt_n, f(xt_n)), ищется пересечение с Ox
            # Но в комбинированном методе, если x_n - это точка для Ньютона, то xt_n - это "другой конец" для хорды.
            # Формула для хорд: x_new = x_curr - f(x_curr) * (x_other_end - x_curr) / (f(x_other_end) - f(x_curr))
            # Здесь xt_n уточняется хордой, используя x_n как неподвижную точку для итерации хорд (по схеме из лекций)
            # xt_n_new = xt_n - fxt_n * (xt_n - x_n) / (fxt_n - fx_n) # Это была бы обычная хорда для xt_n
            # По методичке: xt_{k+1} = xt_k - f(xt_k) * (xt_k - x_k) / (f(xt_k) - f(x_k)) если x_k неподвижная точка Ньютона
            # или xt_{k+1} = x_k - f(x_k) * (xt_k - x_k) / (f(xt_k) - f(x_k))
            # Для случая (а), xt_n уточняется хордой, x_n - Ньютоном.
            # xt_n_new = xt_n - f(xt_n) * (xt_n - x_n_new) / (f(xt_n) - f(x_n_new)) # Если x_n_new уже посчитан
            # Классическая схема: одна точка движется по касательной, другая - по хорде к текущим приближениям.
            # xt_n_new = x_n - (fx_n * (xt_n - x_n)) / (fxt_n - fx_n) # Если x_n - "якорь" хорды
            xt_n_new = xt_n - fxt_n * (xt_n - x_n) / (fxt_n - fx_n) # Уточнение xt_n методом хорд, где x_n - второй конец хорды
                                                                 # Это если xt_n - подвижная точка хорды
            # По Приложению 1, случай (а):
            # x_{k+1} = x_k - f(x_k)/f'(x_k)
            # \bar{x}_{k+1} = x_k - f(x_k) * (\bar{x}_k - x_k) / (f(\bar{x}_k) - f(x_k))
            # Здесь x_k -> x_n, \bar{x}_k -> xt_n
            # xt_n_new = x_n - fx_n * (xt_n - x_n) / (fxt_n - fx_n) # Это формула для \bar{x}_{k+1}

        else: # Случай (б): x_n - Хорды, xt_n - Ньютон
            if denominator_newton_xtn == 0:
                print(f"Ошибка: Производная f'({xt_n:.4f}) равна нулю. Невозможно применить метод Ньютона.")
                return None
            if denominator_chord == 0:
                print(f"Ошибка: f(xt_n) - f(x_n) равно нулю ({fxt_n:.4f} - {fx_n:.4f}). Невозможно применить метод хорд.")
                return None
            
            # По Приложению 1, случай (б):
            # x_{k+1} = x_k - f(x_k) * (\bar{x}_k - x_k) / (f(\bar{x}_k) - f(x_k))
            # \bar{x}_{k+1} = \bar{x}_k - f(\bar{x}_k)/f'(\bar{x}_k)
            # Здесь x_k -> x_n, \bar{x}_k -> xt_n
            x_n_new = x_n - fx_n * (xt_n - x_n) / (fxt_n - fx_n)
            xt_n_new = xt_n - fxt_n / denominator_newton_xtn
        
        x_n = x_n_new
        xt_n = xt_n_new

        # print(f"Итерация {iterations}: x_n = {x_n:.6f}, xt_n = {xt_n:.6f}, |xt_n - x_n| = {abs(xt_n - x_n):.6f}")

    root = (x_n + xt_n) / 2
    print(f"Корень найден: {root:.4f} после {iterations} итераций.")
    print(f"f({root:.4f}) = {f(root):.6f}")
    print(f"Интервал [{min(x_n, xt_n):.6f}, {max(x_n, xt_n):.6f}], ширина = {abs(xt_n - x_n):.6f}")
    return root

# Функция для построения графика
def plot_function_and_roots(roots, x_min=-5, x_max=5, num_points=400):
    """
    Строит график функции f(x) и отмечает найденные корни.
    roots: список найденных корней.
    x_min, x_max: диапазон для оси X.
    num_points: количество точек для построения графика функции.
    """
    x_vals = np.linspace(x_min, x_max, num_points)
    y_vals = f(x_vals)

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label="f(x) = x^3 - 12x + 10", color='blue')
    
    # Ось X (y=0)
    plt.axhline(0, color='black', lw=0.8)
    # Ось Y (x=0)
    plt.axvline(0, color='black', lw=0.8)

    # Отмечаем найденные корни
    valid_roots = [r for r in roots if r is not None]
    if valid_roots:
        plt.scatter(valid_roots, [f(r) for r in valid_roots], color='red', s=50, zorder=5, label="Найденные корни")
        for i, r_val in enumerate(valid_roots):
            plt.text(r_val, f(r_val) + 0.5, f"x{i+1} ≈ {r_val:.3f}", ha='center', color = 'red')


    plt.title("График функции и найденные корни")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.ylim(min(y_vals.min(), -10, f(-2)-5 if f(-2) else -10), max(y_vals.max(), 10, f(-2)+5 if f(-2) else 10)) # Динамический диапазон Y с учетом экстремумов
    plt.show()

# --- Основной блок ---
if __name__ == "__main__":
    print("Задача 1: Решение нелинейного уравнения x^3 - 12x + 10 = 0")
    print("f(x) = x^3 - 12x + 10")
    print("f'(x) = 3x^2 - 12")
    print("f''(x) = 6x")
    print("\nАналитическое отделение корней (поиск интервалов знакопеременности):")

    print("f(-4) =", f(-4))
    print("f(-3) =", f(-3)) # Корень в [-4, -3]
    print("f(0) =", f(0))
    print("f(1) =", f(1))
    # Уточненный интервал для второго корня:
    # f(0.8) = 0.912, f(0.9) = -0.071 => корень в [0.8, 0.9]
    print("f(0.8) =", f(0.8))
    print("f(0.9) =", f(0.9))
    print("f(2) =", f(2))
    print("f(3) =", f(3))   # Корень в [2, 3]

    epsilon = 0.001
    all_roots = []

    # Поиск корней на выделенных интервалах
    # Интервал 1: [-4, -3]
    # f''(-4) = -24, f''(-3) = -18. Знак f'' < 0 на интервале.
    # f(-4) = -6. f(-4) * f''(-4) = (-6)*(-24) > 0. Случай (а).
    root1 = combined_method(-4, -3, epsilon)
    all_roots.append(root1)

    # Интервал 2: [0.8, 0.9]
    # f''(0.8) = 4.8, f''(0.9) = 5.4. Знак f'' > 0 на интервале.
    # f(0.8) = 0.912. f(0.8) * f''(0.8) = (0.912)*(4.8) > 0. Случай (а).
    root2 = combined_method(0.8, 0.9, epsilon)
    all_roots.append(root2)

    # Интервал 3: [2, 3]
    # f''(2) = 12, f''(3) = 18. Знак f'' > 0 на интервале.
    # f(2) = -6. f(3) = 1.
    # f(2) * f''(2) = (-6)*(12) < 0.
    # f(3) * f''(3) = (1)*(18) > 0. Случай (б).
    root3 = combined_method(2, 3, epsilon)
    all_roots.append(root3)

    print("\nНайденные корни (с точностью до 0.001):")
    if root1 is not None: print(f"Корень 1: {root1:.3f}")
    if root2 is not None: print(f"Корень 2: {root2:.3f}")
    if root3 is not None: print(f"Корень 3: {root3:.3f}")

    # Графическая иллюстрация
    print("\nПостроение графика функции для иллюстрации...")
    # Определяем диапазон для графика на основе найденных корней и экстремумов
    # Экстремумы: x = +/-2. f(2) = -6, f(-2) = 26.
    # Корни примерно -3.7, 0.8, 2.8
    plot_x_min = -4.5
    plot_x_max = 4.0
    if all_roots and all(r is not None for r in all_roots):
        # Немного расширим диапазон за пределы крайних корней
        min_r = min(all_roots)
        max_r = max(all_roots)
        plot_x_min = min(plot_x_min, min_r - 0.5)
        plot_x_max = max(plot_x_max, max_r + 0.5)

    plot_function_and_roots(all_roots, x_min=plot_x_min, x_max=plot_x_max)

