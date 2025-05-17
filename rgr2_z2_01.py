import math

# Задача 2: Численное интегрирование по формуле «трех восьмых»
# Интеграл: integral from 1.3 to 3.46 of (1 + 0.9x^2) / (1.5 + sqrt(0.4x^2 + 0.7)) dx
# n1 = 9, n2 = 12

# Подынтегральная функция
def integrand_func(x):
    numerator = 1 + 0.9 * x**2
    denominator = 1.5 + math.sqrt(0.4 * x**2 + 0.7)
    if denominator == 0:
        # Этот случай маловероятен для данных параметров, но для общей функции стоит учесть
        print(f"Ошибка: знаменатель равен нулю при x = {x}")
        return None 
    return numerator / denominator

# Формула «трех восьмых»
def three_eighths_rule(func, a, b, n):
    """
    Вычисляет определенный интеграл по формуле «трех восьмых».
    func: подынтегральная функция
    a, b: пределы интегрирования
    n: количество разбиений (должно быть кратно 3)
    """
    if n % 3 != 0:
        print(f"Ошибка: количество разбиений n={n} не кратно 3.")
        return None

    h = (b - a) / n
    
    y_values = []
    for i in range(n + 1):
        x_i = a + i * h
        y_i = func(x_i)
        if y_i is None: return None # Ошибка в вычислении y_i
        y_values.append(y_i)

    sum1 = y_values[0] + y_values[n]
    
    sum2 = 0
    for i in range(1, n):
        if i % 3 != 0: # Индексы, не кратные 3 (кроме концов)
            sum2 += y_values[i]
            
    sum3 = 0
    for i in range(3, n, 3): # Индексы, кратные 3 (кроме концов, т.е. y_3, y_6, ...)
        sum3 += y_values[i]
        
    integral_value = (3 * h / 8) * (sum1 + 3 * sum2 + 2 * sum3)
    return integral_value

print("\nЗадача 2: Численное интегрирование")
a = 1.3  # Нижний предел
b = 3.46 # Верхний предел

# Расчет для n1 = 9
n1 = 9
integral_n1 = three_eighths_rule(integrand_func, a, b, n1)
if integral_n1 is not None:
    print(f"Приближенное значение интеграла для n1 = {n1}: {integral_n1:.6f}")

# Расчет для n2 = 12
n2 = 12
integral_n2 = three_eighths_rule(integrand_func, a, b, n2)
if integral_n2 is not None:
    print(f"Приближенное значение интеграла для n2 = {n2}: {integral_n2:.6f}")

# Контроль точности (например, по правилу Рунге, но здесь просто двойной просчет)
# Обычно, чем больше n, тем точнее результат.
# Разница между integral_n1 и integral_n2 может дать оценку погрешности.
if integral_n1 is not None and integral_n2 is not None:
    print(f"Разница между значениями |I(n2) - I(n1)|: {abs(integral_n2 - integral_n1):.6f}")

