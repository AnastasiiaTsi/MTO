import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# 1. Задання підінтегральної функції та аналітичного інтегралу
def f(x):
    """
    Інтенсивність добового навантаження на сервер f(x)
    на інтервалі часу x in [0, 24] год.
    """
    return 50.0 + 20.0 * np.sin(np.pi * x / 12.0) + 5.0 * np.exp(-0.2 * ((x - 12.0) ** 2))

a, b = 0.0, 24.0

# Точне значення інтегралу через аналітичний розв'язок та інтеграл помилок:
# int(50) = 1200; int(20*sin) = 0; int(5*exp) = 5 * sqrt(5*pi) * erf(12*sqrt(0.2))
exact_I0 = 1200.0 + 5.0 * np.sqrt(5.0 * np.pi) * erf(12.0 * np.sqrt(0.2))

# 2. Складова квадратурна формула Сімпсона
def simpson_composite(func, a_val, b_val, n_sub):
    """
    Складова формула Сімпсона при парній кількості розбиттів n_sub
    """
    if n_sub % 2 != 0:
        n_sub += 1
    h = (b_val - a_val) / n_sub
    x_nodes = np.linspace(a_val, b_val, n_sub + 1)
    y_nodes = func(x_nodes)
    integ = (h / 3.0) * (y_nodes[0] + y_nodes[-1] + 
                         4.0 * np.sum(y_nodes[1:-1:2]) + 
                         2.0 * np.sum(y_nodes[2:-1:2]))
    return integ

# 3. Дослідження збіжності та визначення N_opt
N_values = np.arange(4, 1002, 2)
errors = []
N_opt = None
eps_target = 1e-12

for N in N_values:
    I_val = simpson_composite(f, a, b, N)
    err = abs(I_val - exact_I0)
    errors.append(err)
    if err <= eps_target and N_opt is None:
        N_opt = N

eps_opt = abs(simpson_composite(f, a, b, N_opt) - exact_I0)

# 4. Розрахунки для базового розбиття N_0
# N_0 береться кратним 8. Вибираємо N_0 = 16 для гарантованого входу
# в асимптотичний режим збіжності методів Рунге-Ромберга та Ейткена
N0 = 16
eps0 = abs(simpson_composite(f, a, b, N0) - exact_I0)

# 5. Метод Рунге-Ромберга
I_N0 = simpson_composite(f, a, b, N0)
I_N0_half = simpson_composite(f, a, b, N0 // 2)
I_R = I_N0 + (I_N0 - I_N0_half) / 15.0
epsR = abs(I_R - exact_I0)

# 6. Метод Ейткена
I_N0_quart = simpson_composite(f, a, b, N0 // 4)
I_E = (I_N0_half**2 - I_N0 * I_N0_quart) / (2.0 * I_N0_half - (I_N0 + I_N0_quart))
epsE = abs(I_E - exact_I0)
p_aitken = (1.0 / np.log(2.0)) * np.log(abs((I_N0_quart - I_N0_half) / (I_N0_half - I_N0)))

# 7. Адаптивний алгоритм чисельного інтегрування (Сімпсон + Рунге)
eval_count = 0

def adaptive_simpson_step(func, a_s, b_s, tol, fa, fb, fc):
    global eval_count
    c = 0.5 * (a_s + b_s)
    h = b_s - a_s
    d = 0.5 * (a_s + c)
    e = 0.5 * (c + b_s)
    fd = func(d)
    fe = func(e)
    eval_count += 2
    
    # Оцінка на інтервалі [a, b] та двох половинах
    I1 = (h / 6.0) * (fa + 4.0 * fc + fb)
    I2 = (h / 12.0) * (fa + 4.0 * fd + 2.0 * fc + 4.0 * fe + fb)
    
    # Локальний критерій Рунге: |I1 - I2|/15 <= tol
    if abs(I1 - I2) <= 15.0 * tol:
        return I2 + (I2 - I1) / 15.0
    else:
        return adaptive_simpson_step(func, a_s, c, tol / 2.0, fa, fc, fd) + \
               adaptive_simpson_step(func, c, b_s, tol / 2.0, fc, fb, fe)

def adaptive_integrate(func, a_s, b_s, tol):
    global eval_count
    eval_count = 3
    fa = func(a_s)
    fb = func(b_s)
    c = 0.5 * (a_s + b_s)
    fc = func(c)
    return adaptive_simpson_step(func, a_s, b_s, tol, fa, fb, fc), eval_count

tol_list = [1e-3, 1e-5, 1e-7, 1e-9, 1e-11]
adapt_results = []

for tol in tol_list:
    val, cnt = adaptive_integrate(f, a, b, tol)
    adapt_results.append((tol, val, abs(val - exact_I0), cnt))

# 8. Побудова та збереження графіків (у режимі файлового виводу)

# Графік 1: Підінтегральна функція навантаження f(x)
plt.figure(figsize=(9, 4.5))
x_plot = np.linspace(a, b, 500)
plt.plot(x_plot, f(x_plot), 'tab:blue', lw=2, 
         label=r'$f(x) = 50 + 20\sin\left(\frac{\pi x}{12}\right) + 5e^{-0.2(x-12)^2}$')
plt.title('Рис. 1. Інтенсивність добового навантаження на сервер f(x)')
plt.xlabel('Час, x (год)')
plt.ylabel('Навантаження, f(x)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('chart1_load_function.png', dpi=300)
plt.close()

# Графік 2: Залежність похибки формули Сімпсона від N
plt.figure(figsize=(9, 4.8))
clean_errors = np.maximum(errors, 1e-15)
plt.semilogy(N_values, clean_errors, 'b-', lw=1.5, label=r'$\epsilon(N) = |I(N) - I_0|$')
plt.axhline(eps_target, color='r', linestyle='--', label=r'Поріг точності $\epsilon = 10^{-12}$')
plt.axvline(N_opt, color='g', linestyle=':', label=f'$N_{{opt}} = {N_opt}$')
plt.title('Рис. 2. Залежність похибки формули Сімпсона від числа розбиттів N')
plt.xlabel('Число розбиттів, N')
plt.ylabel('Абсолютна похибка')
plt.ylim(1e-15, 10)
plt.grid(True, which="both", linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('chart2_error_vs_n.png', dpi=300)
plt.close()

# Графік 3: Робота адаптивного алгоритму (похибка та кількість викликів)
tols = [r[0] for r in adapt_results]
real_errs = [r[2] for r in adapt_results]
counts = [r[3] for r in adapt_results]

fig, ax1 = plt.subplots(figsize=(9, 4.8))

color_err = 'tab:red'
ax1.set_xlabel(r'Заданий допуск точності $\delta$')
ax1.set_ylabel('Фактична похибка', color=color_err)
line1 = ax1.loglog(tols, real_errs, 'o--', color=color_err, lw=2, label='Фактична похибка')
ax1.tick_params(axis='y', labelcolor=color_err)
ax1.grid(True, which="both", linestyle='--', alpha=0.5)

ax2 = ax1.twinx()
color_cnt = 'tab:blue'
ax2.set_ylabel('Кількість викликів f(x)', color=color_cnt)
line2 = ax2.semilogx(tols, counts, 's-', color=color_cnt, lw=2, label='Кількість викликів f(x)')
ax2.tick_params(axis='y', labelcolor=color_cnt)

# Об'єднана легенда для обох осей
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center left')

plt.title('Рис. 3. Характеристики роботи адаптивного алгоритму')
fig.tight_layout()
plt.savefig('chart3_adaptive_performance.png', dpi=300)
plt.close()

# 9. Вивід числових результатів у термінал
print("=" * 65)
print(f"Аналітичний інтеграл I_0: {exact_I0:.12f}")
print(f"N_opt (точність <= {eps_target}): {N_opt}, eps_opt = {eps_opt:.2e}")
print(f"Базове розбиття N_0 = {N0}: eps_0 = {eps0:.2e}")
print(f"Метод Рунге-Ромберга: I_R = {I_R:.12f}, eps_R = {epsR:.2e}")
print(f"Метод Ейткена: I_E = {I_E:.12f}, eps_E = {epsE:.2e}, емпіричний порядок p = {p_aitken:.2f}")
print("=" * 65)
print("Усі графіки збережено у файли: chart1_load_function.png, chart2_error_vs_n.png, chart3_adaptive_performance.png")