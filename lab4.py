import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. Аналітичне розв'язання та точне значення похідної (Завдання 1)
# Модель вологості ґрунту: M(t) = 50*exp(-0.1*t) + 5*sin(t)
def M(t):
    return 50.0 * np.exp(-0.1 * t) + 5.0 * np.sin(t)

def dM_exact(t):
    # Аналітична похідна: M'(t) = -5*exp(-0.1*t) + 5*cos(t)
    return -5.0 * np.exp(-0.1 * t) + 5.0 * np.cos(t)

t0 = 1.0
exact_val = dM_exact(t0)

print("=" * 70)
print("Аналітичний розрахунок:")
print(f"Точка диференціювання: t0 = {t0}")
print(f"Аналітичний вираз похідної: M'(t) = -5*exp(-0.1*t) + 5*cos(t)")
print(f"Точне значення похідної M'(1): {exact_val:.12f}")
print("=" * 70)

# 2. Регуляризація за кроком h та дослідження похибки (Завдання 2)
# Дослідження похибки в діапазоні кроків h від 1e-20 до 1e3
def central_diff(f, x, h):
    return (f(x + h) - f(x - h)) / (2.0 * h)

# Логарифмічна сітка кроків від 10^-18 до 10^2
h_steps = np.logspace(-18, 2, 400)
errors = np.array([abs(central_diff(M, t0, h) - exact_val) for h in h_steps])

# Пошук мінімальної похибки та оптимального кроку h0
min_err_idx = np.argmin(errors)
h0_opt = h_steps[min_err_idx]
R0_opt = errors[min_err_idx]

print(f"\nРегуляризація за кроком h:")
print(f"Оптимальний крок h0:       {h0_opt:.2e}")
print(f"Найкраща досягнута точність R0: {R0_opt:.2e}")

# Графік залежності похибки від кроку h (U-подібна крива)
plt.figure(figsize=(9, 5))
plt.loglog(h_steps, errors, label=r"Похибка $R(h) = |D(h) - M'(t_0)|$", color="tab:blue", linewidth=2)
plt.scatter([h0_opt], [R0_opt], color="red", s=80, zorder=5, label=f"Оптимум: $h_0 \\approx {h0_opt:.1e}$, $R_0 \\approx {R0_opt:.1e}$")
plt.axvline(h0_opt, color="red", linestyle="--", alpha=0.5)
plt.xlabel("Крок диференціювання $h$ (log scale)")
plt.ylabel("Абсолютна похибка $R$ (log scale)")
plt.title("Залежність похибки чисельного диференціювання від величини кроку h")
plt.grid(True, which="both", linestyle=":")
plt.legend()
plt.tight_layout()
plt.show()

# 3. Дослідження методу Рунге-Ромберга 
h_base = 1e-3

# Наближення з кроками h та 2h
D_h = central_diff(M, t0, h_base)
D_2h = central_diff(M, t0, 2.0 * h_base)

R1 = abs(D_h - exact_val)
R_2h = abs(D_2h - exact_val)

# Уточнення за правилом Рунге-Ромберга (для другого порядку p=2: знаменник 2^2 - 1 = 3)
D_RR = D_h + (D_h - D_2h) / 3.0
R2 = abs(D_RR - exact_val)

print("\n" + "=" * 70)
print(f"Метод Рунге-Ромберга (базовий крок h = {h_base}):")
print(f"D(h)  = {D_h:.12f} | Похибка R1   = {R1:.4e}")
print(f"D(2h) = {D_2h:.12f} | Похибка R(2h) = {R_2h:.4e}")
print(f"D_RR  = {D_RR:.12f} | Похибка R2   = {R2:.4e}")
print(f"Коефіцієнт зменшення похибки: {R1 / R2:.2f} разів")
print("=" * 70)

# 4. Метод Ейткена та оцінка порядку точності (Завдання 7)
# Три кроки: h, 2h, 4h
D_4h = central_diff(M, t0, 4.0 * h_base)
R_4h = abs(D_4h - exact_val)

# Екстраполяція Ейткена
denom_aitken = 2.0 * D_2h - (D_4h + D_h)
D_E = (D_2h**2 - D_4h * D_h) / denom_aitken
R3 = abs(D_E - exact_val)

# Оцінка фактичного порядку точності p
p_aitken = (1.0 / np.log(2.0)) * np.log(abs((D_4h - D_2h) / (D_2h - D_h)))

print("\n" + "=" * 70)
print(f"Метод Уйткена та оцінка порядку точності р (кроки: h, 2h, 4h):")
print(f"D(4h) = {D_4h:.12f} | Похибка R(4h) = {R_4h:.4e}")
print(f"D_E   = {D_E:.12f} | Похибка R3   = {R3:.4e}")
print(f"Чисельно оцінений порядок точності p: {p_aitken:.4f} (теоретичний p = 2)")
print("=" * 70)

# 5. Порівняльна таблиця результатів
results_data = [
    {"Метод": "Точне значення", "Крок": "-", "Значення похідної": f"{exact_val:.12f}", "Абсолютна похибка": f"{0.0:.1e}"},
    {"Метод": "Центральна різниця D(h)", "Крок": f"h = {h_base}", "Значення похідної": f"{D_h:.12f}", "Абсолютна похибка": f"{R1:.4e}"},
    {"Метод": "Центральна різниця D(2h)", "Крок": f"2h = {2*h_base}", "Значення похідної": f"{D_2h:.12f}", "Абсолютна похибка": f"{R_2h:.4e}"},
    {"Метод": "Центральна різниця D(4h)", "Крок": f"4h = {4*h_base}", "Значення похідної": f"{D_4h:.12f}", "Абсолютна похибка": f"{R_4h:.4e}"},
    {"Метод": "Рунге-Ромберг", "Крок": "h, 2h", "Значення похідної": f"{D_RR:.12f}", "Абсолютна похибка": f"{R2:.4e}"},
    {"Метод": "Ейткен", "Крок": "h, 2h, 4h", "Значення похідної": f"{D_E:.12f}", "Абсолютна похибка": f"{R3:.4e}"},
    {"Метод": "Оптимальний крок h0", "Крок": f"h0 = {h0_opt:.2e}", "Значення похідної": f"{central_diff(M, t0, h0_opt):.12f}", "Абсолютна похибка": f"{R0_opt:.4e}"}
]

df_results = pd.DataFrame(results_data)
print("\n Зведена таблиця точності чисельного диференціювання")
print(df_results.to_string(index=False))