import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

months = np.arange(1, 25)
temps = np.array([
    -2, 0, 5, 10, 15, 20, 23, 22, 17, 10, 5, 0,
    -10, 3, 7, 13, 19, 20, 22, 21, 18, 15, 10, 3
], dtype=float)

csv_filename = "temperature_data.csv"
df_temp = pd.DataFrame({"Month": months, "Temp": temps})
df_temp.to_csv(csv_filename, index=False)

print("=" * 60)
print(f"Дані успішно протабульовано та збережено у '{csv_filename}':")
print(df_temp.to_string(index=False))
print("=" * 60)

# 2. Метод найменших квдратів та метод Гаусса
def form_normal_matrix(x, m):
    """Формування матриці B нормальної системи: b_kl = sum(x_i^(k+l))."""
    B = np.zeros((m + 1, m + 1), dtype=float)
    for k in range(m + 1):
        for l in range(m + 1):
            B[k, l] = np.sum(x ** (k + l))
    return B

def form_free_vector(x, y, m):
    """Формування вектора правих частин C: c_k = sum(y_i * x_i^k)."""
    C = np.zeros(m + 1, dtype=float)
    for k in range(m + 1):
        C[k] = np.sum(y * (x ** k))
    return C

def gauss_solve_pivot(A_in, b_in):
    """
    Розв'язання системи A*x = b методом Гаусса 
    з вибором головного елемента по стовпцях.
    """
    n = len(b_in)
    A = A_in.copy().astype(float)
    b = b_in.copy().astype(float)

    # Прямий хід
    for k in range(n):
        # Пошук рядка з найбільшим за модулем елементом у k-му стовпчику
        max_row = k + np.argmax(np.abs(A[k:, k]))
        if max_row != k:
            A[[k, max_row]] = A[[max_row, k]]
            b[[k, max_row]] = b[[max_row, k]]

        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    # Зворотний хід
    x_sol = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        x_sol[i] = (b[i] - np.sum(A[i, i + 1:] * x_sol[i + 1:])) / A[i, i]
    return x_sol

def evaluate_poly(coefs, x):
    """Обчислення полінома: phi(x) = sum(a_j * x^j)."""
    val = np.zeros_like(x, dtype=float)
    for j, a in enumerate(coefs):
        val += a * (x ** j)
    return val

def calc_variance(y_true, y_approx):
    """Обчислення середньоквадратичного відхилення (дисперсії)."""
    return np.sqrt(np.sum((y_approx - y_true) ** 2) / len(y_true))

# 3. ДДослідження дисперсії для степенів
degrees = np.arange(1, 11)
variances = []
models = {}

for m in degrees:
    B = form_normal_matrix(months, m)
    C = form_free_vector(months, temps, m)
    coefs = gauss_solve_pivot(B, C)
    y_appr = evaluate_poly(coefs, months)
    var = calc_variance(temps, y_appr)
    
    variances.append(var)
    models[m] = {"coefs": coefs, "approx": y_appr, "var": var}

# Таблиця дисперсій
df_var = pd.DataFrame({"Степінь m": degrees, "Дисперсія (СКО) delta": variances})
print("\nДослідження дисперсії для сетпені полінома:")
print(df_var.to_string(index=False))

# Визначення оптимуму
opt_m = degrees[np.argmin(variances)]
print(f"\nОптимальний степінь за критерієм мінімуму похибки: m = {opt_m}")
print("Знайдені коефіцієнти a_0, ..., a_m:")
for idx, c in enumerate(models[opt_m]["coefs"]):
    print(f"  a_{idx} = {c:+.8e}")

# 4. Побудова графіків
x_dense = np.linspace(1, 24, 400)
x_future = np.array([25, 26, 27])
y_future = evaluate_poly(models[opt_m]["coefs"], x_future)

fig = plt.figure(figsize=(12, 14))

# Графік 1: Дисперсія від степеня полінома
ax1 = fig.add_subplot(3, 1, 1)
ax1.plot(degrees, variances, "o-", color="tab:blue", linewidth=2, markersize=7)
ax1.scatter([opt_m], [models[opt_m]["var"]], color="red", s=120, zorder=5, label=f"Оптимум (m={opt_m})")
ax1.set_title("Залежність дисперсії (СКО) від степеня апроксимуючого полінома", fontsize=12)
ax1.set_xlabel("Степінь полінома m")
ax1.set_ylabel("Дисперсія delta (°C)")
ax1.set_xticks(degrees)
ax1.grid(True, linestyle=":")
ax1.legend()

# Графік 2: Апроксимація та екстраполяція (прогноз)
ax2 = fig.add_subplot(3, 1, 2)
ax2.scatter(months, temps, color="black", s=35, zorder=5, label="Фактичні спостереження (24 міс.)")
for m_sample, col in [(2, "tab:cyan"), (4, "tab:green"), (opt_m, "tab:red")]:
    ax2.plot(x_dense, evaluate_poly(models[m_sample]["coefs"], x_dense), 
             label=f"Поліном m={m_sample}", color=col, linewidth=1.8)

# Точки прогнозу
ax2.scatter(x_future, y_future, color="magenta", marker="*", s=140, zorder=6, label="Прогноз (25, 26, 27 міс.)")
ax2.plot(np.append(months[-1], x_future), np.append(temps[-1], y_future), "--", color="magenta", alpha=0.8)
ax2.set_title(f"Апроксимація температурного профілю та прогноз на 3 місяці (m={opt_m})", fontsize=12)
ax2.set_xlabel("Місяць")
ax2.set_ylabel("Температура (°C)")
ax2.set_xticks(np.arange(1, 28))
ax2.grid(True, linestyle=":")
ax2.legend()

# Графік 3: Абсолютна похибка апроксимації
ax3 = fig.add_subplot(3, 1, 3)
for m_sample, col in [(2, "tab:cyan"), (4, "tab:green"), (opt_m, "tab:red")]:
    err = np.abs(temps - models[m_sample]["approx"])
    ax3.plot(months, err, "o-", label=f"Похибка epsilon для m={m_sample}", color=col, linewidth=1.5)

ax3.set_title("Графік абсолютної похибки апроксимації epsilon(x) = |f(x) - phi(x)|", fontsize=12)
ax3.set_xlabel("Місяць")
ax3.set_ylabel("Абсолютна похибка (°C)")
ax3.set_xticks(months)
ax3.grid(True, linestyle=":")
ax3.legend()

plt.tight_layout()
plt.show()

# 5. Вивід прогнозу температури на наступні 3 місяці
print("=" * 60)
print(f"Прогноз температури на наступні 3 місяці (для m = {opt_m}):")
for m_idx, t_pred in zip(x_future, y_future):
    print(f"  Місяць {m_idx}: {t_pred:+.2f} °C")
print("=" * 60)