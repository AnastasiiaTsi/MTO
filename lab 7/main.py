import random
import matplotlib.pyplot as plt
import numpy as np

# 1. Генерація системи та робота з текстовими файлами
def generate_and_save_system(
    n=100,
    exact_val=2.5,
    matrix_file="matrix_A.txt",
    vector_file="vector_b.txt",
):
  
  random.seed(42)  # Фіксація генератора для повної відтворюваності експерименту
  A = []
  for i in range(n):
    row = [random.uniform(-10.0, 10.0) for _ in range(n)]
    # Умова строгого діагонального переважання: |a_ii| > sum_{j != i} |a_ij|
    sum_off_diag = sum(abs(row[j]) for j in range(n) if j != i)
    row[i] = sum_off_diag + random.uniform(5.0, 15.0)
    A.append(row)

  # Заданий точний вектор розв'язку x_i = 2.5 (i = 1..n)
  x_exact = [exact_val] * n

  # Обчислення вектора b = A * x_exact
  b = [sum(A[i][j] * x_exact[j] for j in range(n)) for i in range(n)]

  # Експорт матриці A у текстовий файл
  with open(matrix_file, "w", encoding="utf-8") as f_a:
    for row in A:
      f_a.write(" ".join(f"{val:.10e}" for val in row) + "\n")

  # Експорт вектора b у текстовий файл
  with open(vector_file, "w", encoding="utf-8") as f_b:
    for val in b:
      f_b.write(f"{val:.10e}\n")

  print(
      f"[+] Згенеровано та збережено СЛАР розмірністю {n}x{n} у файли:"
      f" {matrix_file}, {vector_file}"
  )


def read_matrix(filename):
  """Зчитує матрицю A з текстового файлу."""
  A = []
  with open(filename, "r", encoding="utf-8") as f:
    for line in f:
      if line.strip():
        A.append([float(x) for x in line.split()])
  return A


def read_vector(filename):
  """Зчитує вектор b з текстового файлу."""
  b = []
  with open(filename, "r", encoding="utf-8") as f:
    for line in f:
      if line.strip():
        b.append(float(line.strip()))
  return b

# 2. Допоміжні функції матрично-векторних операцій 
def matrix_vector_mult(A, x):
  """Обчислення добутку матриці на вектор: y = A * x."""
  n = len(A)
  return [sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]


def vector_norm_inf(v):
  """Обчислення нескінченної норми вектора (норми Чебишова): max_i |v_i|."""
  return max(abs(x) for x in v)


def matrix_norm_inf(A):
  """Обчислення нескінченної норми матриці: max_i sum_j |a_ij|."""
  return max(sum(abs(val) for val in row) for row in A)

# 3. Реалізація ітераційних методів розв'язку СЛАР 
def solve_simple_iteration(A, b, x0, eps=1e-14, max_iter=50000):
  """Метод простої ітерації:

  X^(k+1) = (E - tau * A) * X^(k) + tau * b
  де 0 < tau < 2 / ||A||. Обираємо tau = 1 / ||A||_inf для гарантованої
  збіжності.
  """
  n = len(A)
  norm_A = matrix_norm_inf(A)
  tau = 1.0 / norm_A

  x = list(x0)
  history = []

  for it in range(1, max_iter + 1):
    Ax = matrix_vector_mult(A, x)
    x_next = [x[i] - tau * (Ax[i] - b[i]) for i in range(n)]

    diff = max(abs(x_next[i] - x[i]) for i in range(n))
    history.append(diff)
    x = x_next

    if diff < eps:
      return x, it, history

  return x, max_iter, history


def solve_jacobi(A, b, x0, eps=1e-14, max_iter=50000):
  """Метод Якобі:

  x_i^(k+1) = (b_i - sum_{j != i} a_ij * x_j^(k)) / a_ii
  """
  n = len(A)
  x = list(x0)
  x_next = [0.0] * n
  history = []

  for it in range(1, max_iter + 1):
    diff = 0.0
    for i in range(n):
      sum_off = sum(A[i][j] * x[j] for j in range(n) if j != i)
      x_next[i] = (b[i] - sum_off) / A[i][i]
      diff = max(diff, abs(x_next[i] - x[i]))

    history.append(diff)
    x = list(x_next)

    if diff < eps:
      return x, it, history

  return x, max_iter, history


def solve_seidel(A, b, x0, eps=1e-14, max_iter=50000):
  """Метод Гауса-Зейделя:

  x_i^(k+1) = (b_i - sum_{j < i} a_ij * x_j^(k+1) - sum_{j > i} a_ij * x_j^(k)) /
  a_ii
  """
  n = len(A)
  x = list(x0)
  history = []

  for it in range(1, max_iter + 1):
    diff = 0.0
    for i in range(n):
      s1 = sum(A[i][j] * x[j] for j in range(i))
      s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
      new_val = (b[i] - s1 - s2) / A[i][i]

      diff = max(diff, abs(new_val - x[i]))
      x[i] = new_val

    history.append(diff)

    if diff < eps:
      return x, it, history

  return x, max_iter, history

# 4. Візуалізація результатів 
def plot_results(
    n,
    eps,
    hist_sd,
    it_sd,
    hist_jc,
    it_jc,
    hist_sp,
    it_sp,
    x_sd,
    x_jc,
    exact_val=2.5,
):
  """Будує графіки швидкості збіжності та розподілу похибок за компонентами."""
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

  # Графік 1: Збіжність за ітераціями 
  ax1.plot(
      range(1, len(hist_sd) + 1),
      hist_sd,
      "b-o",
      markersize=4,
      linewidth=1.8,
      label=f"Гауса–Зейделя ({it_sd} іт.)",
  )
  ax1.plot(
      range(1, len(hist_jc) + 1),
      hist_jc,
      "g-s",
      markersize=3.5,
      linewidth=1.6,
      label=f"Якобі ({it_jc} іт.)",
  )

  # Відобразимо перші 60 кроків простої ітерації для контрасту напівлогарифмічного масштабу
  cutoff = min(60, len(hist_sp))
  ax1.plot(
      range(1, cutoff + 1),
      hist_sp[:cutoff],
      "r-^",
      markersize=3.5,
      linewidth=1.5,
      label="Простої ітерації (початок спаду)",
  )

  ax1.axhline(
      eps,
      color="black",
      linestyle="--",
      linewidth=1.2,
      label=r"Поріг збіжності $\varepsilon_0 = 10^{-14}$",
  )
  ax1.set_yscale("log")
  ax1.set_title(
      "Рис. 1. Швидкість збіжності ітераційних методів (n = 100)",
      fontsize=11,
      fontweight="bold",
      pad=10,
  )
  ax1.set_xlabel("Номер ітерації (k)", fontsize=10)
  ax1.set_ylabel(
      r"Норма нев'язки $\|AX^{(k)} - b\|_{\infty}$ (log scale)", fontsize=10
  )
  ax1.set_xlim(-1, 55)
  ax1.legend(loc="upper right", frameon=True, fontsize=9)
  ax1.grid(True, which="both", ls="--", alpha=0.5)

  # Графік 2: Покомпонентний розподіл похибки 
  components = np.arange(1, n + 1)
  err_jacobi = [abs(x_jc[i] - exact_val) for i in range(n)]
  err_seidel = [abs(x_sd[i] - exact_val) for i in range(n)]

  ax2.plot(
      components,
      err_jacobi,
      color="#2ca02c",
      alpha=0.7,
      linewidth=1.2,
      label="Фінальна похибка: Якобі",
  )
  ax2.plot(
      components,
      err_seidel,
      color="#1f77b4",
      linewidth=1.4,
      label="Фінальна похибка: Гауса–Зейделя",
  )
  ax2.axhline(
      eps,
      color="red",
      linestyle=":",
      linewidth=1.2,
      label=r"Вимога точності $10^{-14}$",
  )

  ax2.set_yscale("log")
  ax2.set_title(
      "Рис. 2. Розподіл залишкової похибки $|x_i^{(final)} - 2.5|$ за"
      " компонентами",
      fontsize=11,
      fontweight="bold",
      pad=10,
  )
  ax2.set_xlabel("Індекс компоненти (i = 1..100)", fontsize=10)
  ax2.set_ylabel("Абсолютна похибка від еталону (log scale)", fontsize=10)
  ax2.set_ylim(1e-17, 1e-12)
  ax2.legend(loc="upper right", frameon=True, fontsize=9)
  ax2.grid(True, which="both", ls="--", alpha=0.5)

  plt.tight_layout()
  plt.savefig("lab7_plots.png", dpi=300)
  print("[+] Графіки збережено у файл lab7_plots.png")
  plt.show()


if __name__ == "__main__":
  n = 100
  eps = 1e-14
  exact_val = 2.5

  # 1. Генерація системи та запис у файли
  generate_and_save_system(n=n, exact_val=exact_val)

  # 2. Зчитування з файлів
  A = read_matrix("matrix_A.txt")
  b = read_vector("vector_b.txt")

  # 3. Початкове наближення: x_i^(0) = 1.0 (i = 1..n)
  x0 = [1.0] * n

  print("\nПараметри задачі:")
  print(f" - Розмірність матриці: {n}x{n}")
  print(f" - Нескінченна норма матриці ||A||_inf: {matrix_norm_inf(A):.4f}")
  print(f" - Задана точність (eps_0): {eps}\n")

  # 4. Розв'язання системи трьома методами
  print("Виконання розрахунків...")
  x_sp, it_sp, hist_sp = solve_simple_iteration(A, b, x0, eps=eps)
  x_jc, it_jc, hist_jc = solve_jacobi(A, b, x0, eps=eps)
  x_sd, it_sd, hist_sd = solve_seidel(A, b, x0, eps=eps)

  # Функція розрахунку фінальної нев'язки ||A*x - b||_inf
  def compute_residual(A, b, x):
    Ax = matrix_vector_mult(A, x)
    return vector_norm_inf([Ax[i] - b[i] for i in range(len(b))])

  # Зведення числових результатів
  results = [
      (
          "Гауса–Зейделя",
          it_sd,
          compute_residual(A, b, x_sd),
          max(abs(v - exact_val) for v in x_sd),
      ),
      (
          "Якобі",
          it_jc,
          compute_residual(A, b, x_jc),
          max(abs(v - exact_val) for v in x_jc),
      ),
      (
          "Простої ітерації",
          it_sp,
          compute_residual(A, b, x_sp),
          max(abs(v - exact_val) for v in x_sp),
      ),
  ]

  # Заголовки винесено окремо, щоб уникнути конфліктів символів у f-рядках
  col_method = "Метод"
  col_iters = "Ітерацій"
  col_residual = "Нев'язка ||Ax - b||"
  col_error = "Похибка від 2.5"

  print("-" * 84)
  print(
      f"{col_method:<20} | {col_iters:<10} | {col_residual:<25} |"
      f" {col_error:<20}"
  )
  print("-" * 84)
  for name, it, res, err in results:
    print(f"{name:<20} | {it:<10} | {res:<25.2e} | {err:<20.2e}")
  print("-" * 84)

  # 5. Побудова графіків
  plot_results(
      n,
      eps,
      hist_sd,
      it_sd,
      hist_jc,
      it_jc,
      hist_sp,
      it_sp,
      x_sd,
      x_jc,
      exact_val,
  )