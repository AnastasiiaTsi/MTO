import matplotlib
matplotlib.use('Agg')  
import numpy as np
import matplotlib.pyplot as plt

# 1. Функції роботи з файлами та матричних операцій
def save_matrix(filename, matrix):
    """Збереження матриці у текстовий файл"""
    np.savetxt(filename, matrix, fmt='%.8e')

def load_matrix(filename):
    """Зчитування матриці з текстового файлу"""
    return np.loadtxt(filename)

def matrix_vector_mult(A, x):
    """Обчислення добутку матриці на вектор A * x"""
    return np.dot(A, x)

def vector_norm_inf(v):
    """Обчислення нескінченної норми вектора (максимум модуля)"""
    return np.max(np.abs(v))

# 2. Реалізація LU-розкладу (алгоритм Дулітла: U[i,i] = 1)
def lu_decomposition(A):
    """
    Знаходження LU-розкладу матриці A: A = L * U
    L - нижня трикутна, U - верхня трикутна з одиницями на діагоналі
    """
    n = A.shape[0]
    L = np.zeros((n, n), dtype=np.float64)
    U = np.eye(n, dtype=np.float64)
    
    for k in range(n):
        # Обчислення k-го стовпця матриці L (i >= k)
        for i in range(k, n):
            s1 = np.dot(L[i, :k], U[:k, k])
            L[i, k] = A[i, k] - s1
            
        # Обчислення k-го рядка матриці U (i > k)
        for i in range(k + 1, n):
            s2 = np.dot(L[k, :k], U[:k, i])
            U[k, i] = (A[k, i] - s2) / L[k, k]
            
    return L, U

def save_lu_to_file(filename, L, U):
    """Збереження LU-розкладу в текстовий файл у компактній формі"""
    n = L.shape[0]
    LU_packed = np.copy(L)
    # Записуємо позадіагональні елементи U у верхній трикутник
    for i in range(n):
        for j in range(i + 1, n):
            LU_packed[i, j] = U[i, j]
    save_matrix(filename, LU_packed)

# 3. Розв'язання СЛАР за допомогою LU-розкладу
def forward_substitution(L, B):
    """Прямий хід: розв'язання LZ = B"""
    n = L.shape[0]
    Z = np.zeros(n, dtype=np.float64)
    for k in range(n):
        s = np.dot(L[k, :k], Z[:k])
        Z[k] = (B[k] - s) / L[k, k]
    return Z

def backward_substitution(U, Z):
    """Зворотний хід: розв'язання UX = Z (діагональ U рівна 1)"""
    n = U.shape[0]
    X = np.zeros(n, dtype=np.float64)
    for k in range(n - 1, -1, -1):
        s = np.dot(U[k, k + 1:], X[k + 1:])
        X[k] = Z[k] - s
    return X

def lu_solve(L, U, B):
    """Повний розв'язок СЛАР: L * (U * X) = B"""
    Z = forward_substitution(L, B)
    X = backward_substitution(U, Z)
    return X

# 4. Генерація тестової СЛАР та запис у файли
np.random.seed(42)  # Фіксація випадковості для відтворюваності
n = 100

# Генерація матриці з діагональним домінуванням (стійкість факторизації)
A_init = np.random.uniform(-10.0, 10.0, size=(n, n))
for i in range(n):
    A_init[i, i] = np.sum(np.abs(A_init[i, :])) + np.random.uniform(5.0, 15.0)

# Заданий точний вектор невідомих: всі x_i = 2.5
X_exact = np.full(n, 2.5, dtype=np.float64)

# Вектор правих частин B = A * X
B_init = matrix_vector_mult(A_init, X_exact)

# Запис у файли
save_matrix("matrix_A.txt", A_init)
save_matrix("vector_B.txt", B_init)

# 5. Виконання лабораторного дослідження 
# Зчитування вхідних даних із файлів
A = load_matrix("matrix_A.txt")
B = load_matrix("vector_B.txt")

# Знаходження та збереження LU-розкладу
L, U = lu_decomposition(A)
save_lu_to_file("matrix_LU.txt", L, U)

# Початковий розв'язок СЛАР методом LU-розкладу (X^(0))
X_0 = lu_solve(L, U, B)

# Оцінка похибки початкового розв'язку (вектор нев'язки)
residual_0 = matrix_vector_mult(A, X_0) - B
eps_initial = vector_norm_inf(residual_0)
eps_true_0 = vector_norm_inf(X_0 - X_exact)

# Ітераційне уточнення розв'язку
eps_target = 1e-14
X_curr = np.copy(X_0)
iter_count = 0
max_iters = 20

residuals_history = [eps_initial]
errors_history = [eps_true_0]

while iter_count < max_iters:
    # 1. Вектор нев'язки: R = B - A * X
    R = B - matrix_vector_mult(A, X_curr)
    res_norm = vector_norm_inf(R)
    
    if res_norm <= eps_target:
        break
        
    # 2. Розв'язання A * delta_X = R через уже знайдений LU-розклад
    delta_X = lu_solve(L, U, R)
    
    # 3. Уточнення розв'язку
    X_curr = X_curr + delta_X
    iter_count += 1
    
    # Фіксація нев'язки та похибки
    current_res = vector_norm_inf(matrix_vector_mult(A, X_curr) - B)
    residuals_history.append(current_res)
    errors_history.append(vector_norm_inf(X_curr - X_exact))

# 6. Побудова графіків для звіту
# Графік 1: Збіжність ітераційного уточнення розв'язку
plt.figure(figsize=(8.5, 4.5))
iters_axis = np.arange(len(residuals_history))
plt.semilogy(iters_axis, residuals_history, 'ro-', lw=2, markersize=7, label=r'Норма нев’язки $\|AX^{(m)} - B\|_\infty$')
plt.semilogy(iters_axis, errors_history, 'bs--', lw=1.8, markersize=6, label=r'Похибка від еталону $\|X^{(m)} - X_{true}\|_\infty$')
plt.axhline(eps_target, color='green', linestyle=':', lw=1.5, label=r'Поріг збіжності $eps_0 = 10^{-14}$')
plt.title(f'Рис. 1. Збіжність ітераційного уточнення розв’язку СЛАР (n = {n})')
plt.xlabel('Номер ітерації (m)')
plt.ylabel('Величина похибки / нев’язки (log шкала)')
plt.grid(True, which="both", linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('chart1_convergence.png', dpi=300)
plt.close()

# Графік 2: Покомпонентне порівняння початкової та уточненої похибки
plt.figure(figsize=(10, 4.5))
components = np.arange(1, n + 1)
plt.plot(components, np.abs(X_0 - X_exact), color='crimson', lw=1.5, alpha=0.7, label=r'Початкова похибка $|x_i^{(0)} - 2.5|$')
plt.plot(components, np.abs(X_curr - X_exact), color='navy', lw=1.8, label=r'Уточнена похибка $|x_i^{(final)} - 2.5|$')
plt.title('Рис. 2. Розподіл похибки за компонентами вектора невідомих X')
plt.xlabel('Індекс компоненти (i = 1..100)')
plt.ylabel('Абсолютна похибка')
plt.yscale('log')
plt.grid(True, which="both", linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('chart2_error_profile.png', dpi=300)
plt.close()

# 7. Вивід числових протоколів
print("=" * 70)
print(f"ПРОТОКОЛ РОЗВ'ЯЗАННЯ СЛАР (Розмірність n = {n}):")
print("-" * 70)
print(f"Початкова норма нев'язки (max |AX^(0) - B|): {eps_initial:.4e}")
print(f"Початкова похибка від еталону max |X^(0) - 2.5|: {eps_true_0:.4e}")
print("-" * 70)
print(f"Кількість ітерацій для досягнення eps_0 = {eps_target}: {iter_count}")
print(f"Фінальна норма нев'язки: {residuals_history[-1]:.4e}")
print(f"Фінальна похибка від еталону: {errors_history[-1]:.4e}")
print("=" * 70)
print("Файли matrix_A.txt, vector_B.txt, matrix_LU.txt успішно сформовано.")
print("Графіки chart1_convergence.png та chart2_error_profile.png успішно збережено.")