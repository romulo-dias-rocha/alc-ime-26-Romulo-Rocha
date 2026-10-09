import numpy as np

def resolve_lu(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = A.shape[0]

    if A.shape[0] != A.shape[1]:
        raise ValueError("A matriz A deve ser quadrada.")

    L = np.eye(n, dtype=float)
    U = A.copy()

    for k in range(n):
        if U[k, k] == 0:
            raise Exception(
                f"Pivô nulo encontrado na posição ({k}, {k}). "
                "Utilize uma função alternativa para a solução do sistema."
            )

        for i in range(k + 1, n):
            m = U[i, k] / U[k, k]
            L[i, k] = m
            U[i, k:] = U[i, k:] - m * U[k, k:]

    y = np.zeros(n, dtype=float)
    for i in range(n):
        soma = sum(L[i, j] * y[j] for j in range(i))
        y[i] = b[i] - soma

    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        soma = sum(U[i, j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - soma) / U[i, i]

    return L, U, x


if __name__ == "__main__":
    A_teste = np.array([[2.0, 1.0, 1.0],
                        [4.0, 3.0, 3.0],
                        [8.0, 7.0, 9.0]])
    b_teste = np.array([5.0, 13.0, 31.0])

    L, U, x = resolve_lu(A_teste, b_teste)

    print("Matriz L:")
    print(L)
    print("\nMatriz U:")
    print(U)
    print("\nVetor Solução x:")
    print(x)