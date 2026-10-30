import os

import numpy as np

TOL_PIVOTE = 1e-12


def _validar_sistema(A, b=None):
    A = np.array(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A debe ser una matriz cuadrada.")
    if b is None:
        return A
    b = np.array(b, dtype=float).ravel()
    if b.shape[0] != A.shape[0]:
        raise ValueError("A y b tienen tamaños incompatibles.")
    return A, b


def sustitucion_adelante(L, y):
    n = len(y)
    x = np.zeros(n)
    for i in range(n):
        x[i] = (y[i] - L[i, :i] @ x[:i]) / L[i, i]
    return x


def sustitucion_atras(U, y):
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def eliminacion_gauss_pivoteo(A, b):
    A, b = _validar_sistema(A, b)
    n = len(b)
    M = np.hstack((A, b.reshape(-1, 1)))
    escala = np.max(np.abs(A))

    for i in range(n):
        p = i + int(np.argmax(np.abs(M[i:, i])))
        if abs(M[p, i]) <= TOL_PIVOTE * escala:
            raise ValueError("La matriz es singular o casi singular.")
        if p != i:
            M[[i, p]] = M[[p, i]]
        factores = M[i + 1:, i] / M[i, i]
        M[i + 1:, i:] -= np.outer(factores, M[i, i:])

    return sustitucion_atras(M[:, :n], M[:, n])


def lu_descomponer(A):
    A = _validar_sistema(A)
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)
    escala = np.max(np.abs(A))

    for i in range(n):
        p = i + int(np.argmax(np.abs(U[i:, i])))
        if abs(U[p, i]) <= TOL_PIVOTE * escala:
            raise ValueError("La matriz es singular o casi singular.")
        if p != i:
            U[[i, p]] = U[[p, i]]
            P[[i, p]] = P[[p, i]]
            L[[i, p], :i] = L[[p, i], :i]
        L[i + 1:, i] = U[i + 1:, i] / U[i, i]
        U[i + 1:, i:] -= np.outer(L[i + 1:, i], U[i, i:])
        U[i + 1:, i] = 0.0

    return P, L, U


def resolver_lu(P, L, U, b):
    b = np.array(b, dtype=float).ravel()
    y = sustitucion_adelante(L, P @ b)
    return sustitucion_atras(U, y)


def factorizacion_lu(A, b):
    A, b = _validar_sistema(A, b)
    P, L, U = lu_descomponer(A)
    return resolver_lu(P, L, U, b)


def norma_vector(v, p=2):
    v = np.abs(np.asarray(v, dtype=float).ravel())
    if v.size == 0:
        return 0.0
    if p == np.inf:
        return float(v.max())
    if p == 1:
        return float(v.sum())
    if p == 2:
        mayor = v.max()
        if mayor == 0:
            return 0.0
        w = v / mayor
        return float(mayor * np.sqrt(w @ w))
    if p > 1:
        return float(np.sum(v ** p) ** (1.0 / p))
    raise ValueError("p debe ser mayor o igual a 1.")


def _autovalor_dominante(M, tol=1e-14, max_iter=200000):
    n = M.shape[0]
    v = np.random.default_rng(0).uniform(0.5, 1.5, n)
    v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(max_iter):
        w = M @ v
        nw = np.sqrt(w @ w)
        if nw == 0:
            return 0.0
        v = w / nw
        nuevo = float(v @ (M @ v))
        if abs(nuevo - lam) <= tol * abs(nuevo):
            return nuevo
        lam = nuevo
    return lam


def norma_matriz(A, tipo=2):
    A = np.asarray(A, dtype=float)
    if tipo == 1:
        return float(np.max(np.sum(np.abs(A), axis=0)))
    if tipo == np.inf:
        return float(np.max(np.sum(np.abs(A), axis=1)))
    if tipo == "fro":
        return norma_vector(A.ravel(), 2)
    if tipo == 2:
        return float(np.sqrt(max(_autovalor_dominante(A.T @ A), 0.0)))
    raise ValueError("Tipo de norma no soportado.")


def inversa(A):
    A = _validar_sistema(A)
    n = A.shape[0]
    P, L, U = lu_descomponer(A)
    inv = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        inv[:, j] = resolver_lu(P, L, U, e)
    return inv


def numero_condicion(A, tipo=2):
    A = _validar_sistema(A)
    return norma_matriz(A, tipo) * norma_matriz(inversa(A), tipo)


def error_relativo(x_ref, x):
    x_ref = np.asarray(x_ref, dtype=float)
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        return float("inf")
    den = norma_vector(x_ref, 2)
    num = norma_vector(x_ref - x, 2)
    return num / den if den > 0 else num


def calcular_residuo(A, b, x):
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        return float("inf")
    return norma_vector(np.asarray(b, dtype=float) - np.asarray(A, dtype=float) @ x, 2)


def _diagonal_valida(A):
    d = np.diag(A).copy()
    if np.any(np.abs(d) < TOL_PIVOTE):
        raise ValueError("Elemento diagonal demasiado pequeño.")
    return d


def jacobi(A, b, tol=1e-8, max_iter=5000, historial=None, x0=None):
    A, b = _validar_sistema(A, b)
    d = _diagonal_valida(A)
    R = A - np.diagflat(d)
    x = np.zeros(len(b)) if x0 is None else np.array(x0, dtype=float)
    limite = 1e12 * max(1.0, norma_vector(b, np.inf))

    for k in range(max_iter):
        x_nuevo = (b - R @ x) / d
        error = norma_vector(x_nuevo - x, np.inf)
        if historial is not None:
            historial.append(error)
        if not np.isfinite(error) or error > limite:
            return x, k + 1, False
        if error < tol:
            return x_nuevo, k + 1, True
        x = x_nuevo

    return x, max_iter, False


def gauss_seidel(A, b, tol=1e-8, max_iter=5000, historial=None, x0=None):
    A, b = _validar_sistema(A, b)
    d = _diagonal_valida(A)
    n = len(b)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    limite = 1e12 * max(1.0, norma_vector(b, np.inf))

    for k in range(max_iter):
        x_ant = x.copy()
        for i in range(n):
            suma = A[i, :i] @ x[:i] + A[i, i + 1:] @ x[i + 1:]
            x[i] = (b[i] - suma) / d[i]
        error = norma_vector(x - x_ant, np.inf)
        if historial is not None:
            historial.append(error)
        if not np.isfinite(error) or error > limite:
            return x_ant, k + 1, False
        if error < tol:
            return x, k + 1, True

    return x, max_iter, False


if __name__ == "__main__":
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    A = np.load(os.path.join(base, "dataset", "A_normal.npy"))
    b = np.load(os.path.join(base, "dataset", "b_normal.npy"))
    ref = np.linalg.solve(A, b)

    resultados = [
        ("Gauss", eliminacion_gauss_pivoteo(A, b), None, True),
        ("LU", factorizacion_lu(A, b), None, True),
    ]
    x, it, ok = jacobi(A, b)
    resultados.append(("Jacobi", x, it, ok))
    x, it, ok = gauss_seidel(A, b, max_iter=250000)
    resultados.append(("Gauss-Seidel", x, it, ok))

    for nombre, x, it, ok in resultados:
        estado = "convergió" if ok else "no convergió"
        iters = "" if it is None else f" en {it} iteraciones"
        print(
            f"{nombre}: {estado}{iters} | "
            f"residuo {calcular_residuo(A, b, x):.2e} | "
            f"error vs NumPy {error_relativo(ref, x):.2e}"
        )