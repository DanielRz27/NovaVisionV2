import csv
import json
import os
import time

import numpy as np

from entrenamiento import (
    BASE_DIR,
    DATASET_DIR,
    LAMBDA_DEFECTO,
    cargar_entrenamiento,
    cargar_imagenes,
    clasificar,
    entrenar,
    metricas_clasificacion,
    predecir_valor,
)
from image_processing import NOMBRES_BASE
from numerical_methods import (
    calcular_residuo,
    eliminacion_gauss_pivoteo,
    error_relativo,
    factorizacion_lu,
    gauss_seidel,
    jacobi,
    lu_descomponer,
    numero_condicion,
    resolver_lu,
)

DIR_RESULTADOS = os.path.join(BASE_DIR, "docs", "resultados")
DIR_FIGURAS = os.path.join(BASE_DIR, "docs", "figuras")
LAMBDAS = (0.001, 0.01, 0.1, 1, 10, 100, 1000)
AZUL = "#1F3864"
AZUL_CLARO = "#8FAADC"
GRIS = "#7F7F7F"
ROJO = "#C0504D"


def sistemas_conocidos():
    sistemas = []

    A1 = np.array([[4.0, 1.0, 1.0], [1.0, 5.0, 2.0], [1.0, 2.0, 6.0]])
    sistemas.append(("S1 diagonal dominante 3x3", A1, np.array([1.0, 2.0, 3.0]), True))

    A2 = np.array([[0.0, 2.0, 1.0], [1.0, 1.0, 1.0], [2.0, 1.0, 3.0]])
    sistemas.append(("S2 requiere pivoteo 3x3", A2, np.array([2.0, -1.0, 3.0]), False))

    rng = np.random.default_rng(2)
    B = rng.uniform(-1, 1, (10, 10))
    A3 = (B + B.T) / 2
    np.fill_diagonal(A3, 0.0)
    A3 = A3 + np.diag(np.abs(A3).sum(axis=1) + 1.0)
    sistemas.append(("S3 SPD dominante 10x10", A3, np.arange(1.0, 11.0), True))

    idx = np.arange(8)
    A4 = 1.0 / (idx[:, None] + idx[None, :] + 1.0)
    sistemas.append(("S4 Hilbert 8x8 (mal condicionada)", A4, np.ones(8), False))

    return [
        {"nombre": n, "A": A, "x": x, "b": A @ x, "iterativos": it}
        for n, A, x, it in sistemas
    ]


def radio_espectral(M):
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def matrices_iteracion(A):
    d = np.diag(A)
    TJ = -(A - np.diag(d)) / d[:, None]
    TGS = -np.linalg.solve(np.tril(A), np.triu(A, 1))
    return TJ, TGS


def resolver_con_historial(f, A, b, **kw):
    historial = []
    t0 = time.perf_counter()
    x, iteraciones, convergio = f(A, b, historial=historial, **kw)
    return {
        "x": x,
        "iteraciones": iteraciones,
        "convergio": convergio,
        "tiempo": time.perf_counter() - t0,
        "historial": historial,
    }


def nivel_1():
    filas = []
    for s in sistemas_conocidos():
        c1 = numero_condicion(s["A"], 1)

        def fila(metodo, x, iteraciones, convergio):
            return {
                "sistema": s["nombre"],
                "metodo": metodo,
                "cond_1": c1,
                "error_relativo": error_relativo(s["x"], x),
                "residuo": calcular_residuo(s["A"], s["b"], x),
                "iteraciones": iteraciones,
                "convergio": convergio,
            }

        filas.append(fila("Gauss", eliminacion_gauss_pivoteo(s["A"], s["b"]), None, True))
        filas.append(fila("LU", factorizacion_lu(s["A"], s["b"]), None, True))
        if s["iterativos"]:
            for nombre, f in (("Jacobi", jacobi), ("Gauss-Seidel", gauss_seidel)):
                x, it, ok = f(s["A"], s["b"], tol=1e-12, max_iter=50000)
                filas.append(fila(nombre, x, it, ok))
    return filas


def comparar_metodos(A, b, tope_corto=5000, tope_largo=250000):
    ref = np.linalg.solve(A, b)

    def fila(metodo, x, iteraciones, convergio, tiempo, **extra):
        return {
            "metodo": metodo,
            "iteraciones": iteraciones,
            "convergio": convergio,
            "tiempo_s": tiempo,
            "error_relativo_vs_numpy": error_relativo(ref, x),
            "residuo": calcular_residuo(A, b, x),
            **extra,
        }

    filas = []

    t0 = time.perf_counter()
    x = eliminacion_gauss_pivoteo(A, b)
    filas.append(fila("Gauss", x, 0, True, time.perf_counter() - t0))

    t0 = time.perf_counter()
    P, L, U = lu_descomponer(A)
    t1 = time.perf_counter()
    x = resolver_lu(P, L, U, b)
    t2 = time.perf_counter()
    filas.append(
        fila(
            "LU", x, 0, True, t2 - t0,
            tiempo_factorizacion_s=t1 - t0,
            tiempo_solo_resolver_s=t2 - t1,
        )
    )

    for nombre, f, tope in (
        (f"Jacobi (máx. {tope_corto})", jacobi, tope_corto),
        (f"Gauss-Seidel (máx. {tope_corto})", gauss_seidel, tope_corto),
        (f"Gauss-Seidel (máx. {tope_largo})", gauss_seidel, tope_largo),
    ):
        r = resolver_con_historial(f, A, b, max_iter=tope)
        filas.append(fila(nombre, r["x"], r["iteraciones"], r["convergio"], r["tiempo"]))

    return filas


def colinealidad(X, nombres):
    Xc = X - X.mean(axis=0)
    vif = {}
    for j in range(X.shape[1]):
        otras = np.delete(Xc, j, axis=1)
        y = Xc[:, j]
        coef = np.linalg.lstsq(otras, y, rcond=None)[0]
        ss_res = np.sum((y - otras @ coef) ** 2)
        ss_tot = np.sum(y ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vif[nombres[j]] = float(min(1.0 / max(1 - r2, 1e-16), 1e16))

    resultado = {
        "rango": int(np.linalg.matrix_rank(Xc)),
        "n": int(X.shape[1]),
        "vif": vif,
    }
    if all(c in nombres for c in ("brillo_promedio", "intensidad_centro", "intensidad_fondo")):
        i_b, i_c, i_f = (
            nombres.index("brillo_promedio"),
            nombres.index("intensidad_centro"),
            nombres.index("intensidad_fondo"),
        )
        resultado["dif_brillo"] = float(
            np.max(np.abs(X[:, i_b] - (0.25 * X[:, i_c] + 0.75 * X[:, i_f])))
        )
    return resultado


def validar_cruzada(X, y, lambda_reg, k=5, semilla=0):
    orden = np.random.default_rng(semilla).permutation(len(y))
    aciertos = []
    for fold in np.array_split(orden, k):
        entrenamiento = np.setdiff1d(orden, fold)
        modelo = entrenar(X[entrenamiento], y[entrenamiento], lambda_reg)
        pred = clasificar(predecir_valor(X[fold], modelo))
        aciertos.append(np.mean(pred == y[fold]) * 100)
    return float(np.mean(aciertos))


def barrido_lambda(X_tr, y_tr, X_te, y_te, lambdas=LAMBDAS, tope=60000):
    filas = []
    for lam in lambdas:
        modelo = entrenar(X_tr, y_tr, lam)
        A, b = modelo["A"], modelo["b"]
        TJ, TGS = matrices_iteracion(A)
        _, it, ok = gauss_seidel(A, b, max_iter=tope)
        pred = clasificar(predecir_valor(X_te, modelo))
        filas.append(
            {
                "lambda": lam,
                "cond2": numero_condicion(A, 2),
                "rho_jacobi": radio_espectral(TJ),
                "rho_gs": radio_espectral(TGS),
                "iter_gs": it,
                "gs_convergio": ok,
                "acc_cv": validar_cruzada(X_tr, y_tr, lam),
                "acc_test": float(np.mean(pred == y_te) * 100),
            }
        )
    return filas


def evaluar_clasificador(modelo, X_tr, y_tr, X_te, y_te, rutas_te):
    y_hat_tr = predecir_valor(X_tr, modelo)
    y_hat_te = predecir_valor(X_te, modelo)
    pred_tr = clasificar(y_hat_tr)
    pred_te = clasificar(y_hat_te)

    ss_res = np.sum((y_tr - y_hat_tr) ** 2)
    ss_tot = np.sum((y_tr - y_tr.mean()) ** 2)

    errores = []
    for i in np.where(pred_te != y_te)[0]:
        errores.append(
            {
                "imagen": os.path.relpath(rutas_te[i], DATASET_DIR).replace(os.sep, "/"),
                "real": "perro" if y_te[i] == 1 else "gato",
                "predicho": "perro" if pred_te[i] == 1 else "gato",
                "y_hat": round(float(y_hat_te[i]), 4),
                "tipo": "FP" if pred_te[i] == 1 else "FN",
            }
        )
    errores.sort(key=lambda e: (e["tipo"] != "FP", -abs(e["y_hat"] - 0.5)))

    return {
        "metricas_test": metricas_clasificacion(y_te, pred_te),
        "metricas_train": metricas_clasificacion(y_tr, pred_tr),
        "r2_train": float(1 - ss_res / ss_tot),
        "fuera_de_01_test": int(np.sum((y_hat_te < 0) | (y_hat_te > 1))),
        "errores": errores,
        "y_hat_train": y_hat_tr,
    }


def comparar_modo_ampliado(Xa_tr, y_tr, Xa_te, y_te, lambda_reg):
    filas = []
    for etiqueta, n in (("base (12)", 12), ("ampliado (28)", Xa_tr.shape[1])):
        modelo = entrenar(Xa_tr[:, :n], y_tr, lambda_reg)
        acc_tr = np.mean(clasificar(predecir_valor(Xa_tr[:, :n], modelo)) == y_tr) * 100
        acc_te = np.mean(clasificar(predecir_valor(Xa_te[:, :n], modelo)) == y_te) * 100
        filas.append(
            [
                etiqueta,
                n,
                f"{acc_tr:.2f}",
                f"{acc_te:.2f}",
                f"{numero_condicion(modelo['A'], 2):.2e}",
            ]
        )
    return filas


def _limpiar(obj):
    if isinstance(obj, dict):
        return {k: _limpiar(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_limpiar(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _limpiar(obj.tolist())
    if isinstance(obj, np.generic):
        return _limpiar(obj.item())
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _formato(v):
    if isinstance(v, (float, np.floating)):
        return "inf" if not np.isfinite(v) else f"{v:.4g}"
    return "" if v is None else v


def guardar_csv(nombre, filas, columnas):
    os.makedirs(DIR_RESULTADOS, exist_ok=True)
    ruta = os.path.join(DIR_RESULTADOS, nombre)
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(columnas)
        for fila in filas:
            if isinstance(fila, dict):
                fila = [fila.get(c) for c in columnas]
            w.writerow([_formato(v) for v in fila])


def _guardar_figura(fig, nombre):
    os.makedirs(DIR_FIGURAS, exist_ok=True)
    fig.savefig(os.path.join(DIR_FIGURAS, nombre), dpi=160, bbox_inches="tight")


def graficar(modelo, metodos, barrido, evaluacion, y_tr, X_tr):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    fig, ax = plt.subplots(figsize=(8, 3.6))
    nombres = [m["metodo"] for m in metodos][::-1]
    tiempos = [max(m["tiempo_s"], 1e-7) for m in metodos][::-1]
    colores = [AZUL if m["convergio"] else GRIS for m in metodos][::-1]
    ax.barh(nombres, tiempos, color=colores)
    ax.set_xscale("log")
    ax.set_xlabel("Tiempo de cálculo (s, escala logarítmica)")
    for i, m in enumerate(metodos[::-1]):
        ax.text(tiempos[i] * 1.15, i, f"{m['iteraciones']} it.", va="center", fontsize=8)
    _guardar_figura(fig, "fig_comparacion_metodos.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for lam, color in zip((1, 10, 100, 1000), (AZUL, "#2E75B6", AZUL_CLARO, "#BDD7EE")):
        m = entrenar(X_tr, y_tr, lam)
        r = resolver_con_historial(gauss_seidel, m["A"], m["b"], max_iter=60000)
        ax.semilogy(range(1, len(r["historial"]) + 1), r["historial"], color=color, label=f"Gauss-Seidel, λ={lam}")
    m = entrenar(X_tr, y_tr, 1000)
    r = resolver_con_historial(jacobi, m["A"], m["b"], max_iter=500)
    ax.semilogy(range(1, len(r["historial"]) + 1), r["historial"], color=ROJO, linestyle="--", label="Jacobi, λ=1000")
    ax.set_xscale("log")
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Cambio entre iteraciones (norma infinito)")
    ax.legend(frameon=False)
    _guardar_figura(fig, "fig_convergencia.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    lam = [f["lambda"] for f in barrido]
    ax.loglog(lam, [f["cond2"] for f in barrido], color=AZUL, marker="o")
    ax.set_xlabel("λ (regularización Ridge)")
    ax.set_ylabel("Condicionamiento de A (norma 2)", color=AZUL)
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.semilogx(lam, [f["acc_cv"] for f in barrido], color=ROJO, marker="s", label="Validación cruzada")
    ax2.semilogx(lam, [f["acc_test"] for f in barrido], color=GRIS, marker="^", label="Prueba")
    ax2.set_ylabel("Exactitud (%)")
    ax2.legend(frameon=False, loc="lower left")
    _guardar_figura(fig, "fig_lambda.png")
    plt.close(fig)

    mt = evaluacion["metricas_test"]
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    matriz = np.array([[mt["VN"], mt["FP"]], [mt["FN"], mt["VP"]]])
    ax.imshow(matriz, cmap="Blues")
    ax.set_xticks([0, 1], ["Pred. gato", "Pred. perro"])
    ax.set_yticks([0, 1], ["Real gato", "Real perro"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, matriz[i, j], ha="center", va="center",
                    color="white" if matriz[i, j] > matriz.max() / 2 else "black", fontsize=14)
    _guardar_figura(fig, "fig_matriz_confusion.png")
    plt.close(fig)

    residuos = y_tr - evaluacion["y_hat_train"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(residuos[y_tr == 0], bins=30, alpha=0.75, color=AZUL, label="Gatos")
    ax.hist(residuos[y_tr == 1], bins=30, alpha=0.75, color=ROJO, label="Perros")
    ax.set_xlabel("Residuo (y − ŷ)")
    ax.set_ylabel("Imágenes")
    ax.legend(frameon=False)
    _guardar_figura(fig, "fig_residuos.png")
    plt.close(fig)

    cal = modelo["calibracion"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], color=GRIS, linestyle="--", label="Calibración ideal")
    ax.plot(cal["centros"], cal["prob"], color=AZUL, marker="o", label="Modelo")
    ax.set_xlabel("ŷ promedio del tramo")
    ax.set_ylabel("Fracción de perros")
    ax.legend(frameon=False)
    _guardar_figura(fig, "fig_calibracion.png")
    plt.close(fig)


def main():
    print("Cargando datos...")
    X_tr, y_tr = cargar_entrenamiento()
    _, y_te, rutas_te = cargar_imagenes("test", "base")
    X_te, _, _ = cargar_imagenes("test", "base")

    modelo = entrenar(X_tr, y_tr, LAMBDA_DEFECTO)
    A, b = modelo["A"], modelo["b"]

    print("Sistemas de solución conocida...")
    guardar_csv(
        "tabla_sistemas_conocidos.csv",
        nivel_1(),
        ["sistema", "metodo", "cond_1", "error_relativo", "residuo", "iteraciones", "convergio"],
    )

    print("Comparación de métodos...")
    metodos = comparar_metodos(A, b)
    guardar_csv(
        "tabla_metodos.csv",
        metodos,
        ["metodo", "iteraciones", "convergio", "tiempo_s", "error_relativo_vs_numpy", "residuo"],
    )

    print("Colinealidad...")
    colin = colinealidad(X_tr, NOMBRES_BASE)
    guardar_csv("tabla_vif.csv", [[k, v] for k, v in colin["vif"].items()], ["caracteristica", "vif"])

    print("Barrido de lambda...")
    barrido = barrido_lambda(X_tr, y_tr, X_te, y_te)
    guardar_csv(
        "tabla_lambda.csv",
        [
            [f["lambda"], f["cond2"], f["rho_jacobi"], f["rho_gs"], f["iter_gs"], f["gs_convergio"], f["acc_cv"], f["acc_test"]]
            for f in barrido
        ],
        ["lambda", "cond_2", "rho_jacobi", "rho_gauss_seidel", "iter_gauss_seidel", "gs_convergio", "exactitud_cv", "exactitud_test"],
    )

    print("Clasificador...")
    ev = evaluar_clasificador(modelo, X_tr, y_tr, X_te, y_te, rutas_te)
    guardar_csv("tabla_metricas.csv", list(ev["metricas_test"].items()), ["metrica", "valor"])
    guardar_csv("errores_clasificacion.csv", ev["errores"], ["imagen", "real", "predicho", "y_hat", "tipo"])

    print("Características ampliadas...")
    Xa_tr, _, _ = cargar_imagenes("train", "ampliado")
    Xa_te, _, _ = cargar_imagenes("test", "ampliado")
    comparacion = comparar_modo_ampliado(Xa_tr, y_tr, Xa_te, y_te, LAMBDA_DEFECTO)
    guardar_csv(
        "tabla_caracteristicas.csv",
        comparacion,
        ["conjunto", "n_caracteristicas", "exactitud_train", "exactitud_test", "cond_A"],
    )

    TJ, TGS = matrices_iteracion(A)
    resumen = {
        "lambda": LAMBDA_DEFECTO,
        "n_train": len(y_tr),
        "n_test": len(y_te),
        "radio_espectral_jacobi": radio_espectral(TJ),
        "radio_espectral_gauss_seidel": radio_espectral(TGS),
        "cond_2_A": numero_condicion(A, 2),
        "cond_1_A": numero_condicion(A, 1),
        "cond_inf_A": numero_condicion(A, np.inf),
        "colinealidad": colin,
        "metodos": metodos,
        "metricas_test": ev["metricas_test"],
        "metricas_train": ev["metricas_train"],
        "r2_train": ev["r2_train"],
        "valores_fuera_de_01_test": ev["fuera_de_01_test"],
        "barrido_lambda": barrido,
        "caracteristicas_base_vs_ampliado": comparacion,
    }
    with open(os.path.join(DIR_RESULTADOS, "resumen.json"), "w", encoding="utf-8") as f:
        json.dump(_limpiar(resumen), f, ensure_ascii=False, indent=2)

    print("Figuras...")
    graficar(modelo, metodos, barrido, ev, y_tr, X_tr)

    mt = ev["metricas_test"]
    print(
        f"\nExactitud en prueba: {mt['exactitud']:.2f} % "
        f"(VP {mt['VP']}, VN {mt['VN']}, FP {mt['FP']}, FN {mt['FN']})"
    )


if __name__ == "__main__":
    main()