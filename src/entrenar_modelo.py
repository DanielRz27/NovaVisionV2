import argparse
import json
import os
import time

import numpy as np

from entrenamiento import (
    BASE_DIR,
    MODO_DEFECTO,
    RUTA_MODELO,
    ajustar_calibracion,
    cargar_imagenes,
    clasificar,
    entrenar,
    guardar_modelo,
    metricas_clasificacion,
    predecir_valor,
    validacion_cruzada,
)
from numerical_methods import (
    error_relativo,
    factorizacion_lu,
    gauss_seidel,
)

LAMBDAS_DEFECTO = (300, 1000, 3000, 10000, 30000)


def main():
    parser = argparse.ArgumentParser(description="Entrena el clasificador NovaVision.")
    parser.add_argument("--modo", default=MODO_DEFECTO)
    parser.add_argument("--lambdas", type=float, nargs="+", default=list(LAMBDAS_DEFECTO))
    parser.add_argument("--pliegues", type=int, default=5)
    args = parser.parse_args()

    t0 = time.perf_counter()
    print(f"Extrayendo características (modo {args.modo})...")
    X_tr, y_tr, _ = cargar_imagenes("train", args.modo)
    X_te, y_te, _ = cargar_imagenes("test", args.modo)
    print(f"  entrenamiento: {X_tr.shape} | prueba: {X_te.shape}")

    print(f"Eligiendo λ con validación cruzada de {args.pliegues} pliegues (solo entrenamiento)...")
    cv = {}
    for lam in args.lambdas:
        r = validacion_cruzada(X_tr, y_tr, [lam], k=args.pliegues)[lam]
        cv[lam] = r
        print(f"  λ = {lam:>8g}  exactitud CV = {r['exactitud']:.2f} %")
    lam_opt = max(cv, key=lambda l: cv[l]["exactitud"])
    print(f"λ elegido: {lam_opt:g}")

    print("Entrenando el modelo final con Gauss...")
    modelo = entrenar(X_tr, y_tr, lam_opt, "gauss", args.modo)
    modelo["calibracion"] = ajustar_calibracion(cv[lam_opt]["y_hat"], y_tr)

    A, b = modelo["A"], modelo["b"]
    beta_lu = factorizacion_lu(A, b)
    t1 = time.perf_counter()
    beta_gs, it_gs, ok_gs = gauss_seidel(A, b, tol=1e-10, max_iter=20000)
    t_gs = time.perf_counter() - t1
    comparacion = {
        "error_lu_vs_gauss": error_relativo(modelo["beta"], beta_lu),
        "error_gs_vs_gauss": error_relativo(modelo["beta"], beta_gs),
        "iteraciones_gs": it_gs,
        "gs_convergio": ok_gs,
        "tiempo_gs_s": t_gs,
        "cond_2_A": float(np.linalg.cond(A)),
    }

    pred_tr = clasificar(predecir_valor(X_tr, modelo))
    pred_te = clasificar(predecir_valor(X_te, modelo))
    m_tr = metricas_clasificacion(y_tr, pred_tr)
    m_te = metricas_clasificacion(y_te, pred_te)

    info = {
        "modo": args.modo,
        "n_caracteristicas": int(X_tr.shape[1]),
        "lambda": lam_opt,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "exactitud_cv": cv[lam_opt]["exactitud"],
        "exactitud_train": m_tr["exactitud"],
        "exactitud_test": m_te["exactitud"],
        "cv_por_lambda": {str(l): cv[l]["exactitud"] for l in cv},
        "metricas_test": m_te,
        "comparacion_solvers": comparacion,
    }
    guardar_modelo(modelo, info)

    destino = os.path.join(BASE_DIR, "docs", "resultados")
    os.makedirs(destino, exist_ok=True)
    with open(os.path.join(destino, "modelo_final.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print("\nResultados")
    print(f"  Exactitud validación cruzada : {info['exactitud_cv']:.2f} %")
    print(f"  Exactitud entrenamiento      : {m_tr['exactitud']:.2f} %")
    print(f"  Exactitud prueba (una sola vez): {m_te['exactitud']:.2f} %")
    print(f"  VP {m_te['VP']}  VN {m_te['VN']}  FP {m_te['FP']}  FN {m_te['FN']}")
    print(
        f"  Gauss-Seidel: {it_gs} iteraciones, "
        f"error vs Gauss {comparacion['error_gs_vs_gauss']:.2e}"
    )
    print(f"  Modelo guardado en {os.path.relpath(RUTA_MODELO, BASE_DIR)}")
    print(f"  Tiempo total: {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
