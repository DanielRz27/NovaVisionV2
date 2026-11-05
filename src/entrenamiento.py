"""
Núcleo común de NovaVision.

Antes, el armado de las ecuaciones normales (XᵀX + λI)β = Xᵀy estaba
copiado en prepare_system.py, model.py y api/main.py. Aquí vive una sola
versión, que usan los tres y también validation.py.

Flujo:  X (m × n)  →  z-score  →  matriz de diseño (m × n+1)
        →  ecuaciones normales + Ridge  →  β con un método propio
"""

import json
import os

import numpy as np

from image_processing import (
    NOMBRES_POR_MODO,
    caracteristicas_desde_imagen,
    extraer_caracteristicas,
)
from numerical_methods import (
    eliminacion_gauss_pivoteo,
    factorizacion_lu,
)

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATASET_DIR = os.path.join(BASE_DIR, "dataset")

LAMBDA_DEFECTO = 0.01
MODO_DEFECTO = "avanzado"
DIR_MODELO = os.path.join(BASE_DIR, "modelo")
RUTA_MODELO = os.path.join(DIR_MODELO, "novavision_modelo.npz")
UMBRAL_INCIERTO = 60.0

UMBRAL = 0.5

CLASES = {"Gatos": 0, "Perros": 1}

EXTENSIONES_VALIDAS = (".jpg", ".jpeg", ".png")

# Métodos directos disponibles para resolver las ecuaciones normales
METODOS_DIRECTOS = {
    "gauss": eliminacion_gauss_pivoteo,
    "lu": factorizacion_lu,
}


# ------------------------------------------------------------
# DATOS
# ------------------------------------------------------------

def cargar_entrenamiento(dataset_dir=DATASET_DIR):
    """Carga X_train.npy y y_train.npy."""

    X = np.load(os.path.join(dataset_dir, "X_train.npy"))

    y = np.load(os.path.join(dataset_dir, "y_train.npy"))

    return X, y


def cargar_imagenes(split, modo="base", dataset_dir=DATASET_DIR):
    """
    Extrae las características de todas las imágenes de
    dataset/<split>/{Gatos,Perros}.

    Devuelve X, y y la lista de rutas (útil para el análisis de errores).
    """

    X, y, rutas = [], [], []

    for clase, etiqueta in CLASES.items():

        carpeta = os.path.join(dataset_dir, split, clase)

        if not os.path.exists(carpeta):
            continue

        for nombre in sorted(os.listdir(carpeta)):

            if not nombre.lower().endswith(EXTENSIONES_VALIDAS):
                continue

            ruta = os.path.join(carpeta, nombre)

            try:
                X.append(extraer_caracteristicas(ruta, modo=modo))
                y.append(etiqueta)
                rutas.append(ruta)

            except Exception as e:
                print(f"❌ Error procesando {nombre}: {e}")

    return np.array(X), np.array(y, dtype=float), rutas


# ------------------------------------------------------------
# NORMALIZACIÓN Y SISTEMA DE ECUACIONES
# ------------------------------------------------------------

def calcular_normalizacion(X):
    """Media y desviación por característica (z-score)."""

    media = np.mean(X, axis=0)

    desviacion = np.std(X, axis=0)

    desviacion[desviacion < 1e-12] = 1.0

    return media, desviacion


def normalizar(X, media, desviacion):

    return (X - media) / desviacion


def matriz_diseno(X_norm):
    """Agrega la columna de unos (término independiente β₀)."""

    X_norm = np.atleast_2d(X_norm)

    return np.hstack((np.ones((X_norm.shape[0], 1)), X_norm))


def sistema_normal(X_norm, y, lambda_reg=LAMBDA_DEFECTO):
    """
    Devuelve (A, b) de (XᵀX + λI)β = Xᵀy.
    β₀ no se regulariza.
    """

    D = matriz_diseno(X_norm)

    A = D.T @ D

    b = D.T @ y

    n = A.shape[0]

    I = np.eye(n)

    I[0, 0] = 0

    return A + lambda_reg * I, b


# ------------------------------------------------------------
# ENTRENAMIENTO Y PREDICCIÓN
# ------------------------------------------------------------

def entrenar(X, y, lambda_reg=LAMBDA_DEFECTO, metodo="gauss", modo=None):
    """
    Ajusta el modelo lineal.

    metodo: 'gauss' o 'lu' (ambos implementados por el grupo).
    Devuelve un diccionario con β, la normalización y el sistema (A, b).
    """

    if metodo not in METODOS_DIRECTOS:
        raise ValueError(
            f"Método no disponible: {metodo}. "
            f"Usa uno de {list(METODOS_DIRECTOS)}."
        )

    media, desviacion = calcular_normalizacion(X)

    X_norm = normalizar(X, media, desviacion)

    A, b = sistema_normal(X_norm, y, lambda_reg)

    beta = METODOS_DIRECTOS[metodo](A, b)

    modelo = {
        "beta": beta,
        "media": media,
        "desviacion": desviacion,
        "A": A,
        "b": b,
        "lambda": lambda_reg,
        "metodo": metodo,
        "modo": modo,
    }

    # Curva de calibración empírica (ver calibrar_probabilidad)
    y_hat_train = predecir_valor(X, modelo)

    modelo["calibracion"] = ajustar_calibracion(y_hat_train, y)

    return modelo


def predecir_valor(X, modelo):
    """ŷ = xβ para una o varias imágenes (características sin normalizar)."""

    X_norm = normalizar(np.atleast_2d(X), modelo["media"], modelo["desviacion"])

    return matriz_diseno(X_norm) @ modelo["beta"]


def clasificar(y_hat, umbral=UMBRAL):
    """1 (perro) si ŷ ≥ umbral; 0 (gato) en caso contrario."""

    return (np.asarray(y_hat) >= umbral).astype(int)


# ------------------------------------------------------------
# CALIBRACIÓN: de ŷ a una probabilidad estimada
# ------------------------------------------------------------
# ŷ no es una probabilidad (puede salir de [0, 1]). Con las imágenes de
# entrenamiento se agrupa ŷ en tramos y se mide qué fracción de cada tramo
# eran perros. Esa fracción es una estimación empírica de P(perro | ŷ),
# y es lo que justifica (o no) interpretar ŷ como probabilidad.

def ajustar_calibracion(y_hat, y, n_tramos=10):

    orden = np.argsort(y_hat)

    tramos = np.array_split(orden, n_tramos)

    centros = np.array([np.mean(y_hat[t]) for t in tramos])

    prob = np.array([np.mean(y[t]) for t in tramos])

    # Forzamos que la curva no decrezca (más ŷ ⇒ no menos probabilidad de perro)
    prob = np.maximum.accumulate(prob)

    return {"centros": centros, "prob": prob}


def probabilidad_perro(y_hat, calibracion):
    """P(perro) estimada por interpolación sobre la curva de calibración."""

    return np.interp(
        y_hat,
        calibracion["centros"],
        calibracion["prob"],
    )


def confianza_prediccion(y_hat, calibracion):
    """
    Confianza (en %) en la clase predicha, usando la probabilidad
    calibrada. Siempre está entre 50 % y 100 %.
    """

    p = float(probabilidad_perro(y_hat, calibracion))

    p_clase = p if y_hat >= UMBRAL else 1.0 - p

    return max(p_clase, 0.5) * 100.0


# ------------------------------------------------------------
# MÉTRICAS
# ------------------------------------------------------------

def metricas_clasificacion(y_real, y_pred):

    y_real = np.asarray(y_real).astype(int)

    y_pred = np.asarray(y_pred).astype(int)

    VP = int(np.sum((y_pred == 1) & (y_real == 1)))

    VN = int(np.sum((y_pred == 0) & (y_real == 0)))

    FP = int(np.sum((y_pred == 1) & (y_real == 0)))

    FN = int(np.sum((y_pred == 0) & (y_real == 1)))

    total = len(y_real)

    def pct(num, den):
        return num / den * 100 if den > 0 else 0.0

    return {
        "VP": VP,
        "VN": VN,
        "FP": FP,
        "FN": FN,
        "total": total,
        "exactitud": pct(VP + VN, total),
        "precision": pct(VP, VP + FP),
        "sensibilidad": pct(VP, VP + FN),
        "especificidad": pct(VN, VN + FP),
    }


def validacion_cruzada(X, y, lambdas, k=5, semilla=0, metodo="gauss"):
    orden = np.random.default_rng(semilla).permutation(len(y))
    pliegues = np.array_split(orden, k)
    resultados = {}
    for lam in lambdas:
        y_hat = np.zeros(len(y))
        for pliegue in pliegues:
            fuera = np.setdiff1d(orden, pliegue)
            modelo = entrenar(X[fuera], y[fuera], lam, metodo)
            y_hat[pliegue] = predecir_valor(X[pliegue], modelo)
        exactitud = float(np.mean(clasificar(y_hat) == y) * 100)
        resultados[lam] = {"exactitud": exactitud, "y_hat": y_hat}
    return resultados


def guardar_modelo(modelo, info=None, ruta=RUTA_MODELO):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    np.savez(
        ruta,
        beta=modelo["beta"],
        media=modelo["media"],
        desviacion=modelo["desviacion"],
        centros=modelo["calibracion"]["centros"],
        prob=modelo["calibracion"]["prob"],
        lambda_reg=float(modelo["lambda"]),
        metodo=str(modelo["metodo"]),
        modo=str(modelo["modo"]),
        info=json.dumps(info or {}, ensure_ascii=False),
    )


def cargar_modelo(ruta=RUTA_MODELO):
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            "No existe el modelo entrenado. Ejecuta: python src/entrenar_modelo.py"
        )
    d = np.load(ruta, allow_pickle=False)
    return {
        "beta": d["beta"],
        "media": d["media"],
        "desviacion": d["desviacion"],
        "calibracion": {"centros": d["centros"], "prob": d["prob"]},
        "lambda": float(d["lambda_reg"]),
        "metodo": str(d["metodo"]),
        "modo": str(d["modo"]),
        "info": json.loads(str(d["info"])),
    }


def predecir_imagen(img_bgr, modelo):
    x = caracteristicas_desde_imagen(img_bgr, modelo["modo"])
    y_hat = float(predecir_valor(x, modelo)[0])
    p_perro = float(probabilidad_perro(y_hat, modelo["calibracion"]))
    confianza = confianza_prediccion(y_hat, modelo["calibracion"])
    es_perro = bool(y_hat >= UMBRAL)
    return {
        "clase": "perro" if es_perro else "gato",
        "clase_id": int(es_perro),
        "valor_y": y_hat,
        "probabilidad_perro": p_perro,
        "confianza": confianza,
        "incierto": bool(confianza < UMBRAL_INCIERTO),
    }
