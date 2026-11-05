import os
import random
import sys

from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from entrenamiento import (
    CLASES,
    DATASET_DIR,
    EXTENSIONES_VALIDAS,
    cargar_imagenes,
    cargar_modelo,
    clasificar,
    metricas_clasificacion,
    predecir_imagen,
    predecir_valor,
)
from image_processing import cargar_imagen, imagen_desde_bytes

DIR_WEB = next(
    (
        os.path.join(BASE_DIR, nombre)
        for nombre in ("Web", "web")
        if os.path.isdir(os.path.join(BASE_DIR, nombre))
    ),
    os.path.join(BASE_DIR, "Web"),
)

ETIQUETAS = {"gato": "Gato", "perro": "Perro"}


def _respuesta_prediccion(r):
    return {
        "clase": r["clase"],
        "etiqueta": ETIQUETAS[r["clase"]],
        "valor_y": round(r["valor_y"], 4),
        "probabilidad_perro": round(r["probabilidad_perro"] * 100, 1),
        "confianza": round(r["confianza"], 1),
        "incierto": r["incierto"],
    }


def crear_app(modelo=None, dataset_dir=DATASET_DIR):
    app = Flask(__name__, static_folder=DIR_WEB, static_url_path="")
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
    estado = {"modelo": modelo, "error": None, "evaluacion": None}

    if estado["modelo"] is None:
        try:
            estado["modelo"] = cargar_modelo()
        except FileNotFoundError as e:
            estado["error"] = str(e)

    def sin_modelo():
        return jsonify({"error": estado["error"] or "El modelo no está cargado."}), 503

    @app.route("/")
    def index():
        return send_from_directory(DIR_WEB, "index.html")

    @app.route("/info")
    def info():
        if estado["modelo"] is None:
            return sin_modelo()
        m = estado["modelo"]
        datos = m.get("info", {})
        return jsonify(
            {
                "modo": m["modo"],
                "metodo": m["metodo"],
                "lambda": m["lambda"],
                "n_caracteristicas": int(len(m["beta"]) - 1),
                "n_train": datos.get("n_train"),
                "n_test": datos.get("n_test"),
                "exactitud_cv": datos.get("exactitud_cv"),
                "exactitud_test": datos.get("exactitud_test"),
            }
        )

    @app.route("/predecir", methods=["POST"])
    def predecir():
        if estado["modelo"] is None:
            return sin_modelo()
        archivo = request.files.get("imagen")
        if archivo is None:
            return jsonify({"error": "No se envió ninguna imagen."}), 400
        if archivo.filename == "":
            return jsonify({"error": "Nombre de archivo vacío."}), 400
        if not archivo.filename.lower().endswith(EXTENSIONES_VALIDAS):
            return jsonify({"error": "Formato no permitido. Usa JPG o PNG."}), 400
        try:
            img = imagen_desde_bytes(archivo.read())
            resultado = predecir_imagen(img, estado["modelo"])
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        return jsonify(_respuesta_prediccion(resultado))

    @app.route("/muestra")
    def muestra():
        if estado["modelo"] is None:
            return sin_modelo()
        n = max(2, min(request.args.get("n", 8, type=int), 24))
        semilla = request.args.get("semilla", type=int)
        azar = random.Random(semilla)
        elegidas = []
        for clase, etiqueta in CLASES.items():
            carpeta = os.path.join(dataset_dir, "test", clase)
            if not os.path.isdir(carpeta):
                continue
            archivos = sorted(
                f for f in os.listdir(carpeta) if f.lower().endswith(EXTENSIONES_VALIDAS)
            )
            for nombre in azar.sample(archivos, min(n // 2, len(archivos))):
                elegidas.append((clase, nombre, etiqueta))
        azar.shuffle(elegidas)

        salida = []
        for clase, nombre, etiqueta in elegidas:
            img = cargar_imagen(os.path.join(dataset_dir, "test", clase, nombre))
            r = predecir_imagen(img, estado["modelo"])
            real = "perro" if etiqueta == 1 else "gato"
            salida.append(
                {
                    "archivo": f"{clase}/{nombre}",
                    "url": f"/prueba/{clase}/{nombre}",
                    "real": real,
                    "acierto": r["clase"] == real,
                    **_respuesta_prediccion(r),
                }
            )
        return jsonify({"imagenes": salida})

    @app.route("/evaluar")
    def evaluar():
        if estado["modelo"] is None:
            return sin_modelo()
        if estado["evaluacion"] is None:
            m = estado["modelo"]
            X, y, _ = cargar_imagenes("test", m["modo"], dataset_dir)
            if len(y) == 0:
                return jsonify({"error": "No hay imágenes de prueba."}), 404
            estado["evaluacion"] = metricas_clasificacion(
                y, clasificar(predecir_valor(X, m))
            )
        return jsonify(estado["evaluacion"])

    @app.route("/prueba/<clase>/<archivo>")
    def imagen_prueba(clase, archivo):
        if clase not in CLASES or not archivo.lower().endswith(EXTENSIONES_VALIDAS):
            return jsonify({"error": "Imagen no encontrada."}), 404
        return send_from_directory(os.path.join(dataset_dir, "test", clase), archivo)

    @app.errorhandler(413)
    def demasiado_grande(_):
        return jsonify({"error": "La imagen supera los 10 MB."}), 413

    return app


if __name__ == "__main__":
    app = crear_app()
    app.run(
        host=os.environ.get("NOVAVISION_HOST", "127.0.0.1"),
        port=int(os.environ.get("NOVAVISION_PORT", "5000")),
        debug=os.environ.get("NOVAVISION_DEBUG") == "1",
    )
