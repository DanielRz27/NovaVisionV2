import os
import sys

import numpy as np

from flask import (
    Flask,
    render_template,
    request,
    jsonify
)


# ============================================================
# CONFIGURAR PATH
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        '..'
    )
)

SRC_DIR = os.path.join(
    BASE_DIR,
    'src'
)

sys.path.append(SRC_DIR)


from image_processing import extraer_caracteristicas

from numerical_methods import (
    eliminacion_gauss_pivoteo
)


# ============================================================
# FLASK
# ============================================================

app = Flask(
    __name__,
    template_folder=os.path.join(
        BASE_DIR,
        'web'
    ),
    static_folder=os.path.join(
        BASE_DIR,
        'web'
    ),
    static_url_path=''
)


# ============================================================
# VARIABLES DEL MODELO
# ============================================================

BETA_OPTIMO = None

FEATURE_MEAN = None

FEATURE_STD = None


# ============================================================
# ENTRENAMIENTO
# ============================================================

def entrenar_modelo_al_inicio():

    global BETA_OPTIMO
    global FEATURE_MEAN
    global FEATURE_STD

    try:

        # ----------------------------------------------------
        # Cargar datos
        # ----------------------------------------------------

        X_train = np.load(
            os.path.join(
                BASE_DIR,
                'dataset',
                'X_train.npy'
            )
        )

        y_train = np.load(
            os.path.join(
                BASE_DIR,
                'dataset',
                'y_train.npy'
            )
        )

        # ----------------------------------------------------
        # Cargar parámetros de normalización
        # ----------------------------------------------------

        FEATURE_MEAN = np.load(
            os.path.join(
                BASE_DIR,
                'dataset',
                'feature_mean.npy'
            )
        )

        FEATURE_STD = np.load(
            os.path.join(
                BASE_DIR,
                'dataset',
                'feature_std.npy'
            )
        )

        # ----------------------------------------------------
        # Normalizar
        # ----------------------------------------------------

        X_norm = (
            X_train - FEATURE_MEAN
        ) / FEATURE_STD

        m, n = X_norm.shape

        # ----------------------------------------------------
        # Matriz de diseño
        # ----------------------------------------------------

        X_design = np.hstack(
            (
                np.ones((m, 1)),
                X_norm
            )
        )

        # ----------------------------------------------------
        # Ecuaciones normales
        # ----------------------------------------------------

        A = X_design.T @ X_design

        b = X_design.T @ y_train

        # ----------------------------------------------------
        # Ridge
        # ----------------------------------------------------

        lambda_reg = 0.01

        I = np.eye(n + 1)

        I[0, 0] = 0

        A = A + lambda_reg * I

        # ----------------------------------------------------
        # Gauss propio
        # ----------------------------------------------------

        BETA_OPTIMO = (
            eliminacion_gauss_pivoteo(
                A,
                b
            )
        )

        print(
            "✅ Modelo NovaVision cargado."
        )

        print(
            "   Método: Eliminación de Gauss"
        )

        print(
            f"   Características: {n}"
        )

        print(
            f"   Parámetros beta: "
            f"{np.round(BETA_OPTIMO, 4)}"
        )

    except Exception as e:

        print(
            f"❌ Error al cargar el modelo: {e}"
        )


# ============================================================
# PÁGINA PRINCIPAL
# ============================================================

@app.route('/')
def index():

    return render_template(
        'index.html'
    )


# ============================================================
# PREDICCIÓN
# ============================================================

@app.route(
    '/predecir',
    methods=['POST']
)
def predecir():

    if BETA_OPTIMO is None:

        return jsonify({
            'error':
            'El modelo todavía no está cargado.'
        }), 500

    if 'imagen' not in request.files:

        return jsonify({
            'error':
            'No se envió ninguna imagen.'
        }), 400

    file = request.files['imagen']

    if file.filename == '':

        return jsonify({
            'error':
            'Nombre de archivo vacío.'
        }), 400

    if not file.filename.lower().endswith(
        ('.png', '.jpg', '.jpeg')
    ):

        return jsonify({
            'error':
            'Formato no permitido. '
            'Usa JPG o PNG.'
        }), 400

    temp_path = os.path.join(
        BASE_DIR,
        'temp_captura.jpg'
    )

    try:

        # ----------------------------------------------------
        # Guardar imagen
        # ----------------------------------------------------

        file.save(
            temp_path
        )

        # ----------------------------------------------------
        # Extraer características
        # ----------------------------------------------------

        vector_x = extraer_caracteristicas(
            temp_path
        )

        # ----------------------------------------------------
        # Normalizar exactamente igual
        # que durante entrenamiento
        # ----------------------------------------------------

        vector_x = (
            vector_x - FEATURE_MEAN
        ) / FEATURE_STD

        # ----------------------------------------------------
        # Agregar término independiente
        # ----------------------------------------------------

        x_con_uno = np.insert(
            vector_x,
            0,
            1
        )

        # ----------------------------------------------------
        # Predicción
        # ----------------------------------------------------

        y_pred_raw = float(
            np.dot(
                x_con_uno,
                BETA_OPTIMO
            )
        )

        # ----------------------------------------------------
        # Clasificación
        # ----------------------------------------------------

        if y_pred_raw >= 0.5:

            clase = "Perro 🐶"

        else:

            clase = "Gato 🐱"

        # ----------------------------------------------------
        # Distancia al umbral
        # ----------------------------------------------------

        distancia = abs(
            y_pred_raw - 0.5
        )

        confianza = (
            50 + distancia * 100
        )

        confianza = min(
            max(confianza, 50),
            99.9
        )

        # ----------------------------------------------------
        # Eliminar temporal
        # ----------------------------------------------------

        if os.path.exists(temp_path):

            os.remove(
                temp_path
            )

        return jsonify({

            'clase': clase,

            'valor_y':
            round(y_pred_raw, 6),

            'confianza':
            round(confianza, 1)

        })

    except Exception as e:

        if os.path.exists(temp_path):

            os.remove(
                temp_path
            )

        return jsonify({
            'error': str(e)
        }), 500


# ============================================================
# INICIAR
# ============================================================

if __name__ == '__main__':

    print(
        "🚀 Iniciando servidor web de NovaVision..."
    )

    entrenar_modelo_al_inicio()

    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000
    )