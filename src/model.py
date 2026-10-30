import numpy as np
import os

from image_processing import extraer_caracteristicas
from numerical_methods import eliminacion_gauss_pivoteo


def evaluar_modelo():

    print(
        "🚀 Evaluando el modelo NovaVision...\n"
    )

    # =========================================================
    # 1. CARGAR ENTRENAMIENTO
    # =========================================================

    X_train = np.load(
        'dataset/X_train.npy'
    )

    y_train = np.load(
        'dataset/y_train.npy'
    )

    media = np.load(
        'dataset/feature_mean.npy'
    )

    desviacion = np.load(
        'dataset/feature_std.npy'
    )

    # =========================================================
    # 2. NORMALIZAR ENTRENAMIENTO
    # =========================================================

    X_train_norm = (
        X_train - media
    ) / desviacion

    m, n = X_train_norm.shape

    X_design_train = np.hstack(
        (
            np.ones((m, 1)),
            X_train_norm
        )
    )

    # =========================================================
    # 3. ECUACIONES NORMALES
    # =========================================================

    A = X_design_train.T @ X_design_train

    b = X_design_train.T @ y_train

    # =========================================================
    # 4. REGULARIZACIÓN
    # =========================================================

    lambda_reg = 0.01

    I = np.eye(n + 1)

    I[0, 0] = 0

    A = A + lambda_reg * I

    # =========================================================
    # 5. CALCULAR BETA CON GAUSS PROPIO
    # =========================================================

    beta = eliminacion_gauss_pivoteo(
        A,
        b
    )

    print(
        f"✅ Parámetros beta calculados:"
    )

    print(
        np.round(beta, 4)
    )

    # =========================================================
    # 6. CARGAR TEST
    # =========================================================

    X_test_list = []
    y_test_list = []

    clases = {
        'Gatos': 0,
        'Perros': 1
    }

    for clase, etiqueta in clases.items():

        ruta_clase = os.path.join(
            'dataset/test',
            clase
        )

        if not os.path.exists(ruta_clase):

            print(
                f"⚠️ No existe: {ruta_clase}"
            )

            continue

        archivos = sorted([
            f for f in os.listdir(ruta_clase)
            if f.lower().endswith(
                ('.jpg', '.jpeg', '.png')
            )
        ])

        print(
            f"📸 Evaluando {len(archivos)} "
            f"imágenes de {clase}..."
        )

        for img_name in archivos:

            ruta_img = os.path.join(
                ruta_clase,
                img_name
            )

            try:

                vector_x = extraer_caracteristicas(
                    ruta_img
                )

                # MISMA NORMALIZACIÓN DEL ENTRENAMIENTO

                vector_x = (
                    vector_x - media
                ) / desviacion

                vector_x = np.insert(
                    vector_x,
                    0,
                    1
                )

                X_test_list.append(
                    vector_x
                )

                y_test_list.append(
                    etiqueta
                )

            except Exception as e:

                print(
                    f"❌ Error procesando "
                    f"{img_name}: {e}"
                )

    if len(X_test_list) == 0:

        print(
            "⛔ No hay imágenes de prueba."
        )

        return

    X_test = np.array(
        X_test_list
    )

    y_test = np.array(
        y_test_list
    )

    # =========================================================
    # 7. PREDICCIÓN
    # =========================================================

    y_pred_raw = X_test @ beta

    y_pred = (
        y_pred_raw >= 0.5
    ).astype(int)

    # =========================================================
    # 8. MATRIZ DE CONFUSIÓN
    # =========================================================

    VP = np.sum(
        (y_pred == 1)
        &
        (y_test == 1)
    )

    VN = np.sum(
        (y_pred == 0)
        &
        (y_test == 0)
    )

    FP = np.sum(
        (y_pred == 1)
        &
        (y_test == 0)
    )

    FN = np.sum(
        (y_pred == 0)
        &
        (y_test == 1)
    )

    total = len(y_test)

    exactitud = (
        (VP + VN)
        /
        total
        *
        100
    )

    # =========================================================
    # 9. PRECISIÓN
    # =========================================================

    precision = (
        VP / (VP + FP) * 100
        if (VP + FP) > 0
        else 0
    )

    # =========================================================
    # 10. SENSIBILIDAD
    # =========================================================

    sensibilidad = (
        VP / (VP + FN) * 100
        if (VP + FN) > 0
        else 0
    )

    # =========================================================
    # 11. ESPECIFICIDAD
    # =========================================================

    especificidad = (
        VN / (VN + FP) * 100
        if (VN + FP) > 0
        else 0
    )

    # =========================================================
    # RESULTADOS
    # =========================================================

    print(
        "\n" + "=" * 50
    )

    print(
        "📊 RESULTADOS DE VALIDACIÓN"
    )

    print(
        "=" * 50
    )

    print(
        "\nMatriz de Confusión:"
    )

    print(
        "                 Pred. Gato    Pred. Perro"
    )

    print(
        f"Real Gato          {VN:3d} (VN)       "
        f"{FP:3d} (FP)"
    )

    print(
        f"Real Perro         {FN:3d} (FN)       "
        f"{VP:3d} (VP)"
    )

    print(
        "\n" + "-" * 50
    )

    print(
        f"✅ Exactitud:      {exactitud:.2f}%"
    )

    print(
        f"🎯 Precisión:      {precision:.2f}%"
    )

    print(
        f"🐶 Sensibilidad:   {sensibilidad:.2f}%"
    )

    print(
        f"🐱 Especificidad:  {especificidad:.2f}%"
    )

    print(
        f"📸 Total imágenes: {total}"
    )

    print(
        "=" * 50
    )


if __name__ == "__main__":

    evaluar_modelo()