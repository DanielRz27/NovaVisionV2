import os
import numpy as np
from image_processing import extraer_caracteristicas


EXTENSIONES_VALIDAS = ('.jpg', '.jpeg', '.png')


def construir_matriz_diseno(base_dir):

    X = []
    y = []

    clases = {
        'Gatos': 0,
        'Perros': 1
    }

    for clase, etiqueta in clases.items():

        ruta_clase = os.path.join(base_dir, clase)

        if not os.path.exists(ruta_clase):
            print(f"⚠️ No se encontró la carpeta: {ruta_clase}")
            continue

        archivos = sorted([
            f for f in os.listdir(ruta_clase)
            if f.lower().endswith(EXTENSIONES_VALIDAS)
        ])

        print(
            f"📸 Procesando {len(archivos)} imágenes de {clase}..."
        )

        procesadas = 0
        errores = 0

        for img_name in archivos:

            ruta_img = os.path.join(
                ruta_clase,
                img_name
            )

            try:

                vector_x = extraer_caracteristicas(
                    ruta_img
                )

                X.append(vector_x)
                y.append(etiqueta)

                procesadas += 1

            except Exception as e:

                errores += 1

                print(
                    f"❌ Error procesando {img_name}: {e}"
                )

        print(
            f"   ✅ Procesadas correctamente: {procesadas}"
        )

        if errores > 0:
            print(
                f"   ⚠️ Errores: {errores}"
            )

    X_matrix = np.array(X, dtype=np.float64)
    y_vector = np.array(y, dtype=np.float64)

    return X_matrix, y_vector


if __name__ == "__main__":

    print(
        "🚀 Iniciando construcción del Dataset NovaVision...\n"
    )

    X_train, y_train = construir_matriz_diseno(
        "dataset/train"
    )

    if len(X_train) == 0:

        print(
            "⛔ No se encontraron imágenes."
        )

        exit()

    print(
        "\n✅ ¡Matrices construidas exitosamente!"
    )

    print(
        f"📊 Matriz X: {X_train.shape}"
    )

    print(
        f"🏷️ Vector y: {y_train.shape}"
    )

    print(
        f"🐱 Gatos: {np.sum(y_train == 0)}"
    )

    print(
        f"🐶 Perros: {np.sum(y_train == 1)}"
    )

    np.save(
        "dataset/X_train.npy",
        X_train
    )

    np.save(
        "dataset/y_train.npy",
        y_train
    )

    print(
        "\n💾 Datos guardados:"
    )

    print(
        "   → dataset/X_train.npy"
    )

    print(
        "   → dataset/y_train.npy"
    )