import numpy as np


def preparar_sistema():

    print(
        "Preparando el sistema de ecuaciones normales para NovaVision...\n"
    )

    # =========================================================
    # 1. CARGAR DATOS
    # =========================================================

    X = np.load(
        'dataset/X_train.npy'
    )

    y = np.load(
        'dataset/y_train.npy'
    )

    m, n = X.shape

    print(
        f"Matriz X original: {m} imágenes, {n} características."
    )

    # =========================================================
    # 2. NORMALIZACIÓN DE CARACTERÍSTICAS
    # =========================================================

    media = np.mean(X, axis=0)

    desviacion = np.std(X, axis=0)

    desviacion[desviacion < 1e-12] = 1.0

    X_normalizada = (
        (X - media) / desviacion
    )

    # Guardamos parámetros para utilizar exactamente
    # la misma transformación cuando llegue una imagen nueva.

    np.save(
        'dataset/feature_mean.npy',
        media
    )

    np.save(
        'dataset/feature_std.npy',
        desviacion
    )

    print(
        "✅ Características normalizadas."
    )

    # =========================================================
    # 3. MATRIZ DE DISEÑO
    # =========================================================

    X_design = np.hstack(
        (
            np.ones((m, 1)),
            X_normalizada
        )
    )

    print(
        f"Matriz de Diseño: {X_design.shape}"
    )

    # =========================================================
    # 4. ECUACIONES NORMALES
    # =========================================================

    A = X_design.T @ X_design

    b = X_design.T @ y

    # =========================================================
    # 5. REGULARIZACIÓN RIDGE
    # =========================================================

    lambda_reg = 0.01

    I = np.eye(n + 1)

    # No regularizamos beta_0
    I[0, 0] = 0

    A_regularizada = (
        A + lambda_reg * I
    )

    print(
        f"\nRegularización Ridge: λ = {lambda_reg}"
    )

    print(
        f"Matriz A: {A_regularizada.shape}"
    )

    print(
        f"Vector b: {b.shape}"
    )

    # =========================================================
    # 6. CONDICIONAMIENTO
    # =========================================================

    cond_original = np.linalg.cond(A)

    cond_regularizada = np.linalg.cond(
        A_regularizada
    )

    print(
        f"\nNúmero de condicionamiento original: "
        f"{cond_original:.2e}"
    )

    print(
        f"Número de condicionamiento regularizado: "
        f"{cond_regularizada:.2e}"
    )

    # =========================================================
    # 7. GUARDAR
    # =========================================================

    np.save(
        'dataset/A_normal.npy',
        A_regularizada
    )

    np.save(
        'dataset/b_normal.npy',
        b
    )

    print(
        "\n✅ Sistema lineal guardado exitosamente:"
    )

    print(
        "   → dataset/A_normal.npy"
    )

    print(
        "   → dataset/b_normal.npy"
    )

    return A_regularizada, b


if __name__ == "__main__":

    A, b = preparar_sistema()

    print(
        "\n--- Vista previa del sistema ---"
    )

    print(
        "Primera fila de A:"
    )

    print(
        np.round(A[0], 4)
    )

    print(
        "\nVector b:"
    )

    print(
        np.round(b, 4)
    )