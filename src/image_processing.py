import cv2
import numpy as np

TAM = 64
GRILLA = 4
UMBRAL_OSCURO = 0.30

NOMBRES_BASE = [
    "relacion_ancho_alto",
    "brillo_promedio",
    "contraste",
    "densidad_bordes_canny",
    "bordes_horizontales_sobel_x",
    "bordes_verticales_sobel_y",
    "intensidad_centro",
    "intensidad_fondo",
    "diferencia_centro_fondo",
    "simetria_horizontal",
    "textura_laplaciano",
    "proporcion_pixeles_oscuros",
]

NOMBRES_REGIONES = [
    f"region_{i}_{j}" for i in range(GRILLA) for j in range(GRILLA)
]

NOMBRES_POR_MODO = {
    "base": NOMBRES_BASE,
    "ampliado": NOMBRES_BASE + NOMBRES_REGIONES,
}


def cargar_imagen(ruta):
    try:
        datos = np.fromfile(ruta, dtype=np.uint8)
    except OSError:
        raise ValueError(f"No se pudo leer la imagen en: {ruta}")
    return imagen_desde_bytes(datos)


def imagen_desde_bytes(datos):
    datos = np.frombuffer(datos, dtype=np.uint8) if isinstance(datos, (bytes, bytearray)) else datos
    img = cv2.imdecode(datos, cv2.IMREAD_COLOR) if datos.size else None
    if img is None:
        raise ValueError("El archivo no es una imagen válida.")
    if img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError("La imagen tiene dimensiones inválidas.")
    return img


def _recorte_gris(img_bgr):
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lado = min(h, w)
    y0 = (h - lado) // 2
    x0 = (w - lado) // 2
    cuadrada = gray[y0:y0 + lado, x0:x0 + lado]
    return cv2.resize(cuadrada, (TAM, TAM), interpolation=cv2.INTER_AREA)


def caracteristicas_desde_imagen(img_bgr, modo="base"):
    if modo not in NOMBRES_POR_MODO:
        raise ValueError(f"Modo desconocido: {modo}. Usa {list(NOMBRES_POR_MODO)}.")

    h, w = img_bgr.shape[:2]
    img = _recorte_gris(img_bgr)
    f = img.astype(np.float64) / 255.0

    centro = f[16:48, 16:48]
    mascara = np.ones((TAM, TAM), dtype=bool)
    mascara[16:48, 16:48] = False
    fondo = f[mascara]

    sobel_x = cv2.Sobel(f, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(f, cv2.CV_64F, 0, 1, ksize=3)
    laplaciano = cv2.Laplacian(f, cv2.CV_64F)
    bordes = cv2.Canny(img, 50, 150)

    media_centro = np.mean(centro)
    media_fondo = np.mean(fondo)

    base = [
        np.clip((w / h) / 2.0, 0, 1),
        np.mean(f),
        np.std(f),
        np.count_nonzero(bordes) / bordes.size,
        np.mean(np.abs(sobel_x)),
        np.mean(np.abs(sobel_y)),
        media_centro,
        media_fondo,
        abs(media_centro - media_fondo),
        np.mean(np.abs(f[:, :32] - np.fliplr(f[:, 32:]))),
        np.clip(np.var(laplaciano), 0, 1),
        np.sum(f < UMBRAL_OSCURO) / f.size,
    ]

    if modo == "ampliado":
        paso = TAM // GRILLA
        regiones = f.reshape(GRILLA, paso, GRILLA, paso).mean(axis=(1, 3)).ravel()
        base = base + list(regiones)

    return np.array(base, dtype=np.float64)


def extraer_caracteristicas(ruta_imagen, modo="base"):
    return caracteristicas_desde_imagen(cargar_imagen(ruta_imagen), modo)