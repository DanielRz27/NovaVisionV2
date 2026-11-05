import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from image_processing import (
    NOMBRES_POR_MODO,
    caracteristicas_desde_imagen,
    imagen_desde_bytes,
)


def imagen_sintetica(alto=240, ancho=320, semilla=0):
    rng = np.random.default_rng(semilla)
    img = rng.integers(0, 256, size=(alto, ancho, 3), dtype=np.uint8)
    return cv2.GaussianBlur(img, (7, 7), 0)


class Caracteristicas(unittest.TestCase):
    def test_longitud_coincide_con_los_nombres(self):
        img = imagen_sintetica()
        for modo, nombres in NOMBRES_POR_MODO.items():
            v = caracteristicas_desde_imagen(img, modo)
            self.assertEqual(v.shape, (len(nombres),), modo)
            self.assertTrue(np.all(np.isfinite(v)), modo)

    def test_avanzado_no_incluye_el_brillo_redundante(self):
        self.assertNotIn("brillo_promedio", NOMBRES_POR_MODO["avanzado"])
        self.assertIn("brillo_promedio", NOMBRES_POR_MODO["base"])

    def test_modo_desconocido_falla(self):
        with self.assertRaises(ValueError):
            caracteristicas_desde_imagen(imagen_sintetica(), "inventado")

    def test_bytes_equivalen_a_la_imagen_original(self):
        img = imagen_sintetica(semilla=3)
        ok, buf = cv2.imencode(".png", img)
        self.assertTrue(ok)
        leida = imagen_desde_bytes(buf.tobytes())
        self.assertTrue(np.array_equal(leida, img))
        for modo in NOMBRES_POR_MODO:
            self.assertTrue(
                np.allclose(
                    caracteristicas_desde_imagen(leida, modo),
                    caracteristicas_desde_imagen(img, modo),
                )
            )

    def test_bytes_invalidos_lanzan_error(self):
        with self.assertRaises(ValueError):
            imagen_desde_bytes(b"esto no es una imagen")
        with self.assertRaises(ValueError):
            imagen_desde_bytes(b"")

    def test_imagen_muy_pequena_no_rompe(self):
        for modo in NOMBRES_POR_MODO:
            v = caracteristicas_desde_imagen(imagen_sintetica(40, 30), modo)
            self.assertEqual(len(v), len(NOMBRES_POR_MODO[modo]))


if __name__ == "__main__":
    unittest.main()
