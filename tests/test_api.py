import io
import os
import sys
import unittest

import cv2
import numpy as np

RAIZ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(RAIZ, "src"))
sys.path.insert(0, os.path.join(RAIZ, "api"))

from entrenamiento import entrenar
from image_processing import NOMBRES_POR_MODO
from main import crear_app


def png_bytes(semilla=0):
    rng = np.random.default_rng(semilla)
    img = rng.integers(0, 256, size=(120, 160, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def modelo_pequeno():
    rng = np.random.default_rng(0)
    n = len(NOMBRES_POR_MODO["base"])
    X = rng.normal(size=(120, n))
    y = (X[:, 0] + 0.2 * rng.normal(size=120) > 0).astype(float)
    return entrenar(X, y, lambda_reg=1.0, modo="base")


class API(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cliente = crear_app(modelo=modelo_pequeno()).test_client()

    def test_predecir_devuelve_clase_y_confianza(self):
        r = self.cliente.post(
            "/predecir",
            data={"imagen": (io.BytesIO(png_bytes()), "foto.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(r.status_code, 200)
        d = r.get_json()
        self.assertIn(d["clase"], ("perro", "gato"))
        self.assertGreaterEqual(d["confianza"], 50.0)
        self.assertLessEqual(d["confianza"], 100.0)
        self.assertIn("valor_y", d)

    def test_sin_imagen(self):
        r = self.cliente.post("/predecir", data={}, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)

    def test_extension_no_permitida(self):
        r = self.cliente.post(
            "/predecir",
            data={"imagen": (io.BytesIO(b"x"), "datos.txt")},
            content_type="multipart/form-data",
        )
        self.assertEqual(r.status_code, 400)

    def test_archivo_corrupto_con_extension_valida(self):
        r = self.cliente.post(
            "/predecir",
            data={"imagen": (io.BytesIO(b"no soy una imagen"), "rota.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(r.status_code, 400)
        self.assertIn("error", r.get_json())

    def test_info_y_pagina_principal(self):
        self.assertEqual(self.cliente.get("/info").status_code, 200)
        r = self.cliente.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"NovaVision", r.data)
        r.close()

    def test_no_sirve_archivos_fuera_de_prueba(self):
        self.assertEqual(self.cliente.get("/prueba/Otros/x.jpg").status_code, 404)
        self.assertEqual(self.cliente.get("/prueba/Gatos/x.txt").status_code, 404)


class SinModelo(unittest.TestCase):
    def test_responde_503_con_instrucciones(self):
        import main

        def falla(*a, **k):
            raise FileNotFoundError("No existe el modelo entrenado.")

        anterior = main.cargar_modelo
        main.cargar_modelo = falla
        try:
            r = crear_app().test_client().get("/info")
        finally:
            main.cargar_modelo = anterior
        self.assertEqual(r.status_code, 503)
        self.assertIn("modelo", r.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
