"""
Pruebas automáticas de los métodos numéricos de NovaVision.

Ejecutar desde la carpeta raíz del proyecto:

    python -m unittest discover tests -v
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "src")
)

from numerical_methods import (
    eliminacion_gauss_pivoteo,
    factorizacion_lu,
    lu_descomponer,
    resolver_lu,
    jacobi,
    gauss_seidel,
    norma_vector,
    norma_matriz,
    numero_condicion,
    error_relativo,
)
from validation import sistemas_conocidos
from entrenamiento import entrenar, predecir_valor


class SistemasConocidos(unittest.TestCase):

    def test_metodos_directos_recuperan_la_solucion(self):

        for s in sistemas_conocidos():

            for f in (eliminacion_gauss_pivoteo, factorizacion_lu):

                x = f(s["A"], s["b"])

                # Hilbert está muy mal condicionada: se tolera más error
                tol = 1e-4 if "Hilbert" in s["nombre"] else 1e-10

                self.assertLess(
                    error_relativo(s["x"], x), tol,
                    f"{f.__name__} falló en {s['nombre']}"
                )

    def test_metodos_iterativos_convergen_en_sistemas_dominantes(self):

        for s in sistemas_conocidos():

            if not s["iterativos"]:
                continue

            for f in (jacobi, gauss_seidel):

                x, iters, ok = f(s["A"], s["b"], tol=1e-12)

                self.assertTrue(ok, f"{f.__name__} no convergió en {s['nombre']}")

                self.assertLess(error_relativo(s["x"], x), 1e-9)

    def test_gauss_seidel_converge_mas_rapido_que_jacobi(self):

        s = sistemas_conocidos()[0]

        _, it_j, _ = jacobi(s["A"], s["b"])

        _, it_g, _ = gauss_seidel(s["A"], s["b"])

        self.assertLess(it_g, it_j)

    def test_jacobi_detecta_divergencia(self):

        A = np.array([[1.0, 2.0], [3.0, 1.0]])

        b = np.array([3.0, 4.0])

        _, _, ok = jacobi(A, b)

        self.assertFalse(ok)

    def test_matriz_singular_lanza_error(self):

        A = np.array([[1.0, 2.0], [2.0, 4.0]])

        b = np.array([1.0, 2.0])

        with self.assertRaises(ValueError):
            eliminacion_gauss_pivoteo(A, b)

        with self.assertRaises(ValueError):
            factorizacion_lu(A, b)

    def test_historial_registra_un_error_por_iteracion(self):

        s = sistemas_conocidos()[0]

        h = []

        _, iters, _ = gauss_seidel(s["A"], s["b"], historial=h)

        self.assertEqual(len(h), iters)


class LUReutilizable(unittest.TestCase):

    def test_una_factorizacion_varios_b(self):

        s = sistemas_conocidos()[2]

        P, L, U = lu_descomponer(s["A"])

        self.assertTrue(np.allclose(P @ s["A"], L @ U))

        rng = np.random.default_rng(1)

        for _ in range(3):

            x_real = rng.normal(size=s["A"].shape[0])

            x = resolver_lu(P, L, U, s["A"] @ x_real)

            self.assertLess(error_relativo(x_real, x), 1e-10)


class NormasYCondicionamiento(unittest.TestCase):

    def setUp(self):

        rng = np.random.default_rng(3)

        self.A = rng.normal(size=(6, 6))

        self.v = rng.normal(size=6)

    def test_normas_de_vector(self):

        for p in (1, 2, np.inf):

            self.assertAlmostEqual(
                norma_vector(self.v, p),
                np.linalg.norm(self.v, ord=p)
            )

    def test_normas_de_matriz(self):

        for t in (1, 2, np.inf, "fro"):

            esperado = np.linalg.norm(self.A, ord=t)

            self.assertAlmostEqual(
                norma_matriz(self.A, t) / esperado, 1.0, places=6
            )

    def test_numero_de_condicion(self):

        for t in (1, 2, np.inf):

            self.assertAlmostEqual(
                numero_condicion(self.A, t) / np.linalg.cond(self.A, t),
                1.0, places=6
            )


class EcuacionesNormales(unittest.TestCase):

    def test_gauss_y_lu_coinciden_con_numpy(self):

        rng = np.random.default_rng(5)

        X = rng.normal(size=(200, 5))

        y = (X @ np.array([0.5, -1.0, 0.0, 2.0, 0.3]) > 0).astype(float)

        m_g = entrenar(X, y, lambda_reg=0.01, metodo="gauss")

        m_l = entrenar(X, y, lambda_reg=0.01, metodo="lu")

        beta_np = np.linalg.solve(m_g["A"], m_g["b"])

        self.assertLess(error_relativo(beta_np, m_g["beta"]), 1e-9)

        self.assertLess(error_relativo(beta_np, m_l["beta"]), 1e-9)

        self.assertEqual(len(predecir_valor(X, m_g)), 200)

    def test_ridge_arregla_caracteristicas_colineales(self):

        rng = np.random.default_rng(7)

        X = rng.normal(size=(100, 3))

        X = np.hstack((X, 0.25 * X[:, [0]] + 0.75 * X[:, [1]]))

        y = (X[:, 0] > 0).astype(float)

        with self.assertRaises(ValueError):
            entrenar(X, y, lambda_reg=0.0)

        modelo = entrenar(X, y, lambda_reg=0.01)

        self.assertTrue(np.all(np.isfinite(modelo["beta"])))


if __name__ == "__main__":

    unittest.main()
