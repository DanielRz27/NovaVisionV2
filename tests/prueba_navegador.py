import glob
import os
import socket
import subprocess
import sys
import time
import urllib.request

from playwright.sync_api import sync_playwright

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CAPTURAS = os.path.join(RAIZ, "docs", "capturas")
fallos = []


def comprobar(descripcion, condicion, detalle=""):
    estado = "OK " if condicion else "FALLO"
    print(f"[{estado}] {descripcion}" + (f" -> {detalle}" if detalle else ""))
    if not condicion:
        fallos.append(descripcion)


def puerto_libre():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def esperar_servidor(url, intentos=60):
    for _ in range(intentos):
        try:
            urllib.request.urlopen(url, timeout=1).read()
            return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    os.makedirs(CAPTURAS, exist_ok=True)
    puerto = puerto_libre()
    entorno = dict(os.environ, NOVAVISION_PORT=str(puerto))
    servidor = subprocess.Popen(
        [sys.executable, os.path.join(RAIZ, "api", "main.py")],
        env=entorno,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{puerto}"

    try:
        if not esperar_servidor(base + "/info"):
            print("El servidor no arrancó. ¿Ejecutaste python src/entrenar_modelo.py?")
            return 1

        gato = sorted(glob.glob(os.path.join(RAIZ, "dataset", "test", "Gatos", "*.jpg")))[0]
        perro = sorted(glob.glob(os.path.join(RAIZ, "dataset", "test", "Perros", "*.jpg")))[0]
        corrupta = os.path.join(CAPTURAS, "corrupta.jpg")
        with open(corrupta, "wb") as f:
            f.write(b"esto no es una imagen")

        with sync_playwright() as p:
            navegador = p.chromium.launch(
                args=[
                    "--use-fake-device-for-media-stream",
                    "--use-fake-ui-for-media-stream",
                ]
            )
            contexto = navegador.new_context(
                viewport={"width": 760, "height": 1000}, permissions=["camera"]
            )
            pagina = contexto.new_page()
            errores_consola = []
            pagina.on("pageerror", lambda e: errores_consola.append(str(e)))
            pagina.on(
                "console",
                lambda m: errores_consola.append(m.text)
                if m.type == "error" and "400" not in m.text
                else None,
            )

            pagina.goto(base)
            comprobar("La página carga con el título correcto", "NovaVision" in pagina.title())
            pagina.wait_for_function(
                "document.getElementById('infoModelo').textContent.includes('características')"
            )
            info = pagina.inner_text("#infoModelo")
            comprobar("Muestra la información del modelo", "Modelo:" in info, info)
            comprobar("El botón Analizar empieza desactivado", pagina.is_disabled("#btnAnalizar"))
            pagina.screenshot(path=os.path.join(CAPTURAS, "01_pagina_inicial.png"), full_page=True)

            for nombre, ruta, esperado in (("gato", gato, "Gato"), ("perro", perro, "Perro")):
                pagina.set_input_files("#imagenInput", ruta)
                pagina.wait_for_function("!document.getElementById('btnAnalizar').disabled")
                comprobar(
                    f"Vista previa visible ({nombre})",
                    pagina.is_visible("#preview"),
                )
                pagina.click("#btnAnalizar")
                pagina.wait_for_selector("#resultado.perro, #resultado.gato", timeout=15000)
                texto = " ".join(pagina.inner_text("#resultado").split())
                comprobar(
                    f"Analizar imagen de {nombre} devuelve un resultado completo",
                    "Resultado:" in texto and "Confianza" in texto and "Probabilidad" in texto,
                    texto,
                )
                if nombre == "gato":
                    pagina.screenshot(path=os.path.join(CAPTURAS, "02_resultado.png"), full_page=True)

            pagina.set_input_files("#imagenInput", corrupta)
            pagina.wait_for_function("!document.getElementById('btnAnalizar').disabled")
            pagina.click("#btnAnalizar")
            pagina.wait_for_selector("#resultado.error", timeout=15000)
            comprobar(
                "Una imagen corrupta muestra un error claro",
                "imagen válida" in pagina.inner_text("#resultado"),
                " ".join(pagina.inner_text("#resultado").split()),
            )

            pagina.click("#btnCamara")
            pagina.wait_for_function(
                "document.getElementById('video').videoWidth > 0", timeout=15000
            )
            comprobar("La cámara se activa", pagina.is_visible("#camaraContainer"))
            pagina.click("#btnCapturar")
            pagina.wait_for_function(
                "document.getElementById('fileName').textContent.includes('captura_novavision')"
            )
            comprobar("La captura de la cámara se carga como imagen", True)
            pagina.click("#btnAnalizar")
            pagina.wait_for_selector("#resultado.perro, #resultado.gato", timeout=15000)
            comprobar("La foto de la cámara se puede analizar", True)

            pagina.click("#btnMuestra")
            pagina.wait_for_selector(".rejilla .tarjeta", timeout=30000)
            tarjetas = pagina.locator(".rejilla .tarjeta").count()
            resumen = pagina.inner_text("#muestraResultado .resumen")
            comprobar("La prueba rápida muestra 8 imágenes", tarjetas == 8, resumen)
            pagina.wait_for_function(
                "Array.from(document.querySelectorAll('.rejilla img')).every(i => i.complete && i.naturalWidth > 0)"
            )
            comprobar("Las 8 imágenes de prueba cargan", True)

            pagina.click("#btnEvaluar")
            pagina.wait_for_selector("table.confusion", timeout=60000)
            evaluacion = pagina.inner_text("#evaluacionResultado .resumen")
            comprobar("La evaluación completa muestra la matriz de confusión", True, evaluacion)
            pagina.locator(".prueba-rapida").screenshot(
                path=os.path.join(CAPTURAS, "03_prueba_rapida.png")
            )

            comprobar("No hay errores en la consola del navegador", not errores_consola, "; ".join(errores_consola))
            navegador.close()
    finally:
        if os.path.exists(os.path.join(CAPTURAS, "corrupta.jpg")):
            os.remove(os.path.join(CAPTURAS, "corrupta.jpg"))
        servidor.terminate()
        servidor.wait(timeout=10)

    print()
    if fallos:
        print(f"{len(fallos)} comprobación(es) fallaron.")
        return 1
    print("Todas las comprobaciones del navegador pasaron.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
