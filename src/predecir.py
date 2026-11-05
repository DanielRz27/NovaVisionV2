import argparse
import os
import sys

from entrenamiento import EXTENSIONES_VALIDAS, cargar_modelo, predecir_imagen
from image_processing import cargar_imagen


def listar(rutas):
    for ruta in rutas:
        if os.path.isdir(ruta):
            for raiz, _, archivos in os.walk(ruta):
                for nombre in sorted(archivos):
                    if nombre.lower().endswith(EXTENSIONES_VALIDAS):
                        yield os.path.join(raiz, nombre)
        else:
            yield ruta


def clase_real(ruta):
    carpeta = os.path.basename(os.path.dirname(ruta)).lower()
    if carpeta == "gatos":
        return "gato"
    if carpeta == "perros":
        return "perro"
    return None


def main():
    parser = argparse.ArgumentParser(description="Clasifica imágenes con NovaVision.")
    parser.add_argument("rutas", nargs="+", help="imágenes o carpetas")
    args = parser.parse_args()

    try:
        modelo = cargar_modelo()
    except FileNotFoundError as e:
        sys.exit(str(e))

    total = aciertos = con_real = 0
    for ruta in listar(args.rutas):
        try:
            r = predecir_imagen(cargar_imagen(ruta), modelo)
        except ValueError as e:
            print(f"{ruta}: error ({e})")
            continue
        real = clase_real(ruta)
        marca = ""
        if real is not None:
            con_real += 1
            if real == r["clase"]:
                aciertos += 1
                marca = "  ✓"
            else:
                marca = f"  ✗ (real: {real})"
        total += 1
        aviso = "  [poco seguro]" if r["incierto"] else ""
        print(
            f"{ruta}: {r['clase']}  ŷ={r['valor_y']:.3f}  "
            f"confianza={r['confianza']:.1f} %{aviso}{marca}"
        )

    if con_real:
        print(f"\nAciertos: {aciertos}/{con_real} ({aciertos / con_real * 100:.1f} %)")
    print(f"Imágenes procesadas: {total}")


if __name__ == "__main__":
    main()
