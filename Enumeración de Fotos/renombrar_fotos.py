import os
import glob

# ==========================================
# CONFIGURACION
# ==========================================

carpeta_fotos = "C:/Users/27ale/OneDrive/Escritorio/NovaVision/dataset/train/Gatos"
nombre_base = "gato"
inicio_numero = 1

# ==========================================
# LOGICA DEL SCRIPT
# ==========================================

if not os.path.exists(carpeta_fotos):
    print(f"[ERROR] La carpeta '{carpeta_fotos}' no existe. Revisa la ruta.")
else:
    extensiones = ('*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.webp', '*.heic')
    
    archivos_imagenes = []
    for ext in extensiones:
        # Buscamos las extensiones
        archivos_imagenes.extend(glob.glob(os.path.join(carpeta_fotos, ext)))
        
    # ¡LA CORRECCIÓN! 
    # Usamos "set" para eliminar archivos duplicados y luego ordenamos
    archivos_imagenes = sorted(list(set(archivos_imagenes)))
    
    print(f"[OK] Se encontraron {len(archivos_imagenes)} imagenes unicas en la carpeta.\n")
    
    if len(archivos_imagenes) == 0:
        print("No hay imagenes para renombrar.")
    else:
        for i, archivo in enumerate(archivos_imagenes, start=inicio_numero):
            nombre_archivo, extension = os.path.splitext(archivo)
            nuevo_nombre = f"{nombre_base}_{str(i).zfill(3)}{extension}"
            nueva_ruta = os.path.join(carpeta_fotos, nuevo_nombre)
            
            # Proteccion: Si el archivo ya se llama como queremos, lo saltamos
            if os.path.abspath(archivo) == os.path.abspath(nueva_ruta):
                print(f"-> {os.path.basename(archivo)} (Ya tenia este nombre, saltado)")
                continue
                
            try:
                os.rename(archivo, nueva_ruta)
                print(f"-> {os.path.basename(archivo)}  -->  {nuevo_nombre}")
            except FileExistsError:
                print(f"[AVISO] {nuevo_nombre} ya existe, saltando {os.path.basename(archivo)}")
            except Exception as e:
                print(f"[ERROR] No se pudo renombrar {os.path.basename(archivo)}: {e}")
            
        print(f"\n[LISTO] Se procesaron las fotos correctamente.")