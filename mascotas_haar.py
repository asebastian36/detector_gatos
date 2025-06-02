# mascotas_haar.py

import cv2
import numpy as np
import os
import shutil  # Importamos la librería shutil para eliminar directorios

# --- 1. Configuración del Proyecto ---

# Define la ruta al clasificador Haar que quieres usar para caras de gatos
HAARCASCADE_PATH = 'haarcascade_frontalcatface_extended.xml'

# Parámetros para la función detectMultiScale().
# Ajusta estos valores para optimizar la detección según tus imágenes y condiciones
SCALE_FACTOR = 1.03  # Un buen equilibrio entre detalle y velocidad
MIN_NEIGHBORS = 9  # Punto intermedio para la confianza de detección
MIN_SIZE = (30, 30)  # Tamaño mínimo de la cara de gato. Ajusta si tienes gatos muy pequeños
MAX_SIZE = (500, 500)  # Tamaño máximo

# Rutas de las carpetas para la entrada y salida de imágenes.
INPUT_FOLDER = 'input_images'
OUTPUT_FOLDER = 'output_images'

# --- 2. Carga del Clasificador Haar ---

# Inicializa el clasificador de cascada con el archivo XML especificado
clasificador_gato = cv2.CascadeClassifier(HAARCASCADE_PATH)

# Verifica si el clasificador se cargó correctamente. Si no, el programa sale
if clasificador_gato.empty():
    print(f"ERROR: No se pudo cargar el clasificador desde '{HAARCASCADE_PATH}'.")
    print("Asegúrate de que el archivo .xml está en la misma carpeta que este script o la ruta es correcta.")
    exit()
else:
    print(f"Clasificador cargado exitosamente.")


# --- 3. Funciones de Procesamiento y Detección ---

def detectar_caras_gatos(frame):
    """
    Detecta caras de gatos en un frame (imagen) dado utilizando el clasificador Haar cargado.

    """
    # Convertir la imagen a escala de grises. Las características Haar operan sobre intensidades de píxeles.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Implementación de Filtros de Preprocesamiento ---

    # Ecualización Adaptativa del Histograma (CLAHE)
    # Recomendado para mejorar el contraste localmente en diferentes condiciones de iluminación.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Opción 2: Desenfoque Gaussiano (para reducir ruido)
    # Puede ayudar a eliminar ruido que confunde al clasificador, pero un desenfoque excesivo elimina detalles.
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Aplicar la función detectMultiScale() para encontrar las caras de gatos.
    # Esta función devuelve una lista de rectángulos (x, y, ancho, alto) de los objetos detectados.
    caras_gatos = clasificador_gato.detectMultiScale(
        gray,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=MIN_NEIGHBORS,
        minSize=MIN_SIZE,
        maxSize=MAX_SIZE
    )

    # Dibujar un rectángulo y texto sobre cada cara de gato detectada.
    for (x, y, w, h) in caras_gatos:
        # Dibuja un rectángulo en el frame original (a color). Color verde (0, 255, 0), grosor 2.
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Añade el texto "Gato" encima del recuadro.
        cv2.putText(frame, "Gato", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame


def procesar_imagenes_en_carpeta():
    """
    Procesa todas las imágenes encontradas en la carpeta de entrada (INPUT_FOLDER)
    y guarda las imágenes resultantes (con las detecciones) en la carpeta de salida (OUTPUT_FOLDER).
    """
    # --- NUEVO: Eliminar la carpeta de salida si ya existe ---
    if os.path.exists(OUTPUT_FOLDER):
        print(f"Eliminando la carpeta de salida existente: '{OUTPUT_FOLDER}' y su contenido.")
        shutil.rmtree(OUTPUT_FOLDER)

    # Crea la carpeta de salida (ahora limpia o recién creada).
    os.makedirs(OUTPUT_FOLDER)
    print(f"Creada carpeta de salida: '{OUTPUT_FOLDER}'")

    print(f"\n--- Procesando imágenes de la carpeta: '{INPUT_FOLDER}' ---")

    # Recorre todos los archivos en la carpeta de entrada.
    for filename in os.listdir(INPUT_FOLDER):
        # Procesa solo archivos de imagen comunes.
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(INPUT_FOLDER, filename)  # Ruta completa de la imagen.
            img = cv2.imread(img_path)  # Carga la imagen.

            if img is None:
                print(f"Advertencia: No se pudo cargar la imagen '{img_path}'. Saltando.")
                continue

            print(f"Procesando imagen: '{filename}'")

            # Realiza la detección en una copia de la imagen para no modificar el original.
            img_procesada = detectar_caras_gatos(img.copy())

            # Define la ruta para guardar la imagen procesada.
            output_path = os.path.join(OUTPUT_FOLDER, f"detectado_{filename}")
            cv2.imwrite(output_path, img_procesada)  # Guarda la imagen resultante.

            print(f"Guardada imagen procesada: '{output_path}'")

    print(f"\nProcesamiento de imágenes finalizado. Resultados en: '{OUTPUT_FOLDER}'")


def procesar_webcam():
    """
    Inicia la detección de caras de gatos en tiempo real utilizando la webcam.
    Muestra el feed de la webcam con las detecciones en una ventana.
    """
    cap = cv2.VideoCapture(0)  # '0' se refiere a la webcam predeterminada del sistema.

    if not cap.isOpened():
        print("ERROR: No se pudo abrir la webcam.")
        print("Asegúrate de que la webcam está conectada, encendida y no está siendo usada por otra aplicación.")
        return

    print("\n--- Modo Webcam Activado (Presiona 'q' para salir de la ventana) ---")

    while True:
        ret, frame = cap.read()  # Lee un nuevo frame de la webcam.

        if not ret:
            print("ERROR: No se pudo leer el frame de la webcam. Saliendo...")
            break

        # Realiza la detección en una copia del frame actual.
        frame_procesado = detectar_caras_gatos(frame.copy())

        # Muestra el frame procesado en una ventana llamada 'Deteccion de Caras de Gatos (Webcam)'.
        cv2.imshow('Deteccion de Caras de Gatos (Webcam)', frame_procesado)

        # Espera 1 milisegundo por una tecla. Si la tecla 'q' es presionada, sale del bucle.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera los recursos de la webcam y cierra todas las ventanas de OpenCV.
    cap.release()
    cv2.destroyAllWindows()
    print("Modo Webcam finalizado.")


# --- 4. Ejecución Principal del Programa ---

if __name__ == "__main__":
    print("--- Iniciando Proyecto: Detección de Caras de Gatos con Haar Cascades ---")

    # Preguntar al usuario qué modo quiere usar
    while True:
        choice = input(
            "¿Qué modo de detección quieres usar?\n1. Procesar imágenes de carpeta\n2. Usar webcam\nIngresa 1 o 2: ")
        if choice == '1':
            procesar_imagenes_en_carpeta()
            break
        elif choice == '2':
            procesar_webcam()
            break
        else:
            print("Opción no válida. Por favor, ingresa '1' o '2'.")

    print("\n--- Programa Finalizado ---")