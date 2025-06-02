# Proyecto: Detección de objetos usando características Haar

La finalidad de este proyecto es reconocer gatos con vision por computadora ya sea por medio de imagenes o en tiempo real proporcionado por webcam.

## Teoria basica

Las características de Haar son filtros basados en patrones rectangulares utilizados en visión por computadora, especialmente en el algoritmo de detección de objetos de Viola-Jones. Fueron introducidas en 2001 por Paul Viola y Michael Jones para detectar rostros en imágenes en tiempo real, pero su aplicación se ha extendido a otros objetos (como coches, señales de tráfico, etc.).

## Cómo Funciona

### Documentación del Proyecto: Detección de Caras de Gatos con Haar Cascades (`mascotas_haar.py`)

Este documento describe el funcionamiento del script principal `mascotas_haar.py`, diseñado para detectar caras de gatos. El proyecto aprovecha la técnica de **Clasificadores en Cascada de Haar**, una característica robusta de la biblioteca **OpenCV**.

### 1. Visión General del Proyecto

El script `mascotas_haar.py` implementa un sistema básico de detección de objetos. Su objetivo principal es identificar y marcar la presencia de caras de gatos en dos fuentes de entrada distintas:
1.  **Imágenes estáticas:** Procesando un conjunto de imágenes desde una carpeta.
2.  **Tiempo real:** Utilizando una webcam en vivo.

El programa ofrece al usuario la opción de seleccionar el modo de operación deseado al iniciar. Internamente, utiliza un archivo XML pre-entrenado (`haarcascade_frontalcatface_extended.xml`) que contiene los patrones (conocidos como características Haar) necesarios para reconocer caras frontales de gatos.

### 2. Dependencias del Proyecto

Para ejecutar este proyecto, son necesarias las siguientes librerías de Python:

* **`opencv-python`**: La biblioteca principal de Visión por Computadora, que incluye la implementación de los clasificadores Haar y funciones para el procesamiento de imágenes y video.
* **`os`**: Módulo estándar de Python para interactuar con el sistema operativo, esencial para el manejo de rutas de archivos y directorios.
* **`shutil`**: Módulo estándar de Python que proporciona operaciones de alto nivel con archivos y directorios (por ejemplo, eliminar árboles de directorios).

**Instalación de Dependencias:**
Asegúrate de que estas librerías estén instaladas en tu entorno. Puedes instalarlas fácilmente usando `pip`:

```bash
pip install opencv-python numpy
```

### 3. Estructura del Código

El código de `mascotas_haar.py` está organizado en las siguientes secciones principales para facilitar su comprensión y mantenimiento:

### 3.1. Configuración del Proyecto

Esta sección define las constantes y parámetros globales que controlan el comportamiento y la sensibilidad del detector:

* **`HAARCASCADE_PATH`**: La ruta al archivo XML del clasificador Haar. Se utiliza `'haarcascade_frontalcatface_extended.xml'`, optimizado para la detección de caras de gatos frontales. Este archivo debe residir en el mismo directorio que el script o su ruta debe ser especificada correctamente.
* **`SCALE_FACTOR`**: Controla la reducción de tamaño de la imagen en cada paso de la detección. Un valor de `1.03` significa una reducción del 3%, lo que permite una detección más minuciosa pero a un costo de velocidad ligeramente mayor.
* **`MIN_NEIGHBORS`**: Determina cuántas detecciones superpuestas (vecinos) un rectángulo candidato debe tener para ser considerado una detección válida. Un valor de `9` ofrece un buen equilibrio entre la confianza de detección y la minimización de falsos positivos. Valores más bajos pueden aumentar las detecciones, pero también los errores.
* **`MIN_SIZE` y `MAX_SIZE`**: Definen el rango de tamaño (ancho, alto en píxeles) de los objetos que el detector debe considerar. Esto ayuda a filtrar ruido (objetos muy pequeños) o detecciones erróneas (objetos excesivamente grandes).
* **`INPUT_FOLDER`**: Nombre de la carpeta de donde el script leerá las imágenes para su procesamiento (por ejemplo, `'input_images'`).
* **`OUTPUT_FOLDER`**: Nombre de la carpeta donde se guardarán las imágenes procesadas con las detecciones marcadas (por ejemplo, `'output_images'`).

### 3.2. Carga del Clasificador Haar

Esta sección inicializa el clasificador de cascada:

```python
clasificador_gato = cv2.CascadeClassifier(HAARCASCADE_PATH)

if clasificador_gato.empty():
    # Manejo de error si el archivo XML no se carga
    exit()
else:
    print("Clasificador cargado exitosamente.")
```

* El script intenta cargar el archivo XML del clasificador Haar desde la ruta especificada.
* Se incluye una verificación para asegurar que el clasificador se cargó correctamente. Si el archivo no se encuentra o está corrupto, el programa informará un error y terminará, ya que la carga exitosa del clasificador es fundamental para su funcionamiento.

### 3.3. Funciones de Procesamiento y Detección

Esta sección contiene las funciones principales que encapsulan la lógica de detección:

### `detectar_caras_gatos(frame)`

Esta función toma una imagen (`frame`) como entrada y ejecuta el proceso de detección:

1.  **Conversión a Escala de Grises**: Los clasificadores Haar operan con la intensidad de los píxeles, por lo que la imagen se convierte a escala de grises (`cv2.COLOR_BGR2GRAY`).
2.  **Preprocesamiento de Imagen**:
    * **Ecualización Adaptativa del Histograma (CLAHE)**: Mejora el contraste local de la imagen, lo que es beneficioso en diversas condiciones de iluminación y ayuda a resaltar las características para el clasificador.
    * **Desenfoque Gaussiano (`cv2.GaussianBlur`)**: Aplica un ligero desenfoque para reducir el ruido. Esto puede ayudar a prevenir falsos positivos, aunque un desenfoque excesivo podría eliminar detalles útiles.
3.  **Detección de Caras de Gatos (`clasificador_gato.detectMultiScale`)**: Esta es la función central. Aplica el clasificador Haar sobre la imagen en escala de grises utilizando los `SCALE_FACTOR`, `MIN_NEIGHBORS`, `MIN_SIZE`, y `MAX_SIZE` configurados. Devuelve una lista de rectángulos, donde cada uno representa una cara de gato detectada (`[x, y, ancho, alto]`).
4.  **Dibujar Detecciones**: Para cada cara de gato detectada, se dibuja un rectángulo verde y se añade el texto "Gato" sobre él, proporcionando una representación visual de las detecciones en el `frame` original.
5.  **Retorno de Frame**: La función devuelve el `frame` con los rectángulos y textos dibujados.

#### `procesar_imagenes_en_carpeta()`

Esta función está dedicada al procesamiento por lotes de imágenes estáticas:

* **Preparación de la Carpeta de Salida**: La lógica para eliminar y crear la carpeta `OUTPUT_FOLDER` (`output_images`) ha sido comentada en el script. Esto se debe a que, en un entorno Docker, la gestión de estas carpetas (particularmente su creación y eliminación) es mejor manejada por el sistema host al montar volúmenes, para evitar problemas de permisos.
* **Recorrido de Archivos**: Itera sobre todos los archivos en la `INPUT_FOLDER`. Solo procesa archivos con extensiones comunes de imagen (`.png`, `.jpg`, `.jpeg`).
* **Carga y Procesamiento**: Carga cada imagen, invoca a `detectar_caras_gatos` para la detección y guarda la imagen resultante (con las detecciones marcadas) en la `OUTPUT_FOLDER`, utilizando un prefijo `detectado_`.
* **Retroalimentación**: Imprime mensajes en la consola para informar sobre el progreso y la finalización del procesamiento.

#### `procesar_webcam()`

Esta función gestiona la detección de caras de gatos en tiempo real utilizando la webcam:

* **Inicialización de Webcam**: Intenta abrir la webcam predeterminada del sistema (`cv2.VideoCapture(0)`). En caso de fallo (webcam no conectada, en uso, etc.), imprime un error y finaliza.
* **Bucle de Captura**: Entra en un bucle continuo que lee fotogramas de la webcam.
* **Detección y Visualización**: Para cada fotograma leído, llama a `detectar_caras_gatos` y luego muestra el fotograma procesado en una ventana de OpenCV (`cv2.imshow`).
* **Salida del Bucle**: El bucle se mantiene activo hasta que el usuario presiona la tecla 'q'.
* **Liberación de Recursos**: Al salir, se liberan los recursos de la webcam (`cap.release()`) y se cierran todas las ventanas de OpenCV (`cv2.destroyAllWindows()`).

### 3.4. Ejecución Principal del Programa

Este bloque (`if __name__ == "__main__":`) es el punto de entrada del script:

* **Mensaje de Inicio**: Imprime un mensaje de bienvenida al usuario.
* **Selección de Modo**: Solicita al usuario que elija entre procesar imágenes de una carpeta (opción `1`) o utilizar la webcam (opción `2`).
* **Llamada a Función**: Basándose en la elección del usuario, invoca a la función `procesar_imagenes_en_carpeta()` o `procesar_webcam()` respectivamente.
* **Validación de Entrada**: El programa repite la pregunta hasta que se ingresa una opción válida.
* **Mensaje de Finalización**: Imprime un mensaje al concluir la ejecución del programa.

### 4. Limitaciones del Detector Haar Cascade

Aunque eficientes y adecuados para muchas aplicaciones, los clasificadores Haar Cascade tienen ciertas limitaciones a considerar:

* **Sensibilidad a la Pose**: El clasificador `haarcascade_frontalcatface_extended.xml` está optimizado principalmente para caras frontales. Detecciones desde ángulos, perfiles o poses variadas pueden ser inconsistentes o fallar.
* **Sensibilidad a la Iluminación y Oclusión**: Las variaciones drásticas en la iluminación (sombras, contraluces) o la oclusión parcial de la cara del gato pueden afectar negativamente la precisión de la detección.
* **Compromiso Precisión/Recall**: Siempre existe un equilibrio. Los parámetros `SCALE_FACTOR` y `MIN_NEIGHBORS` permiten ajustar este balance entre la cantidad de gatos detectados correctamente (`recall`) y la minimización de detecciones falsas (`precisión`).

Para aplicaciones que demandan una mayor invariancia a la pose o una robustez superior en condiciones muy diversas, las técnicas más avanzadas basadas en **Deep Learning** (como YOLO o SSD) son generalmente más adecuadas, aunque implican una mayor complejidad y requerimientos computacionales.

### Documentación de Pruebas Unitarias: `test_mascotas_haar.py`

Este archivo, `test_mascotas_haar.py`, contiene un conjunto de **pruebas unitarias** diseñado para verificar la funcionalidad principal de la aplicación de detección de gatos (`mascotas_haar.py`). Las pruebas unitarias son una práctica esencial en el desarrollo de software, ya que permiten asegurar que cada componente individual del código funcione como se espera.

### Explicación del `Dockerfile`

El `Dockerfile` es la **receta** fundamental que Docker utiliza para construir la imagen de tu aplicación. Define el entorno, las dependencias y la configuración necesarios para que el proyecto de se ejecute de manera consistente en cualquier lugar que soporte Docker, encapsulándolo en un "contenedor".

A continuación, se desglosa cada instrucción de este `Dockerfile`:

### 1. **Imagen Base**

```dockerfile
# Usa una imagen base de Python adecuada para OpenCV
FROM python:3.9-slim-buster
```

* **`FROM python:3.9-slim-buster`**: Esta es la instrucción inicial y crucial. Indica a Docker que comience a construir la imagen a partir de una **imagen base** preexistente.
    * `python:3.9-slim-buster`: Especifica que se utilizará una imagen oficial de **Python versión 3.9**.
    * `slim-buster`: Designa una versión "ligera" (`slim`) de Python, construida sobre la distribución **Debian 10 "Buster"**. Las imágenes `slim` son preferibles por su menor tamaño y mayor eficiencia, aunque a menudo requieren la instalación manual de dependencias de sistema adicionales.

### 2. **Directorio de Trabajo**

```dockerfile
# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app
```

* **`WORKDIR /app`**: Esta instrucción crea un directorio llamado `/app` dentro del contenedor y lo establece como el **directorio de trabajo actual**. Todas las instrucciones `COPY` y `RUN` subsiguientes se ejecutarán relativas a este directorio, a menos que se especifique una ruta absoluta. Esto organiza los archivos de la aplicación dentro del contenedor.

### 3. **Instalación de Dependencias del Sistema (Linux)**

```dockerfile
# Instala las dependencias necesarias para OpenCV
# Incluyendo libGL, libGLib2.0 y las dependencias básicas de Qt para la GUI y XCB.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libfontconfig1 \
        libxi6 \
        libxrandr2 \
        libxfixes3 \
        libxcursor1 \
        libxdamage1 \
        libnss3 \
        libgdk-pixbuf2.0-0 \
        libgtk-3-0 \
        libpangocairo-1.0-0 \
        libcups2 \
        libusb-1.0-0 \
        libdbus-1-3 \
        libxcb-xkb1 \
        libxkbcommon-x11-0 \
        libxcb-render-util0 \
        libxcb-sync1 \
        libxcb-xfixes0 \
        libxcb-shape0 \
        libxcb-keysyms1 \
        libxcb-randr0 && \
    rm -rf /var/lib/apt/lists/*
```

* **`RUN`**: Ejecuta comandos dentro del contenedor, similar a como lo harías en una terminal de Linux.
* **`apt-get update && apt-get install -y --no-install-recommends ...`**:
    * `apt-get update`: Actualiza la lista de paquetes disponibles en los repositorios de Debian. Es un paso inicial crítico para asegurar que se instalen las últimas versiones de los paquetes.
    * `apt-get install -y --no-install-recommends`: Instala los paquetes de sistema especificados.
        * `-y`: Responde automáticamente "sí" a cualquier solicitud de confirmación durante la instalación.
        * `--no-install-recommends`: Evita la instalación de paquetes "recomendados" que no son estrictamente necesarios, contribuyendo a un tamaño de imagen Docker más reducido.
    * **Lista de Paquetes**: Esta extensa lista incluye **librerías de bajo nivel esenciales para el correcto funcionamiento de OpenCV en entornos Linux**, especialmente para la visualización de interfaces gráficas (GUI) y el acceso a hardware como la webcam.
        * Incluye librerías para renderizado gráfico (`libgl1-mesa-glx`), utilidades fundamentales (`libglib2.0-0`), y una colección de dependencias para el **sistema gráfico X11, la biblioteca Qt (que OpenCV utiliza internamente para la GUI), GStreamer (para manejo multimedia y de cámara), y otras utilidades comunes de Linux**. Estas fueron las dependencias que se identificaron y resolvieron para habilitar la ventana de la webcam.
* **`&& rm -rf /var/lib/apt/lists/*`**: Este comando, encadenado al anterior, limpia el caché de `apt-get` después de la instalación. Es una práctica recomendada para **minimizar el tamaño final de la imagen Docker**, ya que los archivos de caché no son necesarios una vez que los paquetes están instalados.

### 4. **Copia de Requerimientos e Instalación de Dependencias de Python**

```dockerfile
# Copia requirements.txt para aprovechar el caché de Docker
COPY requirements.txt .

# Instala las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt
```

* **`COPY requirements.txt .`**: Copia el archivo `requirements.txt` (que lista las librerías de Python como `opencv-python` y `numpy`) desde tu máquina local (el directorio donde se encuentra el `Dockerfile`) al directorio de trabajo (`/app`) dentro del contenedor. Se copia en un paso separado para **aprovechar el caché de Docker**: si `requirements.txt` no cambia, Docker no reinstalará estas dependencias en construcciones posteriores, acelerando el proceso.
* **`RUN pip install --no-cache-dir -r requirements.txt`**: Ejecuta el gestor de paquetes de Python (`pip`) para instalar todas las librerías listadas en `requirements.txt`.
    * `--no-cache-dir`: Evita que `pip` almacene en caché los archivos descargados de los paquetes, lo que contribuye a mantener la imagen más pequeña.

### 5. **Copia del Código de la Aplicación**

```dockerfile
# Copia el resto de los archivos de tu aplicación
COPY . .
```

* **`COPY . .`**: Esta instrucción copia **todos los demás archivos y carpetas** de tu directorio local (el `.` de la izquierda, que representa "el directorio actual") al directorio de trabajo (`/app`) dentro del contenedor (el `.` de la derecha). Esto incluye tu script `mascotas_haar.py`, los clasificadores Haar (`.xml`), y la carpeta `input_images`.

### 6. **Punto de Entrada del Contenedor**

```dockerfile
# Comando principal para ejecutar la aplicación
ENTRYPOINT ["python", "mascotas_haar.py"]
```

* **`ENTRYPOINT ["python", "mascotas_haar.py"]`**: Define el comando principal que se ejecutará cada vez que se inicie un contenedor a partir de esta imagen.
    * Cuando ejecutas `docker run deteccion-gatos:latest`, Docker simplemente ejecutará `python mascotas_haar.py` dentro del contenedor.
    * A diferencia de `CMD`, `ENTRYPOINT` es el comando fijo; cualquier argumento adicional que pases a `docker run` se añadirá al final de este `ENTRYPOINT`. Esta configuración es ideal para tu aplicación, ya que siempre se inicia el script principal.

### 1. Introducción a `unittest`

El `unittest` es el **framework estándar de Python para escribir pruebas unitarias**. Proporciona una estructura para crear "clases de prueba" y "métodos de prueba" que examinan el comportamiento del código de tu aplicación.

* **`import unittest`**: Importa la funcionalidad necesaria del framework.
* **`class TestMascotasHaar(unittest.TestCase):`**: Define una clase de prueba. Para que `unittest` la detecte y ejecute, debe heredar de `unittest.TestCase`. Dentro de esta clase, cada método cuyo nombre comience con `test_` será reconocido como una prueba individual.

### 2. Preparación para las Pruebas (`setUp` y `tearDown`)

```python
import unittest
import cv2
import os
import numpy as np
from mascotas_haar import clasificador_gato, detectar_caras_gatos, OUTPUT_FOLDER

class TestMascotasHaar(unittest.TestCase):

    def setUp(self):
        """
        Se ejecuta antes de cada método de prueba. Prepara el entorno para las pruebas individuales.
        """
        self.test_image_path = 'input_images/ejemplo.jpg'
        self.imagen_valida = cv2.imread(self.test_image_path)
        if self.imagen_valida is None:
            self.skipTest(f"Imagen de prueba '{self.test_image_path}' no disponible. Asegúrate de que exista y sea accesible.")
```

* **`setUp(self)`**: Este método se ejecuta **antes de cada prueba individual** (es decir, antes de cada método `test_`). Es ideal para inicializar recursos o preparar datos que múltiples pruebas van a necesitar.
    * `self.test_image_path = 'input_images/ejemplo.jpg'`: Define la ruta a una imagen específica que se usará para realizar pruebas de detección.
        **¡Importante!** Para que estas pruebas sean efectivas, el archivo `ejemplo.jpg` debe existir en la carpeta `input_images` y, preferiblemente, contener una cara de gato para validar las detecciones positivas.
    * `self.imagen_valida = cv2.imread(self.test_image_path)`: Carga la imagen de prueba utilizando OpenCV.
    * `if self.imagen_valida is None: self.skipTest(...)`: Esta es una comprobación robusta. Si la imagen de prueba no se encuentra o no se puede cargar, las pruebas que dependen de ella se **saltarán** (`self.skipTest`), en lugar de fallar. Esto ayuda a distinguir problemas de configuración del entorno de fallos en el código de la aplicación.

## 3. Métodos de Prueba Individuales (`test_...`)

Cada método que comienza con `test_` representa una prueba específica. Estos métodos utilizan los métodos de aserción (`self.assertEqual`, `self.assertTrue`, `self.assertIsNotNone`, etc.) proporcionados por `unittest.TestCase` para verificar si una condición se cumple o no. Si una aserción falla, la prueba se marca como fallida.

### `test_clasificador_cargado(self)`

```python
def test_clasificador_cargado(self):
    """Verifica que el clasificador Haar se haya cargado correctamente desde el XML."""
    self.assertFalse(clasificador_gato.empty(), "El clasificador Haar no se cargó correctamente. Verifica la ruta del XML.")
```

* **Propósito**: Asegura que el archivo clasificador Haar (`haarcascade_frontalcatface_extended.xml` o similar) se haya inicializado y cargado correctamente al inicio del script principal (`mascotas_haar.py`).
* **Aserción**: Comprueba que el objeto `clasificador_gato` (importado del script principal) no esté "vacío". Si lo está, indica un problema al cargar el archivo XML.

### `test_imagen_cargada(self)`

```python
def test_imagen_cargada(self):
    """Comprueba que la imagen de prueba se cargue correctamente en memoria."""
    self.assertIsNotNone(self.imagen_valida, "No se pudo cargar la imagen de prueba. Ruta incorrecta o archivo corrupto.")
```

* **Propósito**: Confirma que la imagen de prueba especificada en `setUp` se haya cargado exitosamente en la memoria.
* **Aserción**: Verifica que `self.imagen_valida` no sea `None`, lo que indicaría un fallo de `cv2.imread()`.

### `test_deteccion_formato_salida(self)`

```python
def test_deteccion_formato_salida(self):
    """Verifica que detectar_caras_gatos devuelve una imagen (np.ndarray) del mismo tamaño."""
    resultados = detectar_caras_gatos(self.imagen_valida.copy())
    self.assertIsInstance(resultados, np.ndarray, "La salida de la función no es un array de NumPy (imagen).")
    self.assertEqual(resultados.shape, self.imagen_valida.shape, "La imagen de salida no tiene el mismo tamaño que la original.")
```

* **Propósito**: Asegura que la función `detectar_caras_gatos` devuelva el tipo de dato esperado (un array de NumPy, que es la representación de imágenes en OpenCV) y que la imagen resultante mantenga las mismas dimensiones que la imagen de entrada.
* **Aserciones**:
    * `self.assertIsInstance`: Confirma que el tipo de la salida es `np.ndarray`.
    * `self.assertEqual`: Compara las dimensiones (alto, ancho, canales de color) de la imagen de entrada y salida, verificando que la función no altere el tamaño de la imagen.

### `test_deteccion_con_funcion_personalizada(self)`

```python
def test_deteccion_con_funcion_personalizada(self):
    """Confirma que la función detectar_caras_gatos siempre devuelve un valor (no None)."""
    procesada = detectar_caras_gatos(self.imagen_valida.copy())
    self.assertIsNotNone(procesada, "La función detectar_caras_gatos regresó None, lo cual no es esperado.")
```

* **Propósito**: Una prueba básica para confirmar que la función `detectar_caras_gatos` siempre retorna un objeto de imagen válido y no `None`, lo que podría indicar un error interno o un camino de código no manejado.
* **Aserción**: Verifica que el valor devuelto no sea `None`.

### `test_creacion_output_folder(self)`

```python
def test_creacion_output_folder(self):
    """Verifica que la carpeta de salida (OUTPUT_FOLDER) pueda ser creada si no existe."""
    if os.path.exists(OUTPUT_FOLDER):
        try:
            os.rmdir(OUTPUT_FOLDER)  # Intenta eliminar si está vacía
        except OSError:
            pass  # Ignora si no está vacía, el objetivo es probar la creación
    os.makedirs(OUTPUT_FOLDER, exist_ok=True) # Crea la carpeta, si ya existe no lanza error
    self.assertTrue(os.path.exists(OUTPUT_FOLDER), "No se pudo crear la carpeta de salida. Verifique permisos.")
```

* **Propósito**: Asegura que la aplicación tiene la capacidad de crear la carpeta designada para las imágenes de salida (`output_images`) si esta no existe previamente.
* **Lógica**:
    * Intenta limpiar la carpeta `output_images` si existe y está vacía. Se incluye un bloque `try-except` para manejar casos donde la carpeta no esté vacía y no pueda ser eliminada, ya que la prueba se centra en la capacidad de *crear*.
    * `os.makedirs(OUTPUT_FOLDER, exist_ok=True)`: Intenta crear la carpeta. `exist_ok=True` previene un error si la carpeta ya existe.
* **Aserción**: Confirma que la carpeta `output_images` realmente existe después de intentar su creación.

### `test_cantidad_caras_detectadas(self)`

```python
def test_cantidad_caras_detectadas(self):
    """Verifica que el proceso de detección se ejecute y devuelva un número válido de caras."""
    gray = cv2.cvtColor(self.imagen_valida, cv2.COLOR_BGR2GRAY)
    caras = clasificador_gato.detectMultiScale(gray)
    self.assertGreaterEqual(len(caras), 0, "La cantidad de caras detectadas debe ser cero o más. El detector no debe fallar.")
```

* **Propósito**: Esta prueba verifica que la función principal de detección (`detectMultiScale`) se ejecuta sin errores y produce un resultado válido (una lista de detecciones que puede estar vacía si no se encuentran gatos).
* **Lógica**: Convierte la imagen de prueba a escala de grises y llama directamente al método `detectMultiScale` del clasificador.
* **Aserción**: `self.assertGreaterEqual(len(caras), 0, ...)`: Afirma que la lista de detecciones tiene una longitud de cero o más. Esta es una prueba de que el detector no se "rompe" y siempre devuelve una lista válida.
    * **Nota**: Para una prueba más estricta y significativa (si la imagen `ejemplo.jpg` contuviera un gato conocido), se podría usar `self.assertGreater(len(caras), 0, ...)` para asegurar que al menos un gato es detectado, o incluso verificar las coordenadas específicas de la detección.

### 4. Ejecución de las Pruebas

```python
if __name__ == '__main__':
    unittest.main()
```

* Este bloque estándar de Python permite que, cuando se ejecuta el archivo `test_mascotas_haar.py` directamente desde la terminal, `unittest.main()` descubra y ejecute automáticamente todas las pruebas definidas en la clase `TestMascotasHaar`.
* Al finalizar, mostrará un resumen detallado del número de pruebas ejecutadas, cuántas pasaron (`OK`), cuántas fallaron (`FAIL`) y cuántas se saltaron (`SKIPPED`).

### Cómo Usar este Archivo de Pruebas

1.  Asegúrarse de que existe el archivo de imagen llamado `ejemplo.jpg` dentro de la carpeta `input_images` en la raíz del proyecto.
2. Abre una terminal en ese directorio raíz de tu proyecto.
3. Ejecuta el script de pruebas con el siguiente comando:
    ```bash
    python test_mascotas_haar.py
    ```

### Explicación del Archivo de Clasificador Haar Cascade: `haarcascade_frontalcatface.xml`

### ¿Qué es un Archivo `.xml` de Clasificador Haar Cascade?

Los archivos `.xml` de clasificadores Haar Cascade, como `haarcascade_frontalcatface.xml`, **no son código ejecutable**. En su lugar, son **archivos de datos** que contienen un modelo de aprendizaje automático pre-entrenado. La biblioteca **OpenCV** carga y utiliza internamente estos archivos para realizar la detección de objetos específicos en imágenes o transmisiones de video.

En esencia, este archivo es el resultado de un proceso de entrenamiento computacional donde un algoritmo ha sido "enseñado" a reconocer patrones visuales.

### Propósito de `haarcascade_frontalcatface.xml`

Este archivo específico contiene los patrones aprendidos y las reglas de decisión para detectar **caras frontales de gatos**.

El proceso de entrenamiento involucra:
1.  **Colección de Datos:** Se utilizan miles de imágenes: "positivas" (que contienen caras de gatos) y "negativas" (que no las contienen).
2.  **Identificación de Características Haar:** El algoritmo aprende a identificar **características Haar**. Estas son patrones rectangulares simples que detectan cambios de contraste en la imagen (por ejemplo, el contraste entre la zona oscura de los ojos y la zona más clara del puente de la nariz).
3.  **Construcción de la Cascada:** Las características se organizan en una "cascada" (una serie de etapas de decisión). Cada etapa es un clasificador simple que descarta rápidamente las regiones de la imagen que claramente no son el objeto buscado. Solo las regiones que pasan todas las etapas de la cascada se consideran una detección válida.

### Nota sobre las Características Haar (`haarcascade_frontalcatface.xml` vs. `_extended`)

Este clasificador, `haarcascade_frontalcatface.xml`, utiliza el **conjunto básico de características Haar**, lo que implica **características horizontales y verticales, pero NO características diagonales**.

Aunque es efectivo, la versión `_extended.xml` (si se utiliza) es generalmente más robusta y precisa porque incorpora también las características diagonales, lo que le permite capturar patrones visuales más complejos.

## Contenido Conceptual del Archivo XML

Aunque es un archivo de datos complejo, conceptualmente, si abrieras y examinaras su contenido XML, encontrarías una estructura jerárquica que describe el modelo:

* El elemento raíz `<cascade>` encapsula toda la definición del clasificador.
* Múltiples elementos `<stage>`: Cada uno representa un paso secuencial en la cascada de detección. Para que una región sea clasificada como "cara de gato", debe pasar exitosamente por todas estas etapas.
    * Dentro de cada `<stage>`, hay elementos `<feature>` que describen los patrones de las características Haar que se buscan (sus posiciones relativas, tamaños y ponderaciones).
    * Elementos `<tree>` que representan árboles de decisión. Estos árboles usan los valores de las características para determinar si una región debe avanzar a la siguiente etapa o ser descartada.
* **Umbrales y Pesos:** Valores numéricos que son fundamentales para el proceso de decisión en cada etapa, indicando la fuerza de las características y los criterios para pasar al siguiente nivel de la cascada.

En esencia, este archivo es una representación matemática muy compacta y eficiente del "conocimiento" que el clasificador ha adquirido sobre cómo se ve una cara de gato frontal.

## Uso en el Código Python (`mascotas_haar.py`)

En el script principal del proyecto, `mascotas_haar.py`, este archivo XML se carga utilizando la biblioteca OpenCV de la siguiente manera:

```python
import cv2

# ... (otras configuraciones) ...

HAARCASCADE_PATH = 'haarcascade_frontalcatface.xml' # O 'haarcascade_frontalcatface_extended.xml'
clasificador_gato = cv2.CascadeClassifier(HAARCASCADE_PATH)

# ... (uso posterior de clasificador_gato.detectMultiScale) ...
```

Al ejecutar `cv2.CascadeClassifier(HAARCASCADE_PATH)`, se le indica a OpenCV que cargue el modelo pre-entrenado desde el archivo XML especificado. Una vez cargado, el objeto `clasificador_gato` se utiliza con la función `detectMultiScale` para encontrar y marcar caras de gatos en imágenes o fotogramas de video, basándose en los patrones y reglas aprendidos y almacenados en este archivo XML.

### Explicación del Archivo de Clasificador Haar Cascade: `haarcascade_frontalcatface_extended.xml`

### Propósito de `haarcascade_frontalcatface_extended.xml`

Este archivo específico contiene los patrones y las reglas de decisión para detectar **caras frontales de gatos**.

El proceso de entrenamiento implica:
1.  **Colección de Datos:** Se utilizan miles de imágenes: "positivas" (que contienen caras de gatos) y "negativas" (que no las contienen).
2.  **Extracción de Características Haar:** El algoritmo aprende a identificar **características Haar**, que son patrones rectangulares simples (áreas claras y oscuras) que son distintivas de las caras de gatos (por ejemplo, el contraste entre los ojos y el puente de la nariz).
3.  **Construcción de la Cascada:** Estas características se organizan en una "cascada", una serie de etapas. Cada etapa es un clasificador simple que rápidamente descarta las regiones de la imagen que no se parecen en nada a una cara de gato. Solo las regiones que pasan todas las etapas de la cascada son consideradas una detección válida.

### Diferencia Clave: Versión "Extended"

La característica más importante de la versión `_extended.xml` es que utiliza el **conjunto COMPLETO de características Haar**, lo que incluye:

* **Características horizontales**
* **Características verticales**
* **¡Características diagonales!**

A diferencia de clasificadores Haar más básicos (como `haarcascade_frontalcatface.xml`), que solo usan características horizontales y verticales, la inclusión de características diagonales permite a este clasificador `_extended` capturar patrones visuales más complejos y sutiles. Esto a menudo se traduce en una **detección más robusta y precisa**, con menos falsos positivos.
