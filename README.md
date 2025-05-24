# Proyecto: Detección de objetos usando características Haar

La finalidad de este proyecto es reconocer gatos con vision por computadora ya sea por medio de imagenes o con imagenes o video proporcionados por webcam.

## Teoria basica

Las características de Haar son filtros basados en patrones rectangulares utilizados en visión por computadora, especialmente en el algoritmo de detección de objetos de Viola-Jones. Fueron introducidas en 2001 por Paul Viola y Michael Jones para detectar rostros en imágenes en tiempo real, pero su aplicación se ha extendido a otros objetos (como coches, señales de tráfico, etc.).

## Cómo Funcionan

**Pasos del Algoritmo Viola-Jones:**

1. Extracción de características:

Se aplican miles de características de Haar sobre la imagen en diferentes escalas.

Cada característica devuelve un valor numérico:
Valor=(Suma de pıˊxeles en aˊrea blanca)−(Suma de pıˊxeles en aˊrea negra)
Valor=(Suma de pıˊxeles en aˊrea blanca)−(Suma de pıˊxeles en aˊrea negra)

2. Selección de características relevantes:

Se usa AdaBoost para elegir las características más discriminativas (ej: las que mejor separan rostros de no rostros).

3. Clasificación en cascada:

Una serie de clasificadores débiles se combinan para formar un clasificador fuerte.

Si una región de la imagen no pasa una etapa, se descarta inmediatamente (ahorra tiempo de procesamiento).

> Enlace al modelo ya entrenado [Modelo de entrenamiento](https://github.com/opencv/opencv/tree/master/data/haarcascades)

## Características de Haar aplicadas al reconocimiento de rostros de gatos

Las características de Haar, originalmente diseñadas para detección de rostros humanos, pueden adaptarse eficazmente para reconocer rostros de gatos gracias a patrones similares en su estructura facial (ojos, nariz, bigotes y orejas).

**¿Por qué Haar funciona para gatos?**

Los gatos comparten rasgos clave con los humanos que las características de Haar pueden capturar:

* **Ojos:** Regiones oscuras en un área clara (pelaje).

* **Bigotes:** Líneas verticales u horizontales.

* **Orejas:** Triángulos oscuros sobre fondo claro.

* **Nariz:** Pequeña región oscura entre los ojos.

Estos patrones se detectan mediante filtros rectangulares que miden contrastes de intensidad.

### **¿Cómo funciona exactamente el modelo `haarcascade_frontalcatface` de OpenCV?**

El modelo **`haarcascade_frontalcatface.xml`** incluido en OpenCV sigue el **mismo principio de las características de Haar y el algoritmo Viola-Jones**, pero está **específicamente entrenado para detectar rostros de gatos frontales**. Aquí te explico cómo funciona y en qué se diferencia de un detector humano:

---

## Entrenamiento del Modelo para Gatos

Fue entrenado con:

- **Miles de imágenes positivas**: Rostros frontales de gatos (de diferentes razas, colores y edades).
- **Imágenes negativas**: Fondos sin gatos o partes del cuerpo no relevantes.

### Proceso de entrenamiento

1. **Extracción de características de Haar**: Se aplicaron filtros rectangulares para capturar patrones típicos de gatos (orejas triangulares, bigotes, ojos verticales).

2. **Selección con AdaBoost**: Se eligieron las ~1,000 características más discriminativas.

3. **Clasificación en cascada**: Se organizaron en etapas para descartar rápidamente zonas no relevantes.

---

### Estructura del Modelo

El archivo `.xml` contiene:
- **Configuración de la cascada**: Número de etapas y características por etapa.
- **Parámetros de cada filtro Haar**: Posición, tamaño y umbrales de decisión.
- **Valores de AdaBoost**: Ponderaciones de las características seleccionadas.

---

### ¿Cómo lo usa OpenCV?

Cuando llamas a `detectMultiScale()`, OpenCV realiza estos pasos:

1. **Preprocesamiento**:
   - Convierte la imagen a escala de grises.
   - Calcula la **imagen integral** (para acelerar las sumas de píxeles).

2. **Detección multiescala**:
   - Escanea la imagen con ventanas de diferentes tamaños (para cubrir gatos cerca/lejos).

3. **Aplicación de la cascada**:
   - En cada ventana, aplica las etapas del clasificador:
     - Si una ventana falla en cualquier etapa, se descarta.
     - Solo las ventanas que pasan todas las etapas se consideran "rostros de gato".

4. **Supresión de solapamientos**:
   - Fusiona detecciones redundantes en un solo rectángulo.

---

### Limitaciones del Modelo

- **Ángulo frontal**: No detecta perfiles o rostros girados.
- **Tamaño mínimo**: Gatos muy pequeños pueden pasar desapercibidos.
- **Falsos positivos**: Patrones similares (como orejas de otros animales) pueden confundirlo.
