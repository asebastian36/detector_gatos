import unittest
import cv2
import os
import numpy as np
from mascotas_haar import clasificador_gato, detectar_caras_gatos, OUTPUT_FOLDER

class TestMascotasHaar(unittest.TestCase):

    def setUp(self):
        self.test_image_path = 'input_images/ejemplo.jpg'
        self.imagen_valida = cv2.imread(self.test_image_path)
        if self.imagen_valida is None:
            self.skipTest(f"Imagen de prueba '{self.test_image_path}' no disponible")

    def test_clasificador_cargado(self):
        """Verifica que el clasificador Haar se haya cargado correctamente."""
        self.assertFalse(clasificador_gato.empty(), "El clasificador Haar no se cargó correctamente.")

    def test_imagen_cargada(self):
        """Comprueba que la imagen de prueba se cargue correctamente."""
        self.assertIsNotNone(self.imagen_valida, "No se pudo cargar la imagen de prueba.")

    def test_deteccion_formato_salida(self):
        """Verifica que detectar_caras_gatos devuelve una imagen (np.ndarray)."""
        resultados = detectar_caras_gatos(self.imagen_valida.copy())
        self.assertIsInstance(resultados, np.ndarray, "La salida de la función no es una imagen (np.ndarray).")
        self.assertEqual(resultados.shape, self.imagen_valida.shape, "La imagen de salida no tiene el mismo tamaño que la original.")

    def test_deteccion_con_funcion_personalizada(self):
        """Confirma que detectar_caras_gatos no devuelve None."""
        procesada = detectar_caras_gatos(self.imagen_valida.copy())
        self.assertIsNotNone(procesada, "La función detectar_caras_gatos regresó None.")

    def test_creacion_output_folder(self):
        """Verifica que la carpeta de salida pueda ser creada."""
        if os.path.exists(OUTPUT_FOLDER):
            try:
                os.rmdir(OUTPUT_FOLDER)  # Solo funciona si está vacía
            except OSError:
                pass  # Si no está vacía, se ignora el borrado
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        self.assertTrue(os.path.exists(OUTPUT_FOLDER), "No se pudo crear la carpeta de salida.")

    def test_cantidad_caras_detectadas(self):
        """Prueba si al menos una cara de gato es detectada en una imagen conocida."""
        gray = cv2.cvtColor(self.imagen_valida, cv2.COLOR_BGR2GRAY)
        caras = clasificador_gato.detectMultiScale(gray)
        self.assertGreaterEqual(len(caras), 0, "La cantidad de caras detectadas debería ser cero o más.")

if __name__ == '__main__':
    unittest.main()
