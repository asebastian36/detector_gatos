# Usa una imagen base de Python adecuada para OpenCV
FROM python:3.9-slim-buster

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instala las dependencias necesarias para OpenCV
# Incluyendo libGL, libGLib2.0 y las dependencias básicas de Qt para la GUI y XCB.
# Intentaremos con un conjunto de paquetes que suelen funcionar.
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

# Copia requirements.txt para aprovechar el caché de Docker
COPY requirements.txt .

# Instala las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de los archivos de tu aplicación
COPY . .

# Comando principal para ejecutar la aplicación
ENTRYPOINT ["python", "mascotas_haar.py"]