# 🩻 Microservicio de Evaluación de Radiografías (CNN + GradCAM)

Este proyecto ofrece un microservicio de inferencia de Visión por Computadora listo para producción. Utiliza una red neuronal convolucional (YOLOv11-cls) para clasificar radiografías y genera mapas de calor (Grad-CAM) para explicar la decisión del modelo.

## 🎯 Características Principales

- Arquitectura CNN SOTA: Utiliza YOLOv11 (You Only Look Once) optimizado para clasificación de imágenes médicas.

- Explicabilidad (XAI): Genera mapas de calor visuales que resaltan las regiones donde el modelo detectó anomalías.

- Alta Performance: Pipeline optimizado con PyTorch y OpenCV, con medición detallada de latencia.

- API Estandarizada: Endpoints documentados automáticamente con Swagger UI.

- Dockerizado: Entorno reproducible basado en Python 3.11 Slim con soporte para librerías gráficas.

## 🏁 Guía de Construcción

### Paso 1: Preparación de Artefactos

- Asegúrate de que el modelo entrenado .pt se encuentre en la ruta correcta:
`src/models/YOLO/xrays_evaluation_model_medium_v1.pt`

### Paso 2: Construcción de la Imagen Docker

- Desde el directorio raíz del repositorio (ingeniia_services/), ejecuta:

  ```bash
  docker build -t genia/xrays-evaluation-cnn:1.0 -f container-images/xrays_evaluation/Dockerfile .
  ```

### Paso 3: Ejecutar el Contenedor

- Una vez construida la imagen:

  ```bash
  docker run -d -p 8080:8080 --name xrays-service genia/xrays-evaluation-cnn:1.0
  ```

### Paso 4: Verificar Funcionamiento

- Accede a la documentación interactiva:
`http://localhost:8080/docs`

## 📝 Cómo Usar la API

El endpoint principal es /cnn_xray_demo.

- Ejemplo de Solicitud (cURL)
  - Debes enviar la imagen codificada en Base64.

    ```bash
    curl -X 'POST' \
      'http://localhost:8080/cnn_xray_demo' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "image_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDA..."
    }'
    ```


- Respuesta Exitosa Esperada

  - El servicio retorna la predicción, la explicabilidad (imagen overlay en base64) y los tiempos de ejecución.

    ```bash
    {
      "prediction": {
        "label": "Anomaly",
        "confidence": 0.985,
        "class_id": 0
      },
      "explainability": {
        "heatmap_base64": "...",
        "overlay_base64": "...",
        "description": "Red indicates high attention regions."
      },
      "performance": {
        "preprocess_time_ms": 12.5,
        "inference_time_ms": 15.2,
        "explainability_time_ms": 8.1,
        "total_latency_ms": 35.8,
        "model_used": "YOLO11m-cls"
      }
    ```
