# TFG: Reconstrucción 3D mediante Triangulación e IA
Trabajo de Fin de Grado - Grado en Ingeniería Aeroespacial  
Escola de Enxeñaría Aeronáutica e do Espazo - Universidade de Vigo  
Autor: Pablo Gómez Campollo

# Descripción
Este repositorio contiene el código desarrollado para el TFG "Drones e IA 
para la reconstrucción de escenarios". El sistema integra técnicas clásicas 
de visión por computador con modelos de inteligencia artificial para la 
reconstrucción tridimensional de escenarios y objetos a partir de imágenes 
monoculares.

El proyecto se estructura en tres módulos principales:

- Odometría visual: estimación del movimiento relativo de la cámara 
  mediante detección de características ORB, emparejamiento FLANN, 
  geometría epipolar y por último triangulación
- Estimación de profundidad: generación de nubes de puntos densas 
  a partir de imágenes individuales mediante el modelo Depth Anything V2.
- Reconstrucción orientada a objetos: segmentación semántica con 
  YOLOv8-seg, estimación de profundidad y fusión de nubes de puntos 
  mediante ICP.

# Estructura
```
TFG_R/
├── data/
│   ├── raw/          #Imágenes de entrada
│   └── output/       #Resultados .png / .ply
├── src/
│   ├── odometria/
│   │   ├── deteccion_orb.py
│   │   ├── emparejamiento_features.py
│   │   ├── odometria_visual.py
│   │   └── triangulacion.py
│   ├── mapa_profundidad/
│   │   ├── profundidad_densa.py
│   │   └── fusion_profundidad.py
│   └── segmentacion/
│       ├── segmentacion_vista_individual.py
│       ├── segmentacion_fusion_icp.py
│       └── segmentacion_fusion_video.py
├── .gitignore
├── README.md
└── requirements.txt
```

# Instalación
```bash
git clone https://github.com/usuario/TFG_R.git
cd TFG_R
pip install -r requirements.tx
```

# Imágenes de prueba
Coloca las imágenes que deseas utilizar en `data/raw/`. Cada script indica en la primera
sección qué imágenes espera encontrar.

# Resultados
Las nubes de puntos se guardan en formato PLY en `data/output/` 
y pueden visualizarse con Meshlab o mediante la vista
interactiva de Open3D que se abre al final de cada script.

# Tecnologías utilizadas
- OpenCV: Visión por computador clásica
- Depth Anything V2: Estimación de profundidad monocular
- YOLOv8: Detección y segmentación semántica
- Open3D: Procesamiento y visualización de nubes de puntos
- PyTorch: Framework de aprendizaje profundo