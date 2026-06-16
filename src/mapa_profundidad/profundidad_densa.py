#Genera mapa de profundidad usando DAV2

from pathlib import Path
import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import open3d as o3d

BASE_PRO = Path(__file__).resolve().parent.parent.parent
IMAGEN_RUTA = BASE_PRO / "data" / "raw" / "edificio.jpg"
PLY_SALIDA = BASE_PRO / "data" / "output" / "nube_profundidad.ply"
MODELO = "depth-anything/Depth-Anything-V2-Small-hf"

def cargar_modelo(modelo_id = MODELO):
    processor = AutoImageProcessor.from_pretrained(modelo_id)
    modelo = AutoModelForDepthEstimation.from_pretrained(modelo_id)
    modelo.eval()
    print("Modelo cargado correctamente.")
    return processor, modelo

def cargar_imagen(ruta, resolucion = (520, 520)):
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró la imagen en: {ruta}")

    imagen = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    imagen_redime = cv2.resize(imagen_rgb, resolucion)
    imagen_pil = Image.fromarray(imagen_redime)

    return imagen, imagen_rgb, imagen_redime, imagen_pil

def inferir_profundidad(processor, modelo, imagen_pil, imagen_rgb):
    inputs = processor(images=imagen_pil, return_tensors="pt")

    with torch.no_grad():
        outputs = modelo(**inputs)
        prediccion = outputs.predicted_depth

    #Convertir al tamaño original con interpolación
    prediccion = torch.nn.functional.interpolate(
        prediccion.unsqueeze(1),
        size=imagen_rgb.shape[:2],
        mode="bicubic",
        align_corners=False
    ).squeeze()

    mapa_profundidad = prediccion.cpu().numpy()

    return mapa_profundidad

def construir_nube_puntos(mapa_profundidad, imagen, paso=6,
                           z_min=0.05, z_max=0.95,  #Variar dependiendo de lo restrictivo que se sea
                           umbral_gradiente=0.15):
    h, w = mapa_profundidad.shape
    f = 0.8 * w
    cx, cy = w / 2, h / 2

    p_bajo = np.percentile(mapa_profundidad,0)
    p_alto = np.percentile(mapa_profundidad, 100)
    mapa_norm = np.clip(mapa_profundidad, p_bajo, p_alto)
    mapa_norm = (mapa_norm - p_bajo) / (p_alto - p_bajo + 1e-8)

    coords_u = np.arange(0, w, paso)
    coords_v = np.arange(0, h, paso)
    mapa_u, mapa_v = np.meshgrid(coords_u, coords_v)
    u = mapa_u.flatten()
    v = mapa_v.flatten()

    #Filtro de rango
    mask_rango = (mapa_norm > z_min) & (mapa_norm < z_max)

    #Filtro de gradiente
    dy, dx = np.gradient(mapa_norm) #numpy devuelve dy primero
    gradiente = np.sqrt(dx**2 + dy**2)
    mask_gradiente = gradiente < umbral_gradiente

    mask_final = mask_rango & mask_gradiente
    filtrados = mask_final[v, u]

    u_final = u[filtrados]
    v_final = v[filtrados]
    Z_final = mapa_norm[v_final, u_final]

    X = (u_final - cx) * Z_final / f
    Y = -(v_final - cy) * Z_final / f

    puntos = np.stack((X, Y, Z_final), axis=-1)
    colores = imagen[v_final, u_final]

    print(f"Puntos tras filtro de rango: {np.sum(mask_rango)}")
    print(f"Puntos tras filtro de gradiente: {np.sum(mask_gradiente)}")
    print(f"Puntos totales: {len(puntos)}")

    return puntos, colores

def guardar_ply(puntos, colores, ruta_salida):
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_salida, "w") as fply:
        fply.write("ply\nformat ascii 1.0\n")
        fply.write(f"element vertex {len(puntos)}\n")
        fply.write("property float x\nproperty float y\nproperty float z\n")
        fply.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fply.write("end_header\n")
        for (X, Y, Z), (b, g, r) in zip(puntos, colores):
            fply.write(f"{X} {Y} {Z} {r} {g} {b}\n")
    print(f"Guardado en: {ruta_salida}")

def ver_nube(puntos, colores):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(puntos)
    pcd.colors = o3d.utility.Vector3dVector(colores[:, ::-1] / 255.0)  
    
    print("Abriendo visor 3D...")
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Nube de ptos",
        width=800,
        height=600
    )

def visualizar(imagen_redime, mapa_profundidad, resolucion = (520, 520)):
    profundidad_visu = cv2.normalize(
        mapa_profundidad, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    ver_imagen = cv2.resize(imagen_redime, resolucion)
    ver_profundidad = cv2.resize(profundidad_visu, resolucion)

    cv2.namedWindow("Imagen original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mapa de profundidad", cv2.WINDOW_NORMAL)
    cv2.imshow("Imagen original", ver_imagen)
    cv2.imshow("Mapa de profundidad", ver_profundidad)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    processor, modelo = cargar_modelo()
    imagen, imagen_rgb, imagen_redime, imagen_pil = cargar_imagen(IMAGEN_RUTA)
    mapa_profundidad = inferir_profundidad(processor, modelo, imagen_pil, imagen_rgb)
    visualizar(imagen_redime, mapa_profundidad)
    puntos, colores = construir_nube_puntos(mapa_profundidad, imagen)
    guardar_ply(puntos, colores, PLY_SALIDA)
    ver_nube(puntos, colores)

if __name__ == "__main__":
    main()