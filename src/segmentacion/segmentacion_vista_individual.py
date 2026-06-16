#Reconstrucción 3D de un objeto individual usando YOLO y DAV2

from pathlib import Path
import cv2
import numpy as np
import torch
import open3d as o3d
from ultralytics import YOLO
from transformers import pipeline
from PIL import Image

BASE_PRO = Path(__file__).resolve().parent.parent.parent
IMAGEN_RUTA = BASE_PRO / "data" / "raw" / "cocherojo1.png"
PLY_SALIDA = BASE_PRO / "data" / "output" / "nube_individual.ply"
MASCARA_SALIDA = BASE_PRO / "data" / "output" / "mascara_objeto.png"
MAPA_SALIDA = BASE_PRO / "data" / "output" / "mapa_profundidad.png"

MODELO_YOLO = "yolov8n-seg.pt"
MODELO_DA = "depth-anything/Depth-Anything-V2-Small-hf"
RESOLUCION = (520, 520)

def cargar_imagen(ruta, tam=RESOLUCION):
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró la imagen en: {ruta}")
    img = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
    return cv2.resize(img, tam)

def cargar_modelos(modelo_yolo=MODELO_YOLO, modelo_da=MODELO_DA):
    device = 0 if torch.cuda.is_available() else -1
    yolo = YOLO(modelo_yolo)
    pipe_depth = pipeline(task="depth-estimation", model=modelo_da, device=device)
    print("Modelos cargados correctamente.")
    return yolo, pipe_depth

def extraer_mascara(resultado, h, w):
    if resultado.masks is None:
        return None
    mejor_area = 0
    mejor_mask = None #En caso de haber más de un objeto, se guarda el de mator tamaño
    for i in range(len(resultado.boxes)):
        m = resultado.masks.data[i].cpu().numpy()
        mask_bin = (m > 0.5).astype(np.uint8)
        mask_resized = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)
        area = mask_resized.sum()
        if area > mejor_area:
            mejor_area = area
            mejor_mask = mask_resized
    return mejor_mask

def guardar_mascara(img, mask, ruta_salida):
    img_mascara = img.copy()
    overlay = img.copy()
    overlay[mask == 1] = [0, 200, 0]
    img_mascara = cv2.addWeighted(img_mascara, 0.6, overlay, 0.4, 0)
    contornos, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(img_mascara, contornos, -1, (0, 255, 0), 2)
    ruta_salida.parent.mkdir(parents= True, exist_ok= True)
    cv2.imwrite(str(ruta_salida), img_mascara)
    return img_mascara

def inferir_profundidad(pipe_depth, img, h, w):
    out = pipe_depth(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    mapa = np.array(out["depth"], dtype=np.float32)
    return cv2.resize(mapa, (w, h), interpolation=cv2.INTER_CUBIC)

def guardar_mapa(mapa, ruta_salida):
    mapa_visu = cv2.normalize(mapa, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    ruta_salida.parent.mkdir(parents = True, exist_ok = True)
    cv2.imwrite(str(ruta_salida), mapa_visu)
    return mapa_visu

def construir_nube(mapa, mask, img, h, w, focal):
    v, u = np.where(mask == 1)
    z = mapa[v, u]
    mask_z = z > 0
    v, u, z = v[mask_z], u[mask_z], z[mask_z]

    x = (u - w / 2) * z /focal
    y = -(v - h / 2) * z /focal

    pts = np.vstack((x, y, z)).T
    colores = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[v, u] / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colores)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors= 50, std_ratio= 0.8)

    print(f"Puntos en la nube: {len(pcd.points)}")
    return pcd

def visualizar(img, img_mascara, mapa_visu, pcd, tam=RESOLUCION):
    img_display = cv2.resize(img, tam)
    mascara_display = cv2.resize(img_mascara, tam)
    prof_display = cv2.resize(mapa_visu, tam)

    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mascara YOLO", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mapa de profundidad", cv2.WINDOW_NORMAL)

    cv2.moveWindow("Original", 0, 100)
    cv2.moveWindow("Mascara YOLO", 660, 100)
    cv2.moveWindow("Mapa de profundidad", 1320, 100)

    cv2.imshow("Original", img_display)
    cv2.imshow("Mascara YOLO", mascara_display)
    cv2.imshow("Mapa de profundidad", prof_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Nube de ptos",
        width=1200,
        height=800
    )

def main():
    img = cargar_imagen(IMAGEN_RUTA)
    h, w = img.shape[:2]
    focal = 0.8 * w

    yolo, pipe_depth = cargar_modelos()

    res = yolo(img, conf= 0.1, verbose =False)[0]
    mask = extraer_mascara(res, h, w)

    if mask is None:
        print("YOLO no detectó nada")
        mask = np.ones((h, w), dtype=np.uint8)
    else:
        print("Objeto detectado correctamente.")

    img_mascara = guardar_mascara(img, mask, MASCARA_SALIDA)

    print("Calculando mapa de profundidad...")
    mapa = inferir_profundidad(pipe_depth,img, h, w)
    mapa_visu = guardar_mapa(mapa, MAPA_SALIDA)

    pcd = construir_nube(mapa, mask, img, h, w,focal)
    PLY_SALIDA.parent.mkdir(parents =True, exist_ok =True)
    o3d.io.write_point_cloud(str(PLY_SALIDA), pcd)
    print(f"Nube guardada en: {PLY_SALIDA}")

    visualizar(img, img_mascara, mapa_visu, pcd)

if __name__ == "__main__":
    main()