#Fusiona nubes de puntos de dos vistas usando DAV2 y pose relativa

from pathlib import Path
import cv2
import numpy as np
import torch
import open3d as o3d
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image

BASE_PRO = Path(__file__).resolve().parent.parent.parent
IMG1_RUTA = BASE_PRO / "data" / "raw" / "cocherojo1.png"
IMG2_RUTA = BASE_PRO / "data" / "raw" / "cocherojo2.png"
PLY_SALIDA = BASE_PRO / "data" / "output" / "fusion_profundidad.ply"
MODELO = "depth-anything/Depth-Anything-V2-Small-hf"
RESOLUCION = (520, 520)
N_FEATURES = 2000
FACTOR_ESCALA = 10

def cargar_imagenes(ruta1, ruta2):
    if not ruta1.exists() or not ruta2.exists():
        raise FileNotFoundError("No se encontraron las imagenes.")

    img1 = cv2.imread(str(ruta1), cv2.IMREAD_COLOR)
    img2 = cv2.imread(str(ruta2), cv2.IMREAD_COLOR)

    return img1, img2

def cargar_modelo(modelo_id=MODELO):
    processor = AutoImageProcessor.from_pretrained(modelo_id)
    modelo = AutoModelForDepthEstimation.from_pretrained(modelo_id)
    modelo.eval()
    print("Modelo cargado correctamente.")
    return processor, modelo

def inferir_profundidad(processor, modelo, imagen, resolucion=RESOLUCION):
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    imagen_redime = cv2.resize(imagen_rgb, resolucion)
    imagen_pil = Image.fromarray(imagen_redime)

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

    return prediccion.cpu().numpy()

def emparejar(img1, img2, n_features=N_FEATURES, ratio=0.75):
    gris1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gris2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=n_features)
    kp1, des1 = orb.detectAndCompute(gris1, None)
    kp2, des2 = orb.detectAndCompute(gris2, None)

    print(f"Kps img1: {len(kp1)} | img2: {len(kp2)}")

    #FLANN + KNN + ratio test 
    indice = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
    busqueda = dict(checks=50)
    flann = cv2.FlannBasedMatcher(indice, busqueda)
    knn_matches = flann.knnMatch(des1, des2, k=2)
    mejores = [m for m, n in knn_matches if m.distance < ratio * n.distance]

    print(f"Matches filtrados: {len(mejores)}")
    return kp1, kp2, gris1, mejores

def estimar_pose(kp1, kp2, mejores, gris1, ransac_th=1):
    h, w = gris1.shape
    focal = 0.8 * w
    cx, cy = w / 2, h / 2

    K = np.array([
        [focal,     0, cx],
        [    0, focal, cy],
        [    0,     0,  1]
    ], dtype=np.float64)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in mejores])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in mejores])

    E, mask = cv2.findEssentialMat(
        pts1, pts2, K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=ransac_th
    )

    inliers = int(mask.sum())
    print(f"Inliers RANSAC: {inliers} / {len(mask)}")

    pts1_in = pts1[mask.ravel() == 1]
    pts2_in = pts2[mask.ravel() == 1]

    _, R, t, _ = cv2.recoverPose(E, pts1_in, pts2_in, K)

    print(f"\nR:\n{R}")
    print(f"\nt:\n{t}")

    return R, t, K, pts1_in, pts2_in

def calcular_escala(pts1_in, mapa1, puntos3D):
    h, w = mapa1.shape
    u_idx = np.clip(pts1_in[:, 0].astype(int), 0, w - 1)
    v_idx = np.clip(pts1_in[:, 1].astype(int), 0, h - 1)

    Z_depth = mapa1[v_idx, u_idx]
    Z_triang = puntos3D[:, 2]

    mask_Z = (Z_depth > 0) & (Z_triang > 0)
    if np.sum(mask_Z) < 5:
        raise RuntimeError("No hay puntos para calcular la escala.")

    escala = np.median(Z_triang[mask_Z] / (Z_depth[mask_Z] + 1e-6))
    print(f"Escala calculada: {escala:.6f}")

    return escala

def construir_nube_puntos(mapa, imagen, escala, paso=3,
                           z_min=0.03, z_max=0.97, #Variar dependiendo de lo restrictivo que se sea
                           umbral_gradiente=0.15):
    h, w = mapa.shape
    f = 0.8 * w
    cx, cy = w / 2, h / 2

    p_bajo = np.percentile(mapa, 3)
    p_alto = np.percentile(mapa, 97)
    mapa_norm = np.clip(mapa, p_bajo, p_alto)
    mapa_norm = (mapa_norm - p_bajo) / (p_alto - p_bajo + 1e-8)

    coords_u = np.arange(0, w, paso)
    coords_v = np.arange(0, h, paso)
    mapa_u, mapa_v = np.meshgrid(coords_u, coords_v)
    u = mapa_u.flatten()
    v = mapa_v.flatten()

    #Filtro de rango
    mask_rango = (mapa_norm > z_min) & (mapa_norm < z_max)

    #Filtro de gradiente
    dy, dx = np.gradient(mapa_norm)  #numpy devuelve dy primero
    gradiente = np.sqrt(dx**2 + dy**2)
    mask_gradiente = gradiente < umbral_gradiente

    mask_final = mask_rango & mask_gradiente
    filtrados = mask_final[v, u]

    u_final = u[filtrados]
    v_final = v[filtrados]
    Z_final = mapa_norm[v_final, u_final] * escala * FACTOR_ESCALA

    X = (u_final - cx) * Z_final / f
    Y = -(v_final - cy) * Z_final / f

    puntos = np.stack((X, Y, Z_final), axis=-1)
    colores = imagen[v_final, u_final]

    print(f"Puntos tras filtrado: {len(puntos)}")
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

def main():
    img1, img2 = cargar_imagenes(IMG1_RUTA, IMG2_RUTA)
    processor, modelo = cargar_modelo()

    print("Calculando mapas de profundidad...")
    mapa1 = inferir_profundidad(processor, modelo, img1)
    mapa2 = inferir_profundidad(processor, modelo, img2)

    print("\nEstimando pose relativa...")
    kp1, kp2, gris1, mejores = emparejar(img1, img2)
    R, t, K, pts1_in, pts2_in = estimar_pose(kp1, kp2, mejores, gris1)

    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))
    puntos4D = cv2.triangulatePoints(P1, P2, pts1_in.T, pts2_in.T)
    puntos3D = (puntos4D[:3] / puntos4D[3]).T
    mask_z = puntos3D[:, 2] > 0
    puntos3D = puntos3D[mask_z]
    pts1_in = pts1_in[mask_z]

    escala = calcular_escala(pts1_in, mapa1, puntos3D)

    print("\nConstruyendo nubes de puntos...")
    puntos1, colores1 = construir_nube_puntos(mapa1, img1, escala)
    puntos2, colores2 = construir_nube_puntos(mapa2, img2, escala)
    puntos2 = (R @ puntos2.T + t * escala * FACTOR_ESCALA).T

    puntos_totales = np.vstack((puntos1, puntos2))
    colores_totales = np.vstack((colores1, colores2))
    print(f"\nPuntos totales tras fusion: {len(puntos_totales)}")

    guardar_ply(puntos_totales, colores_totales, PLY_SALIDA)
    ver_nube(puntos_totales, colores_totales)

if __name__ == "__main__":
    main()