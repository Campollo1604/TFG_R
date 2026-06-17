#Fusión ICP de dos nubes de puntos segmentadas con YOLO y DAV2

from pathlib import Path
import cv2
import numpy as np
import torch
import open3d as o3d
from ultralytics import YOLO
from transformers import pipeline
from PIL import Image
import time

BASE_PRO = Path(__file__).resolve().parent.parent.parent
IMG1_RUTA = BASE_PRO / "data" / "raw" / "tsukuba_l.png"
IMG2_RUTA = BASE_PRO / "data" / "raw" / "tsukuba_r.png"
PLY_SALIDA = BASE_PRO / "data" / "output" / "fusion_icp.ply"
NUBE1_SALIDA = BASE_PRO / "data" / "output" / "nube1.ply"
NUBE2_SALIDA = BASE_PRO / "data" / "output" / "nube2.ply"

MODELO_YOLO = "yolov8n-seg.pt"
MODELO_DA = "depth-anything/Depth-Anything-V2-Small-hf"
RESOLUCION = (520, 520)
N_FEATURES = 10000
FACTOR_ESCALA = 1

def cargar_imagenes(ruta1, ruta2, tam =RESOLUCION):
    if not ruta1.exists() or not ruta2.exists():
        raise FileNotFoundError("No se encontraron las imagenes.")
    img1 = cv2.imread(str(ruta1), cv2.IMREAD_COLOR)
    img2 = cv2.imread(str(ruta2), cv2.IMREAD_COLOR)
    return cv2.resize(img1, tam), cv2.resize(img2, tam)

def cargar_modelos(modelo_yolo =MODELO_YOLO, modelo_da=MODELO_DA):
    device = 0 if torch.cuda.is_available() else -1
    yolo = YOLO(modelo_yolo)
    pipe_depth = pipeline(task="depth-estimation", model=modelo_da, device=device)
    #print("Modelos cargados correctamente.")
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

def inferir_profundidad(pipe_depth, img, h,w):
    out = pipe_depth(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    mapa = np.array(out["depth"], dtype=np.float32)
    return cv2.resize(mapa, (w, h), interpolation=cv2.INTER_CUBIC)

def emparejar(img1, img2, mask1, mask2, n_features=N_FEATURES, ratio=0.75):
    orb = cv2.ORB_create(nfeatures=n_features)
    gris1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gris2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1_all, des1_all = orb.detectAndCompute(gris1, None)
    kp2_all, des2_all = orb.detectAndCompute(gris2, None)

    def filtrar_kp(kp, des, mask):
        indices = [i for i, k in enumerate(kp)
                   if mask[int(k.pt[1]), int(k.pt[0])] == 1]
        return [kp[i] for i in indices], des[indices]

    kp1, des1 = filtrar_kp(kp1_all, des1_all, mask1)
    kp2, des2 = filtrar_kp(kp2_all, des2_all, mask2)

    print(f"Kps en objeto img1: {len(kp1)} | img2: {len(kp2)}")
 
    indice = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
    busqueda = dict(checks=50)
    flann = cv2.FlannBasedMatcher(indice, busqueda)
    knn_matches = flann.knnMatch(des1, des2, k=2)
    mejores = [m for m, n in knn_matches if m.distance < ratio * n.distance]

    print(f"Matches validos: {len(mejores)}")

    if len(mejores) < 6:
        raise RuntimeError("No hay suficientes matches")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in mejores])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in mejores])

    return pts1, pts2

def estimar_pose(pts1, pts2,h, w):
    focal = 0.8 * w
    K = np.array([
        [focal,     0, w / 2],
        [    0, focal, h / 2],
        [    0,     0,     1]
    ], dtype=np.float64)

    E, mask_E = cv2.findEssentialMat(
        pts1, pts2, K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    pts1_in = pts1[mask_E.ravel() == 1]
    pts2_in = pts2[mask_E.ravel() == 1]

    print(f"Inliers RANSAC: {len(pts1_in)} / {len(pts1)}")

    _, R, t, _ = cv2.recoverPose(E, pts1_in, pts2_in, K)
    return R, t, K, pts1_in, pts2_in

def calcular_escala(pts1_in , pts2_in, mapa1,K, R, t):
    h, w = mapa1.shape

    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))
    p4D = cv2.triangulatePoints(P1, P2, pts1_in.T, pts2_in.T)
    p3D = (p4D[:3] / p4D[3]).T

    mask_z = p3D[:, 2] > 0
    p3D = p3D[mask_z]
    pts1_val = pts1_in[mask_z]

    u_idx = np.clip(pts1_val[:, 0].astype(int), 0, w - 1)
    v_idx = np.clip(pts1_val[:, 1].astype(int), 0, h - 1)

    z_depth = mapa1[v_idx, u_idx]
    z_triang = p3D[:, 2]

    mask_escala = (z_depth > 0) & (z_triang > 0)
    if np.sum(mask_escala) < 2:
        raise RuntimeError("No se puede calcular la escala.")

    escala = np.median(z_triang[mask_escala] / (z_depth[mask_escala] + 1e-6))
    print(f"Escala calculada: {escala:.6f}")
    return escala

def construir_nube(mapa, mask, img, escala, h, w, focal):
    v, u = np.where(mask == 1)
    z = mapa[v, u] * escala
    mask_z = z > 0
    v, u, z = v[mask_z], u[mask_z], z[mask_z]

    x = (u - w / 2) * z / focal
    y = -(v - h / 2) * z / focal

    pts = np.vstack((x, y, z)).T
    colores = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[v, u] / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colores)
    #pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.5)

    print(f"Puntos en la nube: {len(pcd.points)}")
    return pcd

def alinear_icp(pcd2, pcd1):
    reg1 = o3d.pipelines.registration.registration_icp(
        pcd2, pcd1, 0.3, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=300)
    )
    pcd2.transform(reg1.transformation)

    reg2 = o3d.pipelines.registration.registration_icp(
        pcd2, pcd1, 0.2, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=300)
    )
    pcd2.transform(reg2.transformation)

    print(f"ICP fitness: {reg2.fitness:.4f} | RMSE: {reg2.inlier_rmse:.4f}")
    return pcd2, reg2.fitness, reg2.inlier_rmse

def guardar_ply(pcd, ruta_salida):
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(ruta_salida), pcd)
    print(f"Guardado: {ruta_salida}")

def ver_nube(pcd):
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Nube de ptos",
        width=1200,
        height=800
    )

def main():
    tiempo_0 = time.time()
    img1, img2 = cargar_imagenes(IMG1_RUTA, IMG2_RUTA)
    h, w = img1.shape[:2]
    focal = 0.8 * w

    yolo, pipe_depth = cargar_modelos()

    tiempo_1 = time.time()
    res1 = yolo(img1, conf= 0.1, verbose= False)[0]
    res2 = yolo(img2, conf= 0.1, verbose= False)[0]

    mask1 = extraer_mascara(res1, h, w)
    mask2 = extraer_mascara(res2, h, w)

    if mask1 is None or mask2 is None:
        raise RuntimeError("YOLO no detectó el objeto en ambas imagenes.")
    tiempo_2 = time.time()
    print(f"Segmentacion YOLO: {tiempo_2 - tiempo_1:.3f} s")

    mapa1 = inferir_profundidad(pipe_depth, img1, h, w)
    mapa2 = inferir_profundidad(pipe_depth, img2, h, w)
    tiempo_3 = time.time()
    print(f"Estimación de profundidad: {tiempo_3 - tiempo_2:.3f} s")

    pts1, pts2 = emparejar(img1, img2, mask1, mask2)
    R, t, K, pts1_in, pts2_in = estimar_pose(pts1, pts2, h, w)

    escala = calcular_escala(pts1_in, pts2_in, mapa1, K, R, t)

    pcd1 = construir_nube(mapa1, mask1, img1, escala, h, w, focal)
    pcd2 = construir_nube(mapa2, mask2, img2, escala, h, w, focal)
    tiempo_4 = time.time()
    print(f"Odometría y construcción de nubes: {tiempo_4 - tiempo_3:.3f} s")

    T_init = np.eye(4)
    T_init[:3, :3] = R
    T_init[:3, 3] = t.ravel() * escala * FACTOR_ESCALA
    pcd2.transform(T_init)

    pcd2, fitness, rmse = alinear_icp(pcd2, pcd1)
    tiempo_5 = time.time()
    print(f"Fusión ICP: {tiempo_5 - tiempo_4:.3f} s")

    fusion = pcd1 + pcd2
    fusion, _ = fusion.remove_statistical_outlier(nb_neighbors =30,std_ratio=1.5)

    guardar_ply(pcd1, NUBE1_SALIDA)
    guardar_ply(pcd2, NUBE2_SALIDA)
    guardar_ply(fusion, PLY_SALIDA)

    tiempo_6 = time.time()
    print(f"\nTiempo total: {tiempo_6 - tiempo_0:.3f} s")
    print(f"\nPuntos totales: {len(fusion.points)}")

    ver_nube(fusion)

if __name__ == "__main__":
    main()