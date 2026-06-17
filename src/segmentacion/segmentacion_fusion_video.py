#Reconstrucción 3D desde vídeo de dron

from pathlib import Path
import cv2
import numpy as np
import torch
import open3d as o3d
from ultralytics import YOLO
from transformers import pipeline
from PIL import Image
import time
import json

BASE_PRO = Path(__file__).resolve().parent.parent.parent
VIDEO_RUTA = BASE_PRO / "data" / "raw" / "dron_video.mp4"
SALIDA_DIR = BASE_PRO / "data" / "output" / "frames_dron"

MODELO_YOLO = "yolov8n-seg.pt"
MODELO_DA = "depth-anything/Depth-Anything-V2-Small-hf"
RESOLUCION = (520, 520)
N_FEATURES = 10000
FACTOR_ESCALA = 1

SEPARACION_FRAMES = 20   #Nº de fotogramas entre cada par
SEPARACION_TIEMPO = 4.0  #Segundos entre grupos


def extraer_pares_video(video_ruta, salida_dir, separacion_frames, separacion_tiempo, tam):
    salida_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_ruta))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    paso_grupo = int(fps * separacion_tiempo)
    pares = []
    idx_inicio = 0
    grupo = 0

    while idx_inicio + separacion_frames < total_frames:
        idx_a = idx_inicio
        idx_b = idx_inicio + separacion_frames

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx_a)
        ret_a, frame_a = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx_b)
        ret_b, frame_b = cap.read()

        if not ret_a or not ret_b:
            break

        ruta_a = salida_dir / f"grupo{grupo:02d}_a_f{idx_a:05d}.png"
        ruta_b = salida_dir / f"grupo{grupo:02d}_b_f{idx_b:05d}.png"
        cv2.imwrite(str(ruta_a), cv2.resize(frame_a, tam))
        cv2.imwrite(str(ruta_b), cv2.resize(frame_b, tam))

        pares.append((ruta_a, ruta_b))
        idx_inicio += paso_grupo
        grupo += 1

    cap.release()
    print(f"Generados {len(pares)} pares de fotogramas.")
    return pares


def cargar_modelos(modelo_yolo=MODELO_YOLO, modelo_da=MODELO_DA):
    device = 0 if torch.cuda.is_available() else -1
    yolo = YOLO(modelo_yolo)
    pipe_depth = pipeline(task="depth-estimation", model=modelo_da, device=device)
    return yolo, pipe_depth


def extraer_mascara(resultado, h, w):
    if resultado.masks is None:
        return None
    mejor_area = 0
    mejor_mask = None
    for i in range(len(resultado.boxes)):
        m = resultado.masks.data[i].cpu().numpy()
        mask_bin = (m > 0.5).astype(np.uint8)
        mask_resized = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)
        area = mask_resized.sum()
        if area > mejor_area:
            mejor_area = area
            mejor_mask = mask_resized
    return mejor_mask


def inferir_profundidad(pipe_depth, img, h, w):
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

    if len(kp1) < 6 or len(kp2) < 6:
        raise RuntimeError("No hay suficientes keypoints en el objeto.")

    #FLANN + KNN + ratio test 
    indice = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
    busqueda = dict(checks=50)
    flann = cv2.FlannBasedMatcher(indice, busqueda)
    knn_matches = flann.knnMatch(des1, des2, k=2)
    mejores = [m for m, n in knn_matches if m.distance < ratio * n.distance]

    if len(mejores) < 6:
        raise RuntimeError("No hay suficientes matches")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in mejores])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in mejores])

    return pts1, pts2


def estimar_pose(pts1, pts2, h, w):
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

    _, R, t, _ = cv2.recoverPose(E, pts1_in, pts2_in, K)
    return R, t, K, pts1_in, pts2_in


def calcular_escala(pts1_in, pts2_in, mapa1,K, R, t):
    h, w = mapa1.shape

    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))
    p4D = cv2.triangulatePoints(P1, P2, pts1_in.T , pts2_in.T)
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
    return escala


def construir_nube(mapa, mask, img,escala, h, w, focal):
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
    return pcd2, reg2.fitness, reg2.inlier_rmse


def guardar_ply(pcd, ruta_salida):
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(ruta_salida), pcd)


def procesar_par(ruta1, ruta2, yolo, pipe_depth, salida_dir, nombre):
    img1 = cv2.imread(str(ruta1), cv2.IMREAD_COLOR)
    img2 = cv2.imread(str(ruta2), cv2.IMREAD_COLOR)
    h, w = img1.shape[:2]
    focal = 0.8 * w

    t0 = time.time()

    res1 = yolo(img1, conf=0.1, verbose= False)[0]
    res2 = yolo(img2, conf=0.1, verbose= False)[0]
    mask1 = extraer_mascara(res1, h, w)
    mask2 = extraer_mascara(res2, h, w)

    if mask1 is None or mask2 is None:
        raise RuntimeError("YOLO no detectó el objeto en ambas imagenes.")

    mapa1 = inferir_profundidad(pipe_depth, img1, h, w)
    mapa2 = inferir_profundidad(pipe_depth, img2, h, w)

    pts1, pts2 = emparejar(img1, img2, mask1, mask2)
    R, t, K, pts1_in, pts2_in = estimar_pose(pts1, pts2, h, w)
    escala = calcular_escala(pts1_in, pts2_in, mapa1, K, R, t)

    pcd1 = construir_nube(mapa1, mask1, img1, escala, h, w, focal)
    pcd2 = construir_nube(mapa2, mask2, img2, escala, h, w, focal)

    T_init = np.eye(4)
    T_init[:3, :3] = R
    T_init[:3, 3] = t.ravel() * escala * FACTOR_ESCALA
    pcd2.transform(T_init)

    pcd2, fitness, rmse = alinear_icp(pcd2, pcd1)

    fusion = pcd1 + pcd2
    fusion, _ = fusion.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)

    guardar_ply(fusion, salida_dir / f"{nombre}_fusion.ply")

    return {
        "par": nombre,
        "fitness": float(fitness),
        "rmse": float(rmse),
        "puntos_totales": len(fusion.points),
        "tiempo_s": round(time.time() - t0, 3)
    }


def main():
    pares = extraer_pares_video(VIDEO_RUTA, SALIDA_DIR, SEPARACION_FRAMES, SEPARACION_TIEMPO, RESOLUCION)
    yolo, pipe_depth = cargar_modelos()

    resultados = []
    for idx, (ruta1, ruta2) in enumerate(pares):
        nombre = f"grupo{idx:02d}"
        print(f"\nProcesando {nombre}: {ruta1.name} - {ruta2.name}")
        try:
            r = procesar_par(ruta1, ruta2, yolo, pipe_depth, SALIDA_DIR, nombre)
            print(f"  Fitness: {r['fitness']:.4f} | RMSE: {r['rmse']:.4f} | Tiempo: {r['tiempo_s']:.2f} s")
            resultados.append(r)
        except RuntimeError as e:
            print(f"  Descartado: {e}")
            resultados.append({"par": nombre, "error": str(e)})

    with open(SALIDA_DIR / "resultados.json", "w") as f:
        json.dump(resultados, f, indent=2)

    print(f"\nProcesados {len(pares)} pares. Resultados en {SALIDA_DIR / 'resultados.json'}")


if __name__ == "__main__":
    main()