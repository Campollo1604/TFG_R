from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
import time

#Rutas del proyecto
BASE_PRO = Path(__file__).resolve().parent.parent.parent
IMG1_RUTA = BASE_PRO / "data" / "raw" / "edificio.jpg"
IMG2_RUTA = BASE_PRO / "data" / "raw" / "edificiogirado.jpg"
PLY_SALIDA = BASE_PRO / "data" / "output" / "nube_tsukuba.ply"
N_FEATURES = 2000

def cargar_imagenes(ruta1, ruta2):
    img1 = cv2.imread(str(ruta1))
    img2 = cv2.imread(str(ruta2))
    if img1 is None or img2 is None:
        raise RuntimeError("No se pudieron cargar las imagenes")
    return img1, img2


def emparejar(img1, img2, n_features=N_FEATURES, ratio=0.75):
    gris1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gris2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=n_features)
    kp1, des1 = orb.detectAndCompute(gris1, None)
    kp2, des2 = orb.detectAndCompute(gris2, None)

    print(f"Kps img1: {len(kp1)} | img2: {len(kp2)}")

    #KNN + ratio test
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des1, des2, k=2)
    mejores = [m for m, n in knn if m.distance < ratio * n.distance]

    print(f"Matches filtrados: {len(mejores)}")
    return kp1, kp2, gris1, mejores

def estimar_pose(kp1, kp2, mejores, gris1, ransac_th=0.2):
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

    E, mask = cv2.findEssentialMat(pts1, pts2, K,
                                    method=cv2.RANSAC,
                                    prob=0.999,
                                    threshold=ransac_th)

    inliers = int(mask.sum())
    print(f"Inliers RANSAC: {inliers} / {len(mask)}")

    pts1_in = pts1[mask.ravel() == 1]
    pts2_in = pts2[mask.ravel() == 1]

    _, R, t, _ = cv2.recoverPose(E, pts1_in, pts2_in, K)

    print(f"\nR:\n{R}")
    print(f"\nt:\n{t}")

    return R, t, K, pts1_in, pts2_in

def triangular(R, t, K, pts1_in, pts2_in, img1, max_dist=1000):
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1)))) #Matrices de proyeccion
    P2 = K @ np.hstack((R, t))

    p4D = cv2.triangulatePoints(P1, P2, pts1_in.T, pts2_in.T) #Coordenadas homogeneas (X,Y,Z,W)
    p3D = (p4D[:3] / p4D[3]).T

    #Cogemos el color de cada punto en la imagen original
    colores = []
    for pto in pts1_in:
        x, y = int(pto[0]), int(pto[1])
        b, g, r = img1[y, x]
        colores.append((r, g, b))
    colores = np.array(colores)

    #Quitamos puntos raros que salen al dividir por W
    mask_fin = np.isfinite(p3D).all(axis=1)
    p3D = p3D[mask_fin]
    colores = colores[mask_fin]

    #Quitamos puntos muy lejanos
    mask_dist = np.linalg.norm(p3D, axis=1) < max_dist
    p3D = p3D[mask_dist]
    colores = colores[mask_dist]

    print(f"Puntos validos: {len(p3D)}")
    return p3D, colores

def guardar_ply(p3D, colores, ruta_salida):
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_salida, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(p3D)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(p3D, colores):
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")
    print(f"Guardado en: {ruta_salida}")

def ver_nube(p3D, colores): #Funcion para la ventana de open3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(p3D)
    pcd.colors = o3d.utility.Vector3dVector(colores / 255.0)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    o3d.visualization.draw_geometries([pcd], window_name="Nube de ptos",
                                       width=800, height=600)


def main():
    img1, img2 = cargar_imagenes(IMG1_RUTA, IMG2_RUTA)
    inicio = time.time()
    kp1, kp2, gris1, mejores = emparejar(img1, img2)
    R, t, K, pts1_in, pts2_in = estimar_pose(kp1, kp2, mejores, gris1)
    fin = time.time()
    print(f"Tiempo: {fin - inicio:.3f} s")
    p3D, colores = triangular(R, t, K, pts1_in, pts2_in, img1)
    guardar_ply(p3D, colores, PLY_SALIDA)
    ver_nube(p3D, colores)


if __name__ == "__main__":
    main()