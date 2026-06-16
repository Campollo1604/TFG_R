#Estimamos la matriz esencial usando RANSAC

from pathlib import Path
import cv2
import numpy as np

#Rutas del proyecto
BASE_PRO = Path(__file__).resolve().parent.parent.parent
IMG1_RUTA = BASE_PRO / "data" / "raw" / "tsukuba_l.png"
IMG2_RUTA = BASE_PRO / "data" / "raw" / "tsukuba_r.png"
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

    #Emparejamiento bruto 
    bf_bruto = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_brutos = bf_bruto.match(des1, des2)

    #KNN + ratio test
    bf_knn = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = bf_knn.knnMatch(des1, des2, k=2)
    mejores = [m for m, n in knn_matches if m.distance < ratio * n.distance]

    print(f"Matches brutos: {len(matches_brutos)} | filtrados: {len(mejores)}")
    return kp1, kp2, des1, des2, gris1, matches_brutos, mejores


def estimar_pose(kp1, kp2, mejores, gris1, ransac_th=2.0):
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

    return R, t, mask, K

def mostrar_matches(img1, img2, kp1, kp2, brutos, mejores, mask):
    vis_brutos = cv2.drawMatches(img1, kp1, img2, kp2, brutos, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    vis_filtrados = cv2.drawMatches(img1, kp1, img2, kp2, mejores, None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    #Solo los inliers tras RANSAC
    inliers = [mejores[i] for i in range(len(mejores)) if mask[i] == 1]
    inliers = sorted(inliers, key=lambda m: m.distance)
    vis_inliers = cv2.drawMatches(img1, kp1, img2, kp2, inliers, None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.namedWindow("Brutos", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Filtrados", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Inliers", cv2.WINDOW_NORMAL)
    cv2.imshow("Brutos", vis_brutos)
    cv2.imshow("Filtrados", vis_filtrados)
    cv2.imshow("Inliers", vis_inliers)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    img1, img2 = cargar_imagenes(IMG1_RUTA, IMG2_RUTA)
    kp1, kp2, des1, des2, gris1, brutos, mejores = emparejar(img1, img2)
    R, t, mask, K = estimar_pose(kp1, kp2, mejores, gris1)
    mostrar_matches(img1, img2, kp1, kp2, brutos, mejores, mask)

if __name__ == "__main__":
    main()