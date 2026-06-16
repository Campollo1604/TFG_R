#Busca el matching de los kps de dos imágenes mediante emparejamiento bruto y filtrado KNN + ratio

from pathlib import Path
import cv2

#Rutas del proyecto
BASE_PRO = Path(__file__).resolve().parent.parent.parent
IMG1_RUTA = BASE_PRO / "data" / "raw" / "edificio.jpg"
IMG2_RUTA = BASE_PRO / "data" / "raw" / "edificiogirado.jpg"
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
    return kp1, kp2, matches_brutos, mejores

def main():
    img1, img2 = cargar_imagenes(IMG1_RUTA, IMG2_RUTA)
    kp1, kp2, brutos, mejores = emparejar(img1, img2)

    #Visualizamos los dos tipos de matches para comparar
    vis_brutos = cv2.drawMatches(img1, kp1, img2, kp2, brutos, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    vis_filtrados = cv2.drawMatches(img1, kp1, img2, kp2, mejores, None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.namedWindow("Brutos", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Filtrados", cv2.WINDOW_NORMAL)
    cv2.imshow("Brutos", vis_brutos)
    cv2.imshow("Filtrados", vis_filtrados)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()