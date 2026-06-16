#Detecta y visualiza los kps de una imagen con ORB

from pathlib import Path
import cv2

#Ruta del proyecto y de lsa fotos
BASE_PRO = Path(__file__).resolve().parent.parent.parent
IMAGEN_RUTA = BASE_PRO / "data" / "raw" / "edificio.jpg"


def detectar_keypoints(ruta, n_features=2000):
    
    img = cv2.imread(str(ruta))
    if img is None:
        raise RuntimeError("No se pudo cargar la imagen")

    #Pasamos a gris porque ORB trabaja mejor asi
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=n_features)
    kp, des = orb.detectAndCompute(gris, None)

    print(f"Kps detectados: {len(kp)}")
    return img, kp, des

def main():
    img, kp, _ = detectar_keypoints(IMAGEN_RUTA)

    #Pintamos los kps encima de la imagen original en color verde
    img_kp = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

    cv2.namedWindow("ORB Keypoints", cv2.WINDOW_NORMAL)
    cv2.imshow("ORB Keypoints", img_kp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()