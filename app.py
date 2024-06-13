import cv2
import numpy as np
import threading

cascPath = "haarcascade_frontalface_default.xml"
eyePath = "haarcascade_eye.xml"
sideFacePath = "haarcascade_profileface.xml"

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + eyePath)
side_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + sideFacePath)

# Verificar se os classificadores foram carregados corretamente
if face_cascade.empty() or eye_cascade.empty() or side_face_cascade.empty():
    print("Erro: Não foi possível carregar os classificadores Haar Cascade.")
    exit()

# Abrir o vídeo
cap = cv2.VideoCapture('video/mourinho.mp4')
if not cap.isOpened():
    print("Erro: Não foi possível abrir o arquivo de vídeo.")
    exit()

cap_lock = threading.Lock()

while True:
    ret, img = cap.read()
    if not ret:
        break
    
    # Converter a imagem para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detectar faces frontais
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Detectar faces de perfil
    side_faces = side_face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Desenhar retângulos ao redor das faces frontais detectadas e detectar olhos
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    
    # Desenhar retângulos ao redor das faces de perfil detectadas
    for (x, y, w, h) in side_faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Exibir o resultado
    cv2.imshow('img', img)
    
    # Esperar pela tecla 'Esc' para sair
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Liberar os recursos
cap.release()
cv2.destroyAllWindows()
