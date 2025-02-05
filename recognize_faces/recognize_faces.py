import cv2
import numpy as np
import os
from mtcnn import MTCNN
from keras_facenet import FaceNet

# Inicializa a detecção e reconhecimento facial
detector = MTCNN()
embedder = FaceNet()

# Diretório onde as imagens serão salvas
IMG_DIR = "img"
os.makedirs(IMG_DIR, exist_ok=True)

# Margem base de pixels ao redor do rosto
BASE_MARGIN = 30

def save_face(name, face_img):
    """Salva a imagem do rosto em um arquivo .jpg"""
    filename = os.path.join(IMG_DIR, f"{name}.jpg")
    cv2.imwrite(filename, face_img)
    print(f"Rosto salvo como {filename}")

def load_faces():
    """Carrega e retorna embeddings dos rostos conhecidos a partir das imagens salvas."""
    known_faces = {}
    for file in os.listdir(IMG_DIR):
        if file.endswith(".jpg"):
            name = os.path.splitext(file)[0]
            img_path = os.path.join(IMG_DIR, file)
            img = cv2.imread(img_path)
            img = preprocess_image(img, (160, 160))
            embedding = embedder.embeddings(img)[0]
            known_faces[name] = embedding
    return known_faces

def preprocess_image(img, size):
    """Redimensiona e expande as dimensões da imagem para ser usada no modelo."""
    img = cv2.resize(img, size)
    return np.expand_dims(img, axis=0)

def adjust_box_with_dynamic_margin(box, frame_shape):
    """Ajusta as coordenadas da caixa do rosto com margem dinâmica e dentro dos limites da imagem."""
    x, y, w, h = box
    dynamic_margin = int(min(w, h) * 0.2)  # Aumenta ou diminui a margem dinamicamente com base no tamanho do rosto
    x = max(0, x - dynamic_margin)
    y = max(0, y - dynamic_margin)
    w = min(w + 2 * dynamic_margin, frame_shape[1] - x)
    h = min(h + 2 * dynamic_margin, frame_shape[0] - y)
    return x, y, w, h

def enhance_lighting(img):
    """Aprimora a iluminação da imagem usando equalização de histograma."""
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])  # Equaliza apenas o canal de luminância
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def process_frame(frame):
    """Redimensiona o frame e melhora a iluminação para aumentar a área de captura."""
    frame = cv2.resize(frame, (0, 0), fx=1.2, fy=1.2)
    return enhance_lighting(frame)

def capture_face():
    """Captura e salva rostos detectados."""
    cap = cv2.VideoCapture(0)
    
    # Ajuste a resolução da webcam para melhorar o desempenho
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Largura da imagem
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Altura da imagem
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = process_frame(frame)
        faces = detector.detect_faces(frame)
        
        if faces:
            for face in faces:
                x, y, w, h = adjust_box_with_dynamic_margin(face['box'], frame.shape)
                
                # Desenha o retângulo ao redor do rosto
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            cv2.imshow("Capture Face", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r') and faces:
            name = input("Digite o nome da pessoa: ")
            x, y, w, h = adjust_box_with_dynamic_margin(faces[0]['box'], frame.shape)
            face_area = frame[y:y+h, x:x+w]
            save_face(name, face_area)
            
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def recognize_faces():
    """Reconhece rostos em tempo real comparando com rostos salvos."""
    cap = cv2.VideoCapture(0)
    
    # Ajuste a resolução da webcam para melhorar o desempenho
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Largura da imagem
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Altura da imagem
    
    known_faces = load_faces()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = process_frame(frame)
        faces = detector.detect_faces(frame)
        
        if faces:
            for face in faces:
                x, y, w, h = adjust_box_with_dynamic_margin(face['box'], frame.shape)
                face_img = preprocess_image(frame[y:y+h, x:x+w], (160, 160))
                embedding = embedder.embeddings(face_img)[0]

                best_match, min_dist = find_best_match(embedding, known_faces)
                label = best_match if min_dist < 0.9 else "Desconhecido"
                
                # Desenha o retângulo e o nome
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow("Recognize Faces", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def find_best_match(embedding, known_faces):
    """Encontra o melhor rosto correspondente ao embedding fornecido."""
    best_match = None
    min_dist = float("inf")
    
    for name, saved_embedding in known_faces.items():
        dist = np.linalg.norm(saved_embedding - embedding)
        if dist < min_dist:
            min_dist = dist
            best_match = name
            
    return best_match, min_dist

# Para capturar rostos, execute:
# capture_face()

# Para reconhecer em tempo real, execute:
# recognize_faces()
