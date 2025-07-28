from flask import Flask, render_template, Response
import cv2
import time
import psutil
import os

app = Flask(__name__)

# Ruta completa al clasificador entrenado
CASCADE_PATH = '/home/romero/Descargas/cascade.xml'

# Limitar hilos usados por OpenCV
cv2.setNumThreads(2)

# Cargar clasificador
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    raise IOError(f"[ERROR] No se pudo cargar el clasificador: {CASCADE_PATH}")

# Captura de video desde IP Webcam
video = cv2.VideoCapture('http://192.168.17.101:8080/video')

# Inicializar procesos
process = psutil.Process(os.getpid())
total_bytes_sent = 0

def generate_frames():
    global total_bytes_sent
    prev_time = time.time()

    while True:
        success, frame = video.read()
        if not success or frame is None:
            continue

        # Redimensionar
        frame = cv2.resize(frame, (320, 240))

        # Iniciar tiempo de frame
        start_time = time.time()

        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detección de rostros
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(60, 60)
        )

        # Aplicar desenfoque a cada rostro detectado
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            blurred = cv2.GaussianBlur(face_roi, (25, 25), 15)
            frame[y:y+h, x:x+w] = blurred

        # Calcular métricas
        end_time = time.time()
        fps = 1.0 / (end_time - start_time + 1e-6)
        mem_mb = process.memory_info().rss / 1024 / 1024
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        total_bytes_sent += len(frame_bytes)
        mb_sent = total_bytes_sent / (1024 * 1024)

        # Mostrar métricas en la imagen
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Mem: {mem_mb:.1f} MB", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Sent: {mb_sent:.1f} MB", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Codificar frame actualizado
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
