import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import numpy as np
import time
import os

def draw_scan_lines(frame, step=30, color=(0, 0, 255), thickness=1):
    height, width, _ = frame.shape
    for x in range(0, width, step):
        cv2.line(frame, (x, 0), (x, height), color, thickness)
    for y in range(0, height, step):
        cv2.line(frame, (0, y), (width, y), color, thickness)
    return frame

def select_camera():
    index = 0
    arr = []
    i = 10
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        index += 1
        i -= 1

    if len(arr) == 0:
        print("No se encontraron cámaras.")
        return -1

    print("Seleccione la cámara a utilizar:")
    for i, cam in enumerate(arr):
        print(f"{i}: Cámara {cam}")

    cam_index = int(input("Ingrese el número de la cámara: "))
    if cam_index < 0 or cam_index >= len(arr):
        print("Selección inválida.")
        return -1

    return arr[cam_index]

def main():
    # Seleccionar la cámara
    cam_index = select_camera()
    if cam_index == -1:
        return

    # Inicializar la cámara
    cap = cv2.VideoCapture(cam_index)

    # Configurar los tiempos de visualización
    start_time = time.time()
    scan_lines_duration = 2  # duración de las líneas de escaneo en segundos
    welcome_duration = 7  # duración del mensaje de bienvenida en segundos

    # Rutas absolutas de los archivos de configuración y pesos
    config_file = os.path.abspath(os.path.join("yolo", "yolov4.cfg"))
    weights_file = os.path.abspath(os.path.join("yolo", "yolov4.weights"))

    # Verificar que los archivos existen
    if not os.path.exists(config_file):
        print(f"Archivo de configuración no encontrado: {config_file}")
        return
    if not os.path.exists(weights_file):
        print(f"Archivo de pesos no encontrado: {weights_file}")
        return

    # Cargar el modelo YOLOv4
    net = cv2.dnn.readNetFromDarknet(config_file, weights_file)
    output_layers = net.getUnconnectedOutLayersNames()

    # Etiquetas de clases COCO
    class_labels = [
        'persona', 'bici', 'coche', 'moto', 'avión', 'autobús', 'tren', 'camión',
        'barco', 'semáforo', 'hidrante', 'señal', 'aparcamiento', 'banco', 'pájaro',
        'gato', 'perro', 'caballo', 'oveja', 'vaca', 'elefante', 'oso', 'cebra', 
        'jirafa', 'mochila', 'paraguas', 'bolsa', 'corbata', 'maleta', 'frisbee',
        'esquí', 'snowboard', 'pelota', 'cometa', 'plato', 'taza', 'tenedor', 'cuchillo',
        'cuchara', 'tazón', 'plátano', 'manzana', 'sándwich', 'naranja', 'brócoli', 
        'zanahoria', 'hot dog', 'pizza', 'dona', 'pastel', 'silla', 'sofá', 'planta', 
        'cama', 'mesa de comedor', 'inodoro', 'TV', 'portátil', 'ratón', 'control remoto', 
        'teclado', 'teléfono', 'microondas', 'horno', 'tostadora', 'fregadero', 'nevera', 
        'libro', 'reloj', 'vaso', 'tijera', 'cartapacio', 'cajonera', 'monitor', 'escritorio', 
        'tobogán', 'ratón', 'paño', 'teclado', 'linterna', 'ventana', 'alfombra', 'carretera', 
        'paso de peatones', 'valla', 'tráfico', 'luces', 'cartelera', 'placa de la calle', 
        'nieve', 'montaña', 'cielo', 'río', 'lago', 'charco', 'carretera', 'piedra', 'puente', 
        'edificio', 'hogar', 'tienda', 'parada de autobús', 'callejón', 'calle', 'park', 
        'garaje', 'puente', 'ferrocarril', 'farola', 'stop', 'hidrante', 'extintor', 'banco',
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time < scan_lines_duration:
            # Mostrar líneas de escaneo
            frame = draw_scan_lines(frame)
        elif elapsed_time < (scan_lines_duration + welcome_duration):
            # Mostrar mensaje de bienvenida
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = 'Bienvenido'
            text_size = cv2.getTextSize(text, font, 2, 3)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = (frame.shape[0] + text_size[1]) // 2
            cv2.putText(frame, text, (text_x, text_y), font, 2, (0, 0, 255), 3)
        else:
            # Realizar la detección de objetos
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            # Procesar las detecciones
            class_ids = []
            confidences = []
            boxes = []
            height, width, _ = frame.shape

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = class_labels[class_ids[i]]
                    confidence = confidences[i]
                    color = (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Mostrar el frame con los objetos detectados
        cv2.imshow('Object Detection', frame)

        # Salir del bucle si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la captura de la cámara y cerrar todas las ventanas
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
