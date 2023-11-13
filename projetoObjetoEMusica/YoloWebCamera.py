import numpy as np
import cv2
import time
import winsound

from csv import DictWriter

camera = cv2.VideoCapture(1)
h, w = None, None

with open('yoloDados/YoloNames.names') as f:
    labels = [line.strip() for line in f]

network = cv2.dnn.readNetFromDarknet('yoloDados/yolov3.cfg', 'yoloDados/yolov3.weights')
layers_names_all = network.getLayerNames()
layers_names_output = [layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()]

probability_minimum = 0.5
threshold = 0.3
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

# Frequências aproximadas para as notas do tema do Pac-Man
pacman_theme_notes = [440, 440, 523, 440, 587, 659, 523, 587, 698]

with open('teste.csv', 'w') as arquivo:
    cabecalho = ['Detectado', 'Acuracia']
    escritor_csv = DictWriter(arquivo, fieldnames=cabecalho)
    escritor_csv.writeheader()

    pacman_index = 0  # Inicializa o índice para o tema do Pac-Man
    while True:
        _, frame = camera.read()

        if w is None or h is None:
            h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        network.setInput(blob)
        start = time.time()
        output_from_network = network.forward(layers_names_output)
        end = time.time()

        print('Tempo gasto atual {:.5f} segundos'.format(end - start))

        bounding_boxes = []
        confidences = []
        class_numbers = []

        for result in output_from_network:
            for detected_objects in result:
                scores = detected_objects[5:]
                class_current = np.argmax(scores)
                confidence_current = scores[class_current]

                if confidence_current > probability_minimum:
                    box_current = detected_objects[0:4] * np.array([w, h, w, h])
                    x_center, y_center, box_width, box_height = box_current

                    # Utiliza as notas do tema do Pac-Man
                    freq = pacman_theme_notes[pacman_index % len(pacman_theme_notes)]
                    duration = 300  # 0.3 segundos para cada beep
                    winsound.Beep(freq, duration)
                    pacman_index += 1

                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))

                    bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                    confidences.append(float(confidence_current))
                    class_numbers.append(class_current)

        results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)

        if len(results) > 0:
            for i in results.flatten():
                x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
                colour_box_current = colours[class_numbers[i]].tolist()
                cv2.rectangle(frame, (x_min, y_min), (x_min + box_width, y_min + box_height),
                              colour_box_current, 2)
                text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])], confidences[i])
                cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)
                escritor_csv.writerow(
                    {"Detectado": text_box_current.split(':')[0], "Acuracia": text_box_current.split(':')[1]})
                print(text_box_current.split(':')[0] + " - " + text_box_current.split(':')[1])

        cv2.imshow('YOLO v3 WebCamera', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

camera.release()
cv2.destroyAllWindows()
