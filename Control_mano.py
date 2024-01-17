#Librerias
import cv2
import mediapipe as mp
import numpy as np
import pyautogui

#Detectar el dibujo
mp_drawing = mp.solutions.drawing_utils
#Detectar la mano
mp_hands = mp.solutions.hands
#Iniciar el proceso de captura del video.
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#Color para el punto de referencia para mover mause
color_mouse_pointer = (255, 0, 255)
# Puntos de la pantalla-juego
# Punto para X
SCREEN_GAME_X_INI = 150
# Punto para Y
SCREEN_GAME_Y_INI = 160
# X + Ancho
SCREEN_GAME_X_FIN = 150 + 780
# Y + Alto
SCREEN_GAME_Y_FIN = 160 + 450

#Calculo para la relacion de aspecto para contruir un rectangulo para el juego
aspect_ratio_screen = (SCREEN_GAME_X_FIN - SCREEN_GAME_X_INI) / (SCREEN_GAME_Y_FIN - SCREEN_GAME_Y_INI)
print("Relación de aspecto", aspect_ratio_screen)

#Espacio para el rectángulo 
X_Y_INI = 100
#Función para calcular distancias
def calculate_distance(x1, y1, x2, y2):
    #Punto 1
    p1 = np.array([x1, y1])
    #Punto 2
    p2 = np.array([x2, y2])
    #Retorna la distancia entre los dos puntos
    return np.linalg.norm(p1 - p2)

#Funcion para para 
def detect_finger_down(hand_landmarks):
    finger_down = False
    #Colores de las lineas de la mano
    #Linea para la base hacia la muñeca
    color_base = (255, 0, 112)
    #Linea del indice a la muñeca
    color_index = (255, 198, 82)
    #Valores para la base de la muñeca posicion 0
    x_base1 = int(hand_landmarks.landmark[0].x * width)
    y_base1 = int(hand_landmarks.landmark[0].y * height)
    #Valores para la base del dedo medio en posición 9
    x_base2 = int(hand_landmarks.landmark[9].x * width)
    y_base2 = int(hand_landmarks.landmark[9].y * height)
    #Valores para el dedo indice posición 8
    x_index = int(hand_landmarks.landmark[8].x * width)
    y_index = int(hand_landmarks.landmark[8].y * height)
    
    #Calculo de distancias entre las bases
    d_base = calculate_distance(x_base1, y_base1, x_base2, y_base2)
    #Calculo de distancias entre el dedo indice hasta la muñeca
    d_base_index = calculate_distance(x_base1, y_base1, x_index, y_index)
    
    # Condicion para la distancia
    if d_base_index < d_base:
        finger_down = True
        #color de las lineas
        color_base = (255, 0, 255)
        color_index = (255, 0, 255)
    #Visuaclización de los circulos y lineas
    cv2.circle(output, (x_base1, y_base1), 5, color_base, 2)
    cv2.circle(output, (x_index, y_index), 5, color_index, 2)
    cv2.line(output, (x_base1, y_base1), (x_base2, y_base2), color_base, 3)
    cv2.line(output, (x_base1, y_base1), (x_index, y_index), color_index, 3)
    return finger_down
#Uso de la detección de la mano
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5) as hands:
    
    #area para controlar con el mouse
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        height, width, _ = frame.shape
        frame = cv2.flip(frame, 1)
        # Dibujando un área proporcional a la del juego
        # ancho menos X_Y_INI *2 para que el ancho quede centrado
        area_width = width - X_Y_INI * 2
        # Cálculo de alto basado en la relacion de aspecto
        area_height = int(area_width / aspect_ratio_screen)
        # Imagen auxiliar para visualizar el rectangulo
        aux_image = np.zeros(frame.shape, np.uint8)
        # Visualizar el rectangulo con los puntos encontrados (color azul -1(lleno))
        aux_image = cv2.rectangle(aux_image, (X_Y_INI, X_Y_INI), (X_Y_INI + area_width, X_Y_INI +area_height), (255, 0, 0), -1)
        # Suma de las dos imagenes auxiliares: frame y auximage 0.7(transparencia)
        output = cv2.addWeighted(frame, 1, aux_image, 0.7, 0)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                #Seleccionar punto medio de la mano con coordenas x , y
                x = int(hand_landmarks.landmark[9].x * width)
                y = int(hand_landmarks.landmark[9].y * height)
                #Visualizar con circulos centradas en el punto
                #Pasar las coordenadas del video hacia las coordenadas del juego
                xm = np.interp(x, (X_Y_INI, X_Y_INI + area_width), (SCREEN_GAME_X_INI, SCREEN_GAME_X_FIN))
                ym = np.interp(y, (X_Y_INI, X_Y_INI + area_height), (SCREEN_GAME_Y_INI, SCREEN_GAME_Y_FIN))
                #Mover el mause con las coordenadas X, Y
                pyautogui.moveTo(int(xm), int(ym))
                
                #Llamado a la afuncion dedo abajo
                if detect_finger_down(hand_landmarks):
                    #Si es verdadero da click
                    pyautogui.click()
                #Visualicacion del punto morado en la mano 
                cv2.circle(output, (x, y), 10, color_mouse_pointer, 3)
                cv2.circle(output, (x, y), 5, color_mouse_pointer, -1)
        #cv2.imshow('Frame', frame)
        #Visualición de la imagen
        cv2.imshow('output', output)
        #Función de enlace con el teclado
        if cv2.waitKey(1) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()