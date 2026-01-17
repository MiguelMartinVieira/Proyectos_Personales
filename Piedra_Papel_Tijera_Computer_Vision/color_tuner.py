import cv2
import numpy as np
import os

# Archivo de configuración
CONFIG_FILE = "color_config.npy"

def nothing(x):
    pass

def save_config(skin_min, skin_max, green_min, green_max):
    data = {
        'skin_lower': skin_min,
        'skin_upper': skin_max,
        'bg_lower': green_min,
        'bg_upper': green_max
    }
    np.save(CONFIG_FILE, data)
    print(f"Configuración guardada en {CONFIG_FILE}")

def main():
    cap = cv2.VideoCapture(0)
    
    cv2.namedWindow('Calibrador')
    
    # Valores iniciales (Defaults)
    # Piel
    cv2.createTrackbar('Skin H Min', 'Calibrador', 0, 179, nothing)
    cv2.createTrackbar('Skin H Max', 'Calibrador', 20, 179, nothing)
    cv2.createTrackbar('Skin S Min', 'Calibrador', 30, 255, nothing)
    cv2.createTrackbar('Skin S Max', 'Calibrador', 255, 255, nothing)
    cv2.createTrackbar('Skin V Min', 'Calibrador', 60, 255, nothing)
    cv2.createTrackbar('Skin V Max', 'Calibrador', 255, 255, nothing)
    
    # Fondo Verde
    cv2.createTrackbar('Bg H Min', 'Calibrador', 35, 179, nothing)
    cv2.createTrackbar('Bg H Max', 'Calibrador', 85, 179, nothing)
    cv2.createTrackbar('Bg S Min', 'Calibrador', 50, 255, nothing)
    cv2.createTrackbar('Bg S Max', 'Calibrador', 255, 255, nothing)
    cv2.createTrackbar('Bg V Min', 'Calibrador', 50, 255, nothing)
    cv2.createTrackbar('Bg V Max', 'Calibrador', 255, 255, nothing)
    
    print("----------------------------------------------------------------")
    print("INSTRUCCIONES DE CALIBRACIÓN:")
    print("1. Ajusta los sliders de 'Skin' para que TUS MANOS se vean BLANCAS en 'Skin Mask'.")
    print("2. Ajusta los sliders de 'Bg' para que el FONDO VERDE se vea BLANCO en 'Green Mask'.")
    print("3. La ventana 'RESULTADO FINAL' debe mostrar SOLO tus manos (blancas) y fondo negro.")
    print("   (Si hay ruido negro en las manos, ajusta Piel. Si hay ruido blanco en fondo, ajusta Bg)")
    print("4. Pulsa 'S' para GUARDAR y SALIR.")
    print("5. Pulsa 'Q' para salir SIN guardar.")
    print("----------------------------------------------------------------")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Leer trackbars
        s_h_min = cv2.getTrackbarPos('Skin H Min', 'Calibrador')
        s_h_max = cv2.getTrackbarPos('Skin H Max', 'Calibrador')
        s_s_min = cv2.getTrackbarPos('Skin S Min', 'Calibrador')
        s_s_max = cv2.getTrackbarPos('Skin S Max', 'Calibrador')
        s_v_min = cv2.getTrackbarPos('Skin V Min', 'Calibrador')
        s_v_max = cv2.getTrackbarPos('Skin V Max', 'Calibrador')
        
        b_h_min = cv2.getTrackbarPos('Bg H Min', 'Calibrador')
        b_h_max = cv2.getTrackbarPos('Bg H Max', 'Calibrador')
        b_s_min = cv2.getTrackbarPos('Bg S Min', 'Calibrador')
        b_s_max = cv2.getTrackbarPos('Bg S Max', 'Calibrador')
        b_v_min = cv2.getTrackbarPos('Bg V Min', 'Calibrador')
        b_v_max = cv2.getTrackbarPos('Bg V Max', 'Calibrador')
        
        # Crear arrays
        lower_skin = np.array([s_h_min, s_s_min, s_v_min])
        upper_skin = np.array([s_h_max, s_s_max, s_v_max])
        
        lower_bg = np.array([b_h_min, b_s_min, b_v_min])
        upper_bg = np.array([b_h_max, b_s_max, b_v_max])
        
        # Máscaras
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        bg_mask = cv2.inRange(hsv, lower_bg, upper_bg)
        
        # Combinación: Piel AND NOT Fondo
        not_bg = cv2.bitwise_not(bg_mask)
        final_mask = cv2.bitwise_and(skin_mask, not_bg)
        
        # Visualización
        scale = 0.5
        h, w = frame.shape[:2]
        small_frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
        small_skin = cv2.cvtColor(cv2.resize(skin_mask, (0,0), fx=scale, fy=scale), cv2.COLOR_GRAY2BGR)
        small_bg = cv2.cvtColor(cv2.resize(bg_mask, (0,0), fx=scale, fy=scale), cv2.COLOR_GRAY2BGR)
        small_final = cv2.cvtColor(cv2.resize(final_mask, (0,0), fx=scale, fy=scale), cv2.COLOR_GRAY2BGR)
        
        cv2.putText(small_skin, "1. SKIN MASK (Busca blanco en mano)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(small_bg, "2. BG MASK (Busca blanco en fondo)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(small_final, "3. RESULTADO (Mano blanca, fondo negro)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        top_row = np.hstack((small_frame, small_final))
        bottom_row = np.hstack((small_skin, small_bg))
        collage = np.vstack((top_row, bottom_row))
        
        cv2.imshow('Calibrador', collage)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_config(lower_skin, upper_skin, lower_bg, upper_bg)
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
