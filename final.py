import cv2
import numpy as np
import math
import time
import random
import threading

try:
    import winsound
    def play_sound(freq, duration):
        """Reproduce sonido en un hilo separado para no congelar el video"""
        threading.Thread(target=winsound.Beep, args=(freq, duration), daemon=True).start()
except ImportError:
    def play_sound(freq, duration):
        pass

# Configuración Global

# Datos de calibración
calibration_file = "calibration_data.npz"
dist_coeffs = None
camera_matrix = None
new_camera_matrix = None
roi_calib = None

import os
if os.path.exists(calibration_file):
    try:
        with np.load(calibration_file) as data:
            camera_matrix = data['mtx']
            dist_coeffs = data['dist']
            print("Datos de calibración cargados correctamente.")
    except Exception as e:
        print(f"Error al cargar datos de calibración: {e}")
else:
    print("No se encontró archivo de calibración, se usará la cámara sin corregir.")


# Configuración HSV
LOWER_RED1 = np.array([0, 120, 70])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([170, 120, 70])
UPPER_RED2 = np.array([180, 255, 255])

LOWER_BLUE = np.array([94, 80, 2])
UPPER_BLUE = np.array([126, 255, 255])

LOWER_YELLOW = np.array([20, 100, 100])
UPPER_YELLOW = np.array([35, 255, 255])

# Colores BGR
COLORS_BGR = {
    "Rojo": (45, 55, 255),
    "Azul": (255, 140, 50),
    "Amarillo": (0, 215, 255)
}

# Colores UI
UI_BACKGROUND = (40, 40, 40)
UI_PRIMARY = (255, 140, 50)
UI_SECONDARY = (100, 200, 100)
UI_ACCENT = (0, 215, 255)
UI_SUCCESS = (100, 200, 100)
UI_WARNING = (0, 165, 255)
UI_DANGER = (80, 80, 255)
UI_TEXT_PRIMARY = (255, 255, 255)
UI_TEXT_SECONDARY = (200, 200, 200)
UI_PLAYER1 = (255, 140, 50)
UI_PLAYER2 = (100, 100, 255)

# Secuencias Objetivo
TARGET_PVP = ["Rojo", "Amarillo", "Azul"]
TARGET_CPU = ["Azul", "Amarillo", "Rojo"]

# Estados Globales del Sistema
STATE_MENU = "MENU"
STATE_GAME_PVP = "GAME_PVP"
STATE_GAME_PVE = "GAME_PVE"

# Estados Internos del Juego
GAME_WAITING = "WAITING"
GAME_COUNTDOWN = "COUNTDOWN"
GAME_RESULT = "RESULT"

# Funciones Auxiliares

def draw_rounded_rectangle(img, pt1, pt2, color, thickness=2, radius=20, fill=False):
    """Dibuja un rectángulo con esquinas redondeadas."""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Asegurar que radius no sea mayor que la mitad del ancho/alto
    radius = min(radius, abs(x2-x1)//2, abs(y2-y1)//2)
    
    if fill:
        # Modo relleno: dibujar rectángulos y círculos rellenos
        # Rectángulo central horizontal
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        # Rectángulos laterales
        cv2.rectangle(img, (x1, y1 + radius), (x1 + radius, y2 - radius), color, -1)
        cv2.rectangle(img, (x2 - radius, y1 + radius), (x2, y2 - radius), color, -1)
        # Círculos en las esquinas (rellenos)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        # Modo contorno: dibujar líneas y arcos
        # Líneas rectas
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        
        # Esquinas redondeadas (arcos)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def draw_text_with_background(img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, 
                               font_scale=1, text_color=(255,255,255), 
                               bg_color=(0,0,0), thickness=2, padding=10, alpha=0.7):
    """Dibuja texto con fondo semitransparente."""
    # Obtener tamaño del texto
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = position
    
    # Crear overlay para transparencia
    overlay = img.copy()
    
    # Dibujar rectángulo de fondo
    cv2.rectangle(overlay, 
                  (x - padding, y - text_height - padding), 
                  (x + text_width + padding, y + baseline + padding), 
                  bg_color, -1)
    
    # Aplicar transparencia
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Dibujar texto
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)
    
    return text_width, text_height

def draw_text_with_outline(img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX,
                           font_scale=1, text_color=(255,255,255), 
                           outline_color=(0,0,0), thickness=2, outline_thickness=None):
    """Dibuja texto con contorno para mejor legibilidad."""
    if outline_thickness is None:
        outline_thickness = thickness + 2
    
    x, y = position
    
    # Dibujar contorno
    cv2.putText(img, text, (x, y), font, font_scale, outline_color, outline_thickness)
    
    # Dibujar texto
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)

def draw_progress_circle(img, center, radius, progress, color, thickness=8):
    """Dibuja un círculo de progreso (0.0 a 1.0)."""
    # Círculo de fondo (gris)
    cv2.circle(img, center, radius, (80, 80, 80), thickness)
    
    # Arco de progreso
    if progress > 0:
        angle = int(360 * progress)
        cv2.ellipse(img, center, (radius, radius), -90, 0, angle, color, thickness)

# Funciones de Visión

def calculate_angle(a, b, c):
    """Calcula el ángulo entre 3 puntos (start, end, far) para detectar dedos."""
    length_a = math.sqrt((b[0] - c[0])**2 + (b[1] - c[1])**2)
    length_b = math.sqrt((a[0] - c[0])**2 + (a[1] - c[1])**2)
    length_c = math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
    if length_a * length_b == 0: return 0
    cos_angle = (length_a**2 + length_b**2 - length_c**2) / (2 * length_a * length_b)
    cos_angle = max(-1, min(1, cos_angle))
    angle = math.acos(cos_angle)
    return math.degrees(angle)

def detect_gesture(roi):
    """Detecta Piedra, Papel o Tijera en una Región de Interés (ROI)."""
    if roi.size == 0: return "..."
    
    # Convertir a HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Valores por defecto (Fallback)
    l_green = np.array([35, 50, 50])
    u_green = np.array([85, 255, 255])
    l_skin = np.array([0, 30, 60])
    u_skin = np.array([20, 255, 255])
    
    # Cargar calibración
    if os.path.exists("color_config.npy"):
        try:
            conf = np.load("color_config.npy", allow_pickle=True).item()
            l_green = conf['bg_lower']
            u_green = conf['bg_upper']
            l_skin = conf['skin_lower']
            u_skin = conf['skin_upper']
        except Exception: pass

    # 1. Máscara Fondo (Chroma)
    bg_mask = cv2.inRange(hsv, l_green, u_green)
    
    # 2. Máscara Piel
    skin_mask = cv2.inRange(hsv, l_skin, u_skin)
    
    # 3. Combinación
    fg_mask = cv2.bitwise_and(skin_mask, cv2.bitwise_not(bg_mask))
    
    # 4. Procesamiento morfológico
    kernel = np.ones((5,5), np.uint8)
    fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
    fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)
    
    _, thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    gesture = "..."
    
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        
        # Filtro de área mínima para evitar ruido
        if cv2.contourArea(contour) > 2000:
            hull = cv2.convexHull(contour, returnPoints=False)
            
            try:
                defects = cv2.convexityDefects(contour, hull)
            except:
                return "..."

            count_defects = 0
            h, w = roi.shape[:2]
            
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(contour[s][0])
                    end = tuple(contour[e][0])
                    far = tuple(contour[f][0])
                    
                    angle = calculate_angle(start, end, far)
                    depth = d / 256.0 
                    
                    # Filtros: Ignorar muñeca (parte baja) y ángulos abiertos
                    if far[1] > (h * 0.9): continue 
                    if depth > (h * 0.15) and angle <= 90:
                        count_defects += 1
                        # Feedback visual (punto rojo en defecto)
                        cv2.circle(roi, far, 6, (0, 0, 255), -1)
            
            # Clasificación
            if count_defects == 0: gesture = "Piedra"
            elif count_defects == 1 or count_defects == 2: gesture = "Tijera"
            elif count_defects >= 3: gesture = "Papel"
            
            # Dibujar contorno para feedback visual
            cv2.drawContours(roi, [contour], -1, (0, 255, 0), 2)
            
    return gesture

def detect_color_ball(frame_hsv):
    """Detecta el color de la bola para el selector de modo."""
    # Máscaras
    mask_red = cv2.inRange(frame_hsv, LOWER_RED1, UPPER_RED1) + cv2.inRange(frame_hsv, LOWER_RED2, UPPER_RED2)
    mask_blue = cv2.inRange(frame_hsv, LOWER_BLUE, UPPER_BLUE)
    mask_yellow = cv2.inRange(frame_hsv, LOWER_YELLOW, UPPER_YELLOW)
    
    # Limpieza morfológica
    kernel = np.ones((5, 5), np.uint8)
    
    # Opening (quitamos ruido blanco)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)

    # Closing (cerramos agujeros negros dentro de la bola)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)

    # Contornos
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    max_area = 0
    detected = None
    contour_draw = None
    MIN_AREA = 2000 # Filtro de tamaño para la bola
    
    def check_contours(contours, label):
        nonlocal max_area, detected, contour_draw
        for c in contours:
            area = cv2.contourArea(c)
            if area > max_area and area > MIN_AREA:
                # Comprobación de Circularidad
                perimeter = cv2.arcLength(c, True)
                if perimeter == 0: continue
                
                # Formula circularidad: 4*pi*area / perimetro^2
                # Un circulo perfecto es 1.0
                circularity = 4 * math.pi * (area / (perimeter * perimeter))
                
                if circularity > 0.60: # 60% de redondez mínimo
                    max_area = area
                    detected = label
                    contour_draw = c

    check_contours(contours_red, "Rojo")
    check_contours(contours_blue, "Azul")
    check_contours(contours_yellow, "Amarillo")
            
    return detected, contour_draw

def determine_winner(p1, p2):
    """Lógica del juego Piedra, Papel, Tijera."""
    if p1 == "..." or p2 == "...": return "Gesto Invalido", (128, 128, 128)
    if p1 == p2: return "EMPATE", (255, 255, 0)
    
    wins = {"Piedra": "Tijera", "Papel": "Piedra", "Tijera": "Papel"}
    
    if wins.get(p1) == p2:
        return "GANA JUGADOR 1", (0, 255, 0) # Verde
    else:
        return "GANA JUGADOR 2", (0, 255, 255) # Amarillo/Cian para J2

# Vistas

def run_menu_screen(frame, state_vars):
    """Lógica y renderizado del MENU PRINCIPAL (Selector de Bolas)."""
    height, width, _ = frame.shape
    
    # Pre-procesamiento: Blur para reducir ruido
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # Detección
    color, contour = detect_color_ball(hsv)
    
    # Lógica de estabilidad
    if color:
        # Dibujar contorno con efecto de brillo
        cv2.drawContours(frame, [contour], -1, COLORS_BGR[color], 5)
        cv2.drawContours(frame, [contour], -1, UI_TEXT_PRIMARY, 2)
        
        # Mostrar detección actual con fondo
        draw_text_with_background(frame, f"Detectando: {color}", (20, 60), 
                                 font_scale=0.9, text_color=UI_TEXT_PRIMARY, 
                                 bg_color=COLORS_BGR[color], thickness=2, padding=15, alpha=0.8)
        
        if color == state_vars['last_detected_color']:
            state_vars['detection_frames'] += 1
        else:
            state_vars['detection_frames'] = 0
            state_vars['last_detected_color'] = color
        
        if state_vars['detection_frames'] > 15: # REQUIRED_FRAMES
            if not state_vars['sequence'] or state_vars['sequence'][-1] != color:
                state_vars['sequence'].append(color)
                state_vars['detection_frames'] = 0
                if len(state_vars['sequence']) > 3:
                    state_vars['sequence'].pop(0)
    else:
        state_vars['detection_frames'] = 0
        state_vars['last_detected_color'] = None

    # ==================== UI DEL MENÚ ====================
    
    # Título principal con efecto de sombra
    title = "PIEDRA, PAPEL O TIJERA"
    title_font_scale = 1.5
    title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, title_font_scale, 4)[0]
    title_x = int((width - title_size[0]) / 2)
    title_y = 80
    
    # Sombra del título
    draw_text_with_outline(frame, title, (title_x, title_y), 
                          font=cv2.FONT_HERSHEY_DUPLEX, font_scale=title_font_scale,
                          text_color=UI_ACCENT, outline_color=(0, 0, 0), 
                          thickness=4, outline_thickness=8)
    
    # Subtítulo
    subtitle = "Selecciona el modo de juego con las bolas de colores"
    subtitle_size = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    subtitle_x = int((width - subtitle_size[0]) / 2)
    draw_text_with_outline(frame, subtitle, (subtitle_x, title_y + 50),
                          font_scale=0.7, text_color=UI_TEXT_SECONDARY,
                          outline_color=(0, 0, 0), thickness=2)
    
    # Panel de instrucciones (esquina superior derecha)
    panel_x = width - 350
    panel_y = 40
    draw_text_with_background(frame, "COMBOS DE COLORES:", (panel_x, panel_y + 25),
                            font_scale=0.7, text_color=UI_TEXT_PRIMARY,
                            bg_color=UI_BACKGROUND, thickness=2, padding=12, alpha=0.85)
    
    # Mostrar combos con círculos de colores
    combo_y = panel_y + 65
    # Combo PvP
    for i, c in enumerate(["Rojo", "Amarillo", "Azul"]):
        cv2.circle(frame, (panel_x + 20 + i*35, combo_y), 12, COLORS_BGR[c], -1)
        cv2.circle(frame, (panel_x + 20 + i*35, combo_y), 14, UI_TEXT_PRIMARY, 2)
    draw_text_with_outline(frame, "-> Jugador vs Jugador", (panel_x + 125, combo_y + 5),
                          font_scale=0.6, text_color=UI_SUCCESS, thickness=1)
    
    # Combo PvE
    combo_y += 40
    for i, c in enumerate(["Azul", "Amarillo", "Rojo"]):
        cv2.circle(frame, (panel_x + 20 + i*35, combo_y), 12, COLORS_BGR[c], -1)
        cv2.circle(frame, (panel_x + 20 + i*35, combo_y), 14, UI_TEXT_PRIMARY, 2)
    draw_text_with_outline(frame, "-> Jugador vs CPU", (panel_x + 125, combo_y + 5),
                          font_scale=0.6, text_color=UI_WARNING, thickness=1)
    
    # Indicador de secuencia actual (parte inferior central)
    seq_label = "SECUENCIA ACTUAL:"
    seq_label_size = cv2.getTextSize(seq_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    seq_label_x = int((width - seq_label_size[0]) / 2)
    
    draw_text_with_outline(frame, seq_label, (seq_label_x, height - 150),
                          font_scale=0.8, text_color=UI_TEXT_PRIMARY,
                          outline_color=(0, 0, 0), thickness=2)
    
    # Dibujar slots de secuencia (3 círculos)
    slot_y = height - 90
    slot_start_x = int(width / 2) - 100
    slot_spacing = 70
    
    for i in range(3):
        slot_x = slot_start_x + (i * slot_spacing)
        
        if i < len(state_vars['sequence']):
            # Círculo lleno con el color detectado
            color_name = state_vars['sequence'][i]
            # Efecto de brillo (círculo exterior)
            cv2.circle(frame, (slot_x, slot_y), 32, COLORS_BGR[color_name], 3)
            cv2.circle(frame, (slot_x, slot_y), 26, COLORS_BGR[color_name], -1)
            cv2.circle(frame, (slot_x, slot_y), 28, UI_TEXT_PRIMARY, 2)
        else:
            # Slot vacío
            cv2.circle(frame, (slot_x, slot_y), 26, (60, 60, 60), -1)
            cv2.circle(frame, (slot_x, slot_y), 28, UI_TEXT_SECONDARY, 2)
            # Número de slot
            num_text = str(i + 1)
            num_size = cv2.getTextSize(num_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.putText(frame, num_text, (slot_x - num_size[0]//2, slot_y + num_size[1]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, UI_TEXT_SECONDARY, 2)

    # Comprobar activación de modos y devolver el posible siguiente estado
    next_global_state = STATE_MENU
    mode_text = ""
    mode_color = UI_SUCCESS
    
    if state_vars['sequence'] == TARGET_PVP:
        mode_text = "JUGADOR VS JUGADOR"
        next_global_state = STATE_GAME_PVP
        mode_color = UI_SUCCESS
    elif state_vars['sequence'] == TARGET_CPU:
        mode_text = "JUGADOR VS CPU"
        next_global_state = STATE_GAME_PVE
        mode_color = UI_WARNING
    
    if mode_text:
        # Panel de confirmación de modo (centro de la pantalla)
        panel_width = 600
        panel_height = 180
        panel_left = int((width - panel_width) / 2)
        panel_top = int((height - panel_height) / 2)
        
        # Crear overlay semitransparente
        overlay = frame.copy()
        draw_rounded_rectangle(overlay, (panel_left, panel_top), 
                             (panel_left + panel_width, panel_top + panel_height),
                             UI_BACKGROUND, thickness=-1, radius=30, fill=True)
        cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)
        
        # Borde del panel con el color del modo
        draw_rounded_rectangle(frame, (panel_left, panel_top), 
                             (panel_left + panel_width, panel_top + panel_height),
                             mode_color, thickness=4, radius=30)
        
        # Título del modo
        mode_title = "MODO SELECCIONADO"
        mode_title_size = cv2.getTextSize(mode_title, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        mode_title_x = int((width - mode_title_size[0]) / 2)
        draw_text_with_outline(frame, mode_title, (mode_title_x, panel_top + 45),
                              font_scale=0.7, text_color=UI_TEXT_SECONDARY,
                              outline_color=(0, 0, 0), thickness=2)
        
        # Nombre del modo (grande y destacado)
        mode_size = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 3)[0]
        mode_x = int((width - mode_size[0]) / 2)
        draw_text_with_outline(frame, mode_text, (mode_x, panel_top + 95),
                              font=cv2.FONT_HERSHEY_DUPLEX, font_scale=1.2,
                              text_color=mode_color, outline_color=(0, 0, 0),
                              thickness=3, outline_thickness=6)
        
        # Instrucción para confirmar
        confirm_text = "Pulsa 'ESPACIO' para CONFIRMAR"
        confirm_size = cv2.getTextSize(confirm_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        confirm_x = int((width - confirm_size[0]) / 2)
        draw_text_with_outline(frame, confirm_text, (confirm_x, panel_top + 145),
                              font_scale=0.8, text_color=UI_ACCENT,
                              outline_color=(0, 0, 0), thickness=2)

    return next_global_state


def run_game_screen(frame, mode, game_vars, cap_ref):
    """Lógica compartida para PvP y PvE."""
    height, width, _ = frame.shape
    
    # Definir ROIs
    box_width = int(width * 0.45)
    box_height = int(height * 0.6)
    margin_top = int(height * 0.15)
    
    # P1 (Izquierda - Humano)
    r1 = (20, margin_top, 20 + box_width, margin_top + box_height) # x1, y1, x2, y2
    
    # P2 (Derecha - Humano o CPU)
    r2 = (width - 20 - box_width, margin_top, width - 20, margin_top + box_height)

    # ==================== DIBUJAR CAJAS DE JUGADORES ====================
    
    # Caja Jugador 1 con esquinas redondeadas
    draw_rounded_rectangle(frame, (r1[0], r1[1]), (r1[2], r1[3]), UI_PLAYER1, thickness=5, radius=20)
    
    # Etiqueta Jugador 1 con fondo
    label_p1 = "JUGADOR 1"
    label_p1_size = cv2.getTextSize(label_p1, cv2.FONT_HERSHEY_DUPLEX, 1, 3)[0]
    label_p1_x = r1[0] + (box_width - label_p1_size[0]) // 2
    draw_text_with_background(frame, label_p1, (label_p1_x, r1[1] - 25),
                            font=cv2.FONT_HERSHEY_DUPLEX, font_scale=1,
                            text_color=UI_TEXT_PRIMARY, bg_color=UI_PLAYER1,
                            thickness=3, padding=15, alpha=0.9)
    
    # Caja Jugador 2 con esquinas redondeadas
    draw_rounded_rectangle(frame, (r2[0], r2[1]), (r2[2], r2[3]), UI_PLAYER2, thickness=5, radius=20)
    
    # Etiqueta Jugador 2 con fondo
    name_p2 = "JUGADOR 2" if mode == STATE_GAME_PVP else "CPU"
    label_p2_size = cv2.getTextSize(name_p2, cv2.FONT_HERSHEY_DUPLEX, 1, 3)[0]
    label_p2_x = r2[0] + (box_width - label_p2_size[0]) // 2
    draw_text_with_background(frame, name_p2, (label_p2_x, r2[1] - 25),
                            font=cv2.FONT_HERSHEY_DUPLEX, font_scale=1,
                            text_color=UI_TEXT_PRIMARY, bg_color=UI_PLAYER2,
                            thickness=3, padding=15, alpha=0.9)

    # Detección en Tiempo Real (Solo para feedback visual)
    roi_p1 = frame[r1[1]:r1[3], r1[0]:r1[2]]
    current_p1 = detect_gesture(roi_p1)
    
    current_p2 = "..."
    if mode == STATE_GAME_PVP:
        roi_p2 = frame[r2[1]:r2[3], r2[0]:r2[2]]
        current_p2 = detect_gesture(roi_p2)
    else:
        current_p2 = "Pensando..." if game_vars['state'] != GAME_WAITING else "..."

    # ==================== MÁQUINA DE ESTADOS DEL JUEGO ====================
    
    if game_vars['state'] == GAME_WAITING:
        # Instrucción central
        text = "Prepara tu gesto"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        text_x = int((width - text_size[0]) / 2)
        draw_text_with_outline(frame, text, (text_x, height - 120),
                              font_scale=1.2, text_color=UI_TEXT_PRIMARY,
                              outline_color=(0, 0, 0), thickness=3)
        
        # Botón de inicio
        start_text = "Presiona ESPACIO para comenzar"
        start_size = cv2.getTextSize(start_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        start_x = int((width - start_size[0]) / 2)
        draw_text_with_background(frame, start_text, (start_x, height - 60),
                                font_scale=0.9, text_color=UI_TEXT_PRIMARY,
                                bg_color=UI_SUCCESS, thickness=2, padding=15, alpha=0.85)
        
        # Mostrar gesto actual detectado con icono
        gesture_p1_y = r1[3] + 70
        draw_text_with_outline(frame, current_p1, (r1[0] + 20, gesture_p1_y),
                              font=cv2.FONT_HERSHEY_DUPLEX, font_scale=1.8,
                              text_color=UI_PLAYER1, outline_color=(0, 0, 0),
                              thickness=4, outline_thickness=7)
        
        if mode == STATE_GAME_PVP:
            gesture_p2_y = r2[3] + 70
            draw_text_with_outline(frame, current_p2, (r2[0] + 20, gesture_p2_y),
                                  font=cv2.FONT_HERSHEY_DUPLEX, font_scale=1.8,
                                  text_color=UI_PLAYER2, outline_color=(0, 0, 0),
                                  thickness=4, outline_thickness=7)

    elif game_vars['state'] == GAME_COUNTDOWN:
        elapsed = time.time() - game_vars['start_time']
        timer = 3 - int(elapsed)
        
        # Sonido
        if timer < game_vars['last_beep'] and timer > 0:
            play_sound(1000, 200)
            game_vars['last_beep'] = timer
        
        if timer > 0:
            # Cuenta regresiva con círculo de progreso
            countdown_center = (int(width / 2), int(height / 2))
            countdown_radius = 120
            
            # Progreso (0 a 3 segundos -> 1.0 a 0.0)
            progress = 1.0 - (elapsed / 3.0)
            
            # Círculo de progreso
            draw_progress_circle(frame, countdown_center, countdown_radius, progress, UI_ACCENT, thickness=15)
            
            # Número de cuenta regresiva con efecto dramático
            timer_str = str(timer)
            # Tamaño de fuente con escala dinámica (pulso)
            scale_factor = 1.0 + (0.3 * (1.0 - (elapsed % 1.0)))  # Pulso cada segundo
            font_scale = 8 * scale_factor
            timer_size = cv2.getTextSize(timer_str, cv2.FONT_HERSHEY_DUPLEX, font_scale, int(15 * scale_factor))[0]
            timer_x = countdown_center[0] - timer_size[0] // 2
            timer_y = countdown_center[1] + timer_size[1] // 2
            
            draw_text_with_outline(frame, timer_str, (timer_x, timer_y),
                                  font=cv2.FONT_HERSHEY_DUPLEX, font_scale=font_scale,
                                  text_color=UI_ACCENT, outline_color=(0, 0, 0),
                                  thickness=int(10 * scale_factor), outline_thickness=int(15 * scale_factor))
            
            # Feedback visual continuo de gestos (pequeño)
            draw_text_with_outline(frame, current_p1, (r1[0] + 20, r1[3] + 70),
                                  font=cv2.FONT_HERSHEY_DUPLEX, font_scale=1.5,
                                  text_color=UI_PLAYER1, outline_color=(0, 0, 0),
                                  thickness=3, outline_thickness=6)
            if mode == STATE_GAME_PVP:
                draw_text_with_outline(frame, current_p2, (r2[0] + 20, r2[3] + 70),
                                      font=cv2.FONT_HERSHEY_DUPLEX, font_scale=1.5,
                                      text_color=UI_PLAYER2, outline_color=(0, 0, 0),
                                      thickness=3, outline_thickness=6)
        else:
            # FINISH
            play_sound(2000, 400)
            finish_text = "¡YA!"
            finish_size = cv2.getTextSize(finish_text, cv2.FONT_HERSHEY_DUPLEX, 6, 15)[0]
            finish_x = int((width - finish_size[0]) / 2)
            finish_y = int(height / 2)
            draw_text_with_outline(frame, finish_text, (finish_x, finish_y),
                                  font=cv2.FONT_HERSHEY_DUPLEX, font_scale=6,
                                  text_color=UI_SUCCESS, outline_color=(0, 0, 0),
                                  thickness=15, outline_thickness=20)
            cv2.imshow(window_name, frame) # Forzar render
            cv2.waitKey(1)
            time.sleep(0.4) # Delay táctico
            
            # Captura Final
            if mode == STATE_GAME_PVP:
                ret, final_frame = cap_ref.read()
                if ret:
                    # Aplicar correccíon de distorsión al frame final también
                    if camera_matrix is not None and dist_coeffs is not None and new_camera_matrix is not None:
                         final_frame = cv2.undistort(final_frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
                         x, y, w, h = roi_calib
                         final_frame = final_frame[y:y+h, x:x+w]
                    
                    frame_f = cv2.flip(final_frame, 1)
                    # Recortes sobre frame final
                    roi1_f = frame_f[r1[1]:r1[3], r1[0]:r1[2]]
                    roi2_f = frame_f[r2[1]:r2[3], r2[0]:r2[2]]
                    game_vars['p1_final'] = detect_gesture(roi1_f)
                    game_vars['p2_final'] = detect_gesture(roi2_f)
                else:
                    game_vars['p1_final'] = current_p1
                    game_vars['p2_final'] = current_p2
            else:
                # Modo CPU: Capturamos P1 y generamos P2
                ret, final_frame = cap_ref.read()
                if ret:
                    # Aplicar correccíon de distorsión al frame final también
                    if camera_matrix is not None and dist_coeffs is not None and new_camera_matrix is not None:
                         final_frame = cv2.undistort(final_frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
                         x, y, w, h = roi_calib
                         final_frame = final_frame[y:y+h, x:x+w]

                    frame_f = cv2.flip(final_frame, 1)
                    roi1_f = frame_f[r1[1]:r1[3], r1[0]:r1[2]]
                    game_vars['p1_final'] = detect_gesture(roi1_f)
                else:
                    game_vars['p1_final'] = current_p1
                
                game_vars['p2_final'] = random.choice(["Piedra", "Papel", "Tijera"])
            
            # Calcular ganador
            res_text, res_color = determine_winner(game_vars['p1_final'], game_vars['p2_final'])
            game_vars['result_text'] = res_text
            game_vars['result_color'] = res_color
            
            # Sonido Final
            if "1" in res_text: play_sound(500, 600)
            elif "2" in res_text or "CPU" in res_text: play_sound(1500, 600)
            else: play_sound(300, 300)

            game_vars['state'] = GAME_RESULT

    elif game_vars['state'] == GAME_RESULT:
        # ==================== PANTALLA DE RESULTADOS ====================
        
        # Mostrar gestos finales con estilo
        gesture_p1_y = r1[3] + 70
        draw_text_with_outline(frame, game_vars['p1_final'], (r1[0] + 20, gesture_p1_y),
                              font=cv2.FONT_HERSHEY_DUPLEX, font_scale=2,
                              text_color=UI_PLAYER1, outline_color=(0, 0, 0),
                              thickness=5, outline_thickness=8)
        
        gesture_p2_y = r2[3] + 70
        draw_text_with_outline(frame, game_vars['p2_final'], (r2[0] + 20, gesture_p2_y),
                              font=cv2.FONT_HERSHEY_DUPLEX, font_scale=2,
                              text_color=UI_PLAYER2, outline_color=(0, 0, 0),
                              thickness=5, outline_thickness=8)
        
        # Banner de resultado (centro superior)
        banner_width = 700
        banner_height = 160
        banner_left = int((width - banner_width) / 2)
        banner_top = 60
        
        # Determinar color del banner según resultado
        if "1" in game_vars['result_text']:
            banner_color = UI_PLAYER1
        elif "2" in game_vars['result_text'] or "CPU" in game_vars['result_text']:
            banner_color = UI_PLAYER2
        else:  # Empate
            banner_color = UI_ACCENT
        
        # Overlay semitransparente para el banner
        overlay = frame.copy()
        draw_rounded_rectangle(overlay, (banner_left, banner_top),
                             (banner_left + banner_width, banner_top + banner_height),
                             UI_BACKGROUND, thickness=-1, radius=30, fill=True)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        
        # Borde del banner con color del ganador
        draw_rounded_rectangle(frame, (banner_left, banner_top),
                             (banner_left + banner_width, banner_top + banner_height),
                             banner_color, thickness=6, radius=30)
        
        # Texto "RESULTADO"
        result_label = "RESULTADO"
        result_label_size = cv2.getTextSize(result_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        result_label_x = int((width - result_label_size[0]) / 2)
        draw_text_with_outline(frame, result_label, (result_label_x, banner_top + 45),
                              font_scale=0.8, text_color=UI_TEXT_SECONDARY,
                              outline_color=(0, 0, 0), thickness=2)
        
        # Texto del ganador (grande y destacado)
        winner_size = cv2.getTextSize(game_vars['result_text'], cv2.FONT_HERSHEY_DUPLEX, 2, 5)[0]
        winner_x = int((width - winner_size[0]) / 2)
        draw_text_with_outline(frame, game_vars['result_text'], (winner_x, banner_top + 110),
                              font=cv2.FONT_HERSHEY_DUPLEX, font_scale=2,
                              text_color=banner_color, outline_color=(0, 0, 0),
                              thickness=5, outline_thickness=9)
        
        # Efectos de confeti simple (puntos aleatorios) para el ganador
        if "GANA" in game_vars['result_text']:
            for _ in range(30):
                x = random.randint(0, width)
                y = random.randint(0, height // 3)
                color_choice = random.choice([UI_ACCENT, UI_SUCCESS, UI_WARNING, UI_PRIMARY])
                cv2.circle(frame, (x, y), random.randint(3, 8), color_choice, -1)
        
        # Instrucciones en la parte inferior
        instructions = "Presiona 'R' para REVANCHA  |  'M' para MENU"
        instr_size = cv2.getTextSize(instructions, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        instr_x = int((width - instr_size[0]) / 2)
        draw_text_with_background(frame, instructions, (instr_x, height - 50),
                                font_scale=0.8, text_color=UI_TEXT_PRIMARY,
                                bg_color=UI_BACKGROUND, thickness=2, padding=12, alpha=0.85)



# Loop Principal

# Configuración de ventana
window_name = 'Sistema de Vision Artificial - Proyecto Final'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Iniciar cámara
cap = cv2.VideoCapture(0)



# Variables de Estado Global
current_global_state = STATE_MENU

# Variables persistentes para el menú
menu_vars = {
    'sequence': [],
    'last_detected_color': None,
    'detection_frames': 0
}

# Variables persistentes para el juego (se reinician al entrar)
game_vars = {
    'state': GAME_WAITING,
    'start_time': 0,
    'last_beep': 4,
    'p1_final': "...",
    'p2_final': "...",
    'result_text': "",
    'result_color': (255, 255, 255)
}

prev_time = 0
try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Aplicar correccíon de distorsión si hay datos
        if camera_matrix is not None and dist_coeffs is not None:
            h, w = frame.shape[:2]
            if new_camera_matrix is None:
                new_camera_matrix, roi_calib = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
            
            # Undistort
            frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
            
            # Recortar la imagen (opcional, pero recomendado para quitar bordes negros)
            x, y, w, h = roi_calib
            frame = frame[y:y+h, x:x+w]

        frame = cv2.flip(frame, 1)

        # FPS Counter
        curr_time = time.time()
        fps = 0
        if prev_time != 0 and (curr_time - prev_time) > 0:
            fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1] - 130, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # CONTROL DE FLUJO POR ESTADOS
        key = cv2.waitKey(1) & 0xFF

        if current_global_state == STATE_MENU:
            possible_next_state = run_menu_screen(frame, menu_vars)
            
            # Si el menú propone un cambio de estado, esperamos confirmación
            if possible_next_state != STATE_MENU:
                if key == 32: # ESPACIO para confirmar
                    current_global_state = possible_next_state
                    # Resetear variables de juego
                    game_vars['state'] = GAME_WAITING
                    menu_vars['sequence'] = [] # Limpiar secuencia para la próxima vez
            
            if key == 32 and possible_next_state == STATE_MENU: 
                 # Si no hay secuencia completa y pulsan espacio, limpiar
                 menu_vars['sequence'] = []

        elif current_global_state in [STATE_GAME_PVP, STATE_GAME_PVE]:
            run_game_screen(frame, current_global_state, game_vars, cap)

            if key == 32 and game_vars['state'] == GAME_WAITING: # ESPACIO empieza juego
                game_vars['state'] = GAME_COUNTDOWN
                game_vars['start_time'] = time.time()
                game_vars['last_beep'] = 4
            
            elif key == ord('r') and game_vars['state'] == GAME_RESULT: # R reinicia ronda
                game_vars['state'] = GAME_WAITING
                game_vars['p1_final'] = "..."
                game_vars['p2_final'] = "..."
                
            elif key == ord('m'): # M vuelve al menú
                current_global_state = STATE_MENU
                game_vars['state'] = GAME_WAITING

        # Mostrar frame final
        cv2.imshow(window_name, frame)
        
        if key == ord('q'): # Salir
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
