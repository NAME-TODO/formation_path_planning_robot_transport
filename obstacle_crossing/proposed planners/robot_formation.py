import cv2
import numpy as np
import math
from typing import Tuple, Optional

def draw_robot_formation(image: np.ndarray, 
                        point_A: Tuple[int, int], 
                        point_B: Tuple[int, int], 
                        d_rob: float, 
                        r_rob: int,
                        circumference_color: Tuple[int, int, int] = (0, 255, 0),
                        robot_color: Tuple[int, int, int] = (255, 0, 0),
                        thickness: int = 2) -> np.ndarray:
    """
    Disegna una formazione di 6 robot su un'immagine.
    
    Args:
        image: Immagine su cui disegnare
        point_A: Primo punto del diametro (x, y)
        point_B: Secondo punto del diametro (x, y)
        d_rob: Distanza euclidea tra robot principali e aggiuntivi
        r_rob: Raggio dei cerchi robot
        circumference_color: Colore della circonferenza (B, G, R)
        robot_color: Colore dei cerchi robot (B, G, R)
        thickness: Spessore delle linee
    
    Returns:
        Immagine modificata con la formazione di robot
    """
    # Copia l'immagine per non modificare l'originale
    result_img = image.copy()
    
    # Calcola il centro e il raggio della circonferenza
    center_x = (point_A[0] + point_B[0]) // 2
    center_y = (point_A[1] + point_B[1]) // 2
    center = (center_x, center_y)
    
    # Calcola il raggio (metà della distanza AB)
    radius = int(math.sqrt((point_B[0] - point_A[0])**2 + (point_B[1] - point_A[1])**2) / 2)
    
    # Disegna la circonferenza
    cv2.circle(result_img, center, radius, circumference_color, thickness)
    
    # Calcola l'angolo del diametro AB rispetto all'asse x
    angle_AB = math.atan2(point_B[1] - point_A[1], point_B[0] - point_A[0])
    
    # Posizioni dei robot principali (robA e robB agli estremi del diametro)
    robA_pos = point_A
    robB_pos = point_B
    
    if d_rob == 0:
        # Caso speciale: 6 robot equidistanti lungo la circonferenza
        positions = []
        for i in range(6):
            angle = i * (2 * math.pi / 6)  # Angoli equidistanti
            x = center_x + int(radius * math.cos(angle))
            y = center_y + int(radius * math.sin(angle))
            positions.append((x, y))
        
        # Disegna tutti i 6 robot equidistanti
        for i, pos in enumerate(positions):
            cv2.circle(result_img, pos, r_rob, robot_color, -1)
            cv2.putText(result_img, f'R{i+1}', 
                       (pos[0] - 10, pos[1] + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    else:
        # Caso normale: posiziona i robot in base a d_rob
        
        # Disegna robA e robB
        cv2.circle(result_img, robA_pos, r_rob, robot_color, -1)
        cv2.circle(result_img, robB_pos, r_rob, robot_color, -1)
        cv2.putText(result_img, 'robA', 
                   (robA_pos[0] - 15, robA_pos[1] + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result_img, 'robB', 
                   (robB_pos[0] - 15, robB_pos[1] + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Calcola le posizioni dei robot aggiuntivi
        additional_robots = []
        
        # Per robA: trova i punti sulla circonferenza a distanza d_rob
        robA_additional = _find_additional_robots(center, radius, robA_pos, d_rob)
        additional_robots.extend(robA_additional)
        
        # Per robB: trova i punti sulla circonferenza a distanza d_rob
        robB_additional = _find_additional_robots(center, radius, robB_pos, d_rob)
        additional_robots.extend(robB_additional)
        
        # Disegna i robot aggiuntivi
        for i, pos in enumerate(additional_robots):
            cv2.circle(result_img, pos, r_rob, robot_color, -1)
            cv2.putText(result_img, f'R{i+1}', 
                       (pos[0] - 10, pos[1] + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return result_img


def _find_additional_robots(center: Tuple[int, int], 
                          radius: int, 
                          main_robot_pos: Tuple[int, int], 
                          d_rob: float) -> list[Tuple[int, int]]:
    """
    Trova le posizioni dei due robot aggiuntivi sulla circonferenza
    a distanza d_rob dal robot principale.
    
    Args:
        center: Centro della circonferenza
        radius: Raggio della circonferenza
        main_robot_pos: Posizione del robot principale
        d_rob: Distanza desiderata
    
    Returns:
        Lista con le posizioni dei due robot aggiuntivi
    """
    cx, cy = center
    mx, my = main_robot_pos
    
    # Calcola l'angolo del robot principale rispetto al centro
    main_angle = math.atan2(my - cy, mx - cx)
    
    # Calcola l'angolo corrispondente alla distanza d_rob sulla circonferenza
    # Usando la formula: d = 2 * R * sin(θ/2), quindi θ = 2 * arcsin(d/(2*R))
    if d_rob > 2 * radius:
        # Se d_rob è troppo grande, posiziona i robot agli antipodi
        delta_angle = math.pi / 2
    else:
        # Calcola l'angolo esatto
        delta_angle = 2 * math.asin(d_rob / (2 * radius))
    
    # Posizioni dei due robot aggiuntivi
    angle1 = main_angle + delta_angle
    angle2 = main_angle - delta_angle
    
    pos1 = (
        int(cx + radius * math.cos(angle1)),
        int(cy + radius * math.sin(angle1))
    )
    pos2 = (
        int(cx + radius * math.cos(angle2)),
        int(cy + radius * math.sin(angle2))
    )
    
    return [pos1, pos2]


def create_test_image(width: int = 800, height: int = 600) -> np.ndarray:
    """
    Crea un'immagine di test con sfondo bianco.
    
    Args:
        width: Larghezza dell'immagine
        height: Altezza dell'immagine
    
    Returns:
        Immagine vuota
    """
    return np.ones((height, width, 3), dtype=np.uint8) * 255


if __name__ == "__main__":
    # Test della funzione
    
    # Crea un'immagine di test
    test_img = cv2.imread('maps/map_final.png')
    
    # Definisce i punti A e B
    point_A = (1044 + 120, 2140)
    point_B = (1044 - 120, 2140)
    
    # Test 1: Caso normale con d_rob > 0
    print("Test 1: Caso normale (d_rob = 100)")
    result1 = draw_robot_formation(test_img, point_A, point_B, d_rob=0, r_rob=15)
    cv2.imwrite('test_formation_normal.png', result1)
    
    """# Test 2: Caso speciale con d_rob = 0
    print("Test 2: Caso speciale (d_rob = 0)")
    result2 = draw_robot_formation(test_img, point_A, point_B, d_rob=0, r_rob=15)
    cv2.imwrite('test_formation_equidistant.png', result2)
    
    # Test 3: Caso con punti diagonali
    print("Test 3: Punti diagonali")
    point_C = (150, 150)
    point_D = (650, 450)
    result3 = draw_robot_formation(test_img, point_C, point_D, d_rob=80, r_rob=12)
    cv2.imwrite('test_formation_diagonal.png', result3)
    
    print("Test completati! Controlla i file:")
    print("- test_formation_normal.png")
    print("- test_formation_equidistant.png") 
    print("- test_formation_diagonal.png")"""