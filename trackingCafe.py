import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import deque
import sys
import time

# Menambahkan path ke folder SORT
sys.path.append("sort")
from sort import Sort

# Load model YOLO - model untuk deteksi umum
model = YOLO("yolo11n.pt")
print("YOLO model loaded. Available classes:", model.names)

# Initialize camera
camera_index = 0  # Change this based on your camera setup
cap = cv2.VideoCapture(camera_index)

# Optional: Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print(f"Error: Cannot open camera {camera_index}")
    exit()

# Inisialisasi tracker SORT untuk orang
person_tracker = Sort(max_age=50, min_hits=5, iou_threshold=0.2)

# Inisialisasi tracker SORT untuk meja
table_tracker = Sort(max_age=100, min_hits=3, iou_threshold=0.3)

# Dictionary untuk menyimpan data
person_data = {}
table_data = {}
person_at_table = {}

# Untuk motion tracking
person_tracks = {}
motion_threshold = 5  # minimal pergerakan dalam piksel untuk dianggap bergerak

# ID kelas untuk objek yang akan dideteksi (berdasarkan model.names)
PERSON_CLASS_ID = 0        # person
TABLE_CLASS_ID = 60        # dining table
CHAIR_CLASS_ID = 56        # chair

# Fungsi untuk mendapatkan warna unik
np.random.seed(42)
def get_color():
    return tuple(map(int, np.random.randint(100, 255, size=3)))

# Fungsi untuk mengonversi detik ke format yang sesuai
def format_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    if hours > 0:
        return f"{hours} hour {minutes} min"
    elif minutes > 0:
        return f"{minutes} min"
    else:
        return f"{secs} sec"

# Fungsi untuk mendeteksi apakah orang berada di meja
def is_person_at_table(person_box, table_box, threshold=0.2):
    # person_box dan table_box dalam format [x1, y1, x2, y2]
    px1, py1, px2, py2 = person_box
    tx1, ty1, tx2, ty2 = table_box
    
    # Hitung koordinat intersection
    ix1 = max(px1, tx1)
    iy1 = max(py1, ty1)
    ix2 = min(px2, tx2)
    iy2 = min(py2, ty2)
    
    # Cek apakah ada intersection
    if ix1 < ix2 and iy1 < iy2:
        # Hitung area
        person_area = (px2 - px1) * (py2 - py1)
        intersection_area = (ix2 - ix1) * (iy2 - iy1)
        
        # Jika intersection cukup besar, anggap orang berada di meja
        if intersection_area / person_area > threshold:
            return True
    
    # Untuk kasus dimana kaki orang berada dalam area meja
    person_bottom_center = ((px1 + px2) // 2, py2)
    if (tx1 <= person_bottom_center[0] <= tx2 and 
        ty1 <= person_bottom_center[1] <= ty2):
        return True
        
    return False

# Fungsi untuk mendeteksi pergerakan
def detect_motion(prev_pos, curr_pos, threshold=5):
    if prev_pos is None or curr_pos is None:
        return False
    
    # Hitung jarak Euclidean antara posisi sebelumnya dan saat ini
    dist = np.sqrt((prev_pos[0] - curr_pos[0])**2 + (prev_pos[1] - curr_pos[1])**2)
    return dist > threshold

# Inisialisasi tracked_tables sebagai empty array
tracked_tables = np.empty((0, 5))
tracked_persons = np.empty((0, 5))

# Background subtractor untuk deteksi gerakan
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

# Loop pemrosesan frame
frame_count = 0
prev_frame = None
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_count += 1
    current_time = time.time()
    
    # Deteksi objek setiap 3 frame untuk performa lebih baik
    if frame_count % 7 == 0:
        results = model(frame)
        
        # Reset deteksi untuk frame ini
        person_detections = []
        table_detections = []
        
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                conf = box.conf[0].item()
                
                # Filter confidence
                if conf < 0.45:
                    continue
                    
                bbox = list(map(int, box.xyxy[0]))
                
                # Pisahkan deteksi orang dan meja
                if class_id == PERSON_CLASS_ID:
                    person_detections.append(bbox + [conf])
                elif class_id == TABLE_CLASS_ID:
                    table_detections.append(bbox + [conf])
                # Menambahkan kursi sebagai indikator tambahan untuk area meja
                elif class_id == CHAIR_CLASS_ID and conf > 0.6:
                    # Perbesarkan area kursi untuk menjadikannya proxy area meja
                    x1, y1, x2, y2 = bbox
                    width, height = x2 - x1, y2 - y1
                    # Perluas area kursi untuk membuat perkiraan area meja
                    expanded_bbox = [
                        max(0, x1 - width//2),
                        max(0, y1 - height//4),
                        min(frame.shape[1], x2 + width//2),
                        min(frame.shape[0], y2 + height//4)
                    ]
                    table_detections.append(expanded_bbox + [conf])
    
        # Update tracker
        person_detections = np.array(person_detections) if len(person_detections) > 0 else np.empty((0, 5))
        table_detections = np.array(table_detections) if len(table_detections) > 0 else np.empty((0, 5))
        
        tracked_persons = person_tracker.update(person_detections)
        tracked_tables = table_tracker.update(table_detections)
    
    # Deteksi gerakan dengan background subtraction
    fg_mask = bg_subtractor.apply(frame)
    # Dilasi untuk memperjelas area gerakan
    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)
    
    # Hapus overlay dari frame sebelumnya
    clean_frame = frame.copy()
    
    # Gambar area meja yang telah terdeteksi
    for table in tracked_tables:
        tx1, ty1, tx2, ty2, table_id = map(int, table)
        
        # Tambahkan atau update data meja
        if table_id not in table_data:
            table_data[table_id] = {
                "color": get_color(),
                "first_seen": current_time,
                "people_count": 0
            }
        
        table_color = table_data[table_id]["color"]
        
        # Buat overlay semi-transparan untuk area meja
        overlay = clean_frame.copy()
        cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), table_color, -1)
        cv2.addWeighted(overlay, 0.3, clean_frame, 0.7, 0, clean_frame)
        
        # Tampilkan label meja
        table_label = f"Meja {table_id}"
        cv2.putText(clean_frame, table_label, (tx1, ty1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, table_color, 2, cv2.LINE_AA)
    
    # Reset data asosiasi orang-meja untuk frame ini
    current_associations = {}
    active_people = set()
    
    # Proses setiap orang yang terdeteksi
    for person in tracked_persons:
        px1, py1, px2, py2, person_id = map(int, person)
        active_people.add(person_id)
        
        # Hitung tengah orang untuk motion tracking
        center_x, center_y = (px1 + px2) // 2, (py1 + py2) // 2
        current_pos = (center_x, center_y)
        
        # Tambahkan atau update data orang
        if person_id not in person_data:
            person_data[person_id] = {
                "color": get_color(),
                "first_seen": current_time,
                "total_time": 0,
                "is_moving": False,
                "last_move_time": current_time
            }
            person_tracks[person_id] = deque(maxlen=30)  # simpan 30 posisi terakhir
        
        # Ambil data orang
        person_color = person_data[person_id]["color"]
        
        # Update track untuk motion trail
        person_tracks[person_id].append(current_pos)
        
        # Deteksi pergerakan berdasarkan posisi orang
        is_moving = False
        if len(person_tracks[person_id]) >= 2:
            prev_pos = person_tracks[person_id][-2]
            is_moving = detect_motion(prev_pos, current_pos, motion_threshold)
            
            # Update status pergerakan
            person_data[person_id]["is_moving"] = is_moving
            if is_moving:
                person_data[person_id]["last_move_time"] = current_time
        
        # Tampilkan motion trail (jejak pergerakan)
        points = list(person_tracks[person_id])
        if len(points) >= 2:
            for i in range(1, len(points)):
                # Warna jejak semakin transparan untuk posisi lama
                alpha = 0.7 * (i / len(points))
                trail_color = tuple([int(c * alpha) for c in person_color])
                cv2.line(clean_frame, points[i-1], points[i], trail_color, 2)
        
        # Tentukan warna berdasarkan gerakan
        box_color = person_color
        if is_moving:
            # Jika bergerak, gunakan warna asli
            box_color = person_color
        else:
            # Jika diam, kurangi intensitas warna (lebih redup)
            time_since_last_move = current_time - person_data[person_id]["last_move_time"]
            if time_since_last_move > 3:  # tidak bergerak > 3 detik
                box_color = tuple([int(c * 0.5) for c in person_color])  # warna lebih redup untuk orang diam
        
        # Gambar bounding box orang
        cv2.rectangle(clean_frame, (px1, py1), (px2, py2), box_color, 2)
        
        # Cek apakah orang berada di meja
        at_table = False
        for table in tracked_tables:
            tx1, ty1, tx2, ty2, table_id = map(int, table)
            
            if is_person_at_table([px1, py1, px2, py2], [tx1, ty1, tx2, ty2]):
                at_table = True
                
                # Update asosiasi untuk frame ini
                if table_id not in current_associations:
                    current_associations[table_id] = []
                current_associations[table_id].append(person_id)
                
                # Catat waktu orang mulai berada di meja ini
                table_key = f"{person_id}_{table_id}"
                if table_key not in person_at_table:
                    person_at_table[table_key] = current_time
                
                # Hitung waktu yang dihabiskan di meja ini
                time_at_table = int(current_time - person_at_table[table_key])
                table_time = format_time(time_at_table)
                
                # Informasi tambahan untuk orang
                info_text = f"Cust. ID {person_id} | Meja {table_id} | {table_time}"
                # Tambahkan indikator pergerakan
                if is_moving:
                    info_text += " | Moving"
                break
        
        # Jika tidak di meja mana pun
        if not at_table:
            elapsed_time = int(current_time - person_data[person_id]["first_seen"])
            elapsed_formatted = format_time(elapsed_time)
            info_text = f"Cust. ID {person_id} | {elapsed_formatted}"
            # Tambahkan indikator pergerakan
            if is_moving:
                info_text += " | Moving"
            
            # Hapus asosiasi dengan meja jika orang sudah tidak di meja
            for table_key in list(person_at_table.keys()):
                if table_key.startswith(f"{person_id}_"):
                    del person_at_table[table_key]
        
        # Tampilkan informasi di atas bounding box
        cv2.putText(clean_frame, info_text, (px1, py1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2, cv2.LINE_AA)
    
    # Update informasi jumlah orang di setiap meja
    for table_id, people_list in current_associations.items():
        table_data[table_id]["people_count"] = len(people_list)
    
    # Tampilkan jumlah orang di meja
    for table in tracked_tables:
        tx1, ty1, tx2, ty2, table_id = map(int, table)
        table_color = table_data[table_id]["color"]
        people_count = table_data[table_id]["people_count"]
        
        # Hitung total waktu untuk semua orang di meja ini
        total_time = 0
        for table_key in person_at_table:
            if table_key.split('_')[1] == str(table_id):
                total_time += int(current_time - person_at_table[table_key])
        
        total_formatted = format_time(total_time)
        
        # Tampilkan info di bawah meja
        status_text = f"Meja {table_id}: {total_formatted} | {people_count} orang"
        cv2.putText(clean_frame, status_text, (tx1, ty2 + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, table_color, 2, cv2.LINE_AA)
    
    # Tampilkan frame hasil
    cv2.imshow(" TREX ", clean_frame)
    
    # Opsional: Tampilkan mask gerakan
    # cv2.imshow("Motion Detection", fg_mask)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        # Mode debug - tampilkan semua deteksi
        debug_frame = frame.copy()
        debug_results = model(debug_frame)
        for result in debug_results:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                conf = box.conf[0].item()
                
                # Tampilkan semua deteksi
                if conf > 0.3:
                    bbox = list(map(int, box.xyxy[0]))
                    x1, y1, x2, y2 = bbox
                    
                    # Tampilkan semua objek yang terdeteksi
                    object_name = model.names[class_id]
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.putText(debug_frame, f"{object_name} {conf:.2f}", (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
        # Tampilkan frame debug
        cv2.imshow("Debug View: All Detections", debug_frame)
    elif key == ord('m'):
        # Toggle tampilan motion mask
        if cv2.getWindowProperty("Motion Detection", cv2.WND_PROP_VISIBLE) < 1:
            cv2.imshow("Motion Detection", fg_mask)
        else:
            cv2.destroyWindow("Motion Detection")

    prev_frame = frame.copy()

cap.release()
cv2.destroyAllWindows()