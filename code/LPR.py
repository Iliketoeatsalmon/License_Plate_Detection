import cv2
import os
from ultralytics import YOLO
from function.charecter import get_thai_character, data_province, split_license_plate_and_province
os.environ["QT_LOGGING_RULES"] = "*.debug=false"

vehicle_model = YOLO("model/license_plate.pt") 
plate_model = YOLO("model/data_plate.pt")  
#print(plate_model.names)

def get_thai_license_plate_from_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print("!!Can't load picture!!")
        return
    original_frame = frame.copy()
    #frame = cv2.resize(frame, (1280, 720))
    vehicle_results = vehicle_model(frame, conf=0.3, verbose=False)
    detected_classes = []

    for result in vehicle_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # สีเขียว = รถ
            
            car_roi = frame[y1:y2, x1:x2]
            plate_results = plate_model(car_roi, conf=0.3, verbose=False)
            plates = []
            
            for plate in plate_results:
                for plate_box in plate.boxes:
                    px1, py1, px2, py2 = map(int, plate_box.xyxy[0])
                    px1, px2 = px1 + x1, px2 + x1
                    py1, py2 = py1 + y1, py2 + y1
                    plates.append((px1, plate_box.cls, (px1, py1, px2, py2)))

            plates.sort(key=lambda x: x[0])
            
            for plate in plates:
                px1, cls, (x1_plate, y1_plate, x2_plate, y2_plate) = plate
                cv2.rectangle(frame, (x1_plate, y1_plate), (x2_plate, y2_plate), (255, 255, 0), 2)
                clsname = plate_model.names[int(cls)]
                detected_classes.append(clsname)

    for item in detected_classes:
        if item in data_province:
            detected_classes.remove(item)
            detected_classes.append(item)
    
    combined_text = "".join(get_thai_character(newval) for newval in detected_classes)
    license_plate, province = split_license_plate_and_province(combined_text)
    
    print(f"License: {license_plate}, City: {province}")
    cv2.imshow("License Plate Detection", original_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ใช้ภาพเป็นอินพุต
image_path = "CarCrops/track_1.jpg"  # แก้ไขเป็น path ของภาพจริง
get_thai_license_plate_from_image(image_path)
