import cv2
import face_recognition
from mtcnn.mtcnn import MTCNN
from PIL import Image, ImageDraw
import pandas as pd
from datetime import datetime
import numpy as np

# تحميل الوجوه المعروفة وأسمائها
def load_known_faces():
    known_face_encodings = []
    known_face_names = []
    
    # إضافة الصور المعروفة
    known_faces = [
        ("./known/barakat.jpg", "Mohamed Barkat"),
        ("./known/treka.jpg", "Mohamed Abu Treka"),
        ("./known/tamer.jpg", "tamer husne"),
        ("./known/ameer.jpg", "ameer mahdi"),  # إضافة وجه وصورة ameer mahdi
    ]
    
    for path, name in known_faces:
        image = face_recognition.load_image_file(path)
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(name)
    
    return known_face_encodings, known_face_names

# تهيئة ملف Excel للحضور
def initialize_excel():
    columns = ["Name", "Time", "Registered"]
    df = pd.DataFrame(columns=columns)
    df.to_excel("attendance.xlsx", index=False, engine='openpyxl')

# تسجيل الحضور في ملف Excel
def log_attendance(name, recorded_names):
    # إذا لم يتم تسجيل حضور الاسم من قبل
    if name not in recorded_names:
        # قراءة ملف Excel الحالي
        df = pd.read_excel("attendance.xlsx", engine='openpyxl')
        # تسجيل الوقت الحالي
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # إضافة صف جديد بالاسم والوقت وقيمة "Yes" في عمود "Registered"
        new_row = pd.DataFrame({"Name": [name], "Time": [current_time], "Registered": ["Yes"]})
        # دمج الصف الجديد مع بيانات Excel الحالية
        df = pd.concat([df, new_row], ignore_index=True)
        # كتابة ملف Excel
        df.to_excel("attendance.xlsx", index=False, engine='openpyxl')
        # إضافة الاسم إلى المجموعة
        recorded_names.add(name)

# تحميل الوجوه المعروفة
known_face_encodings, known_face_names = load_known_faces()

# تهيئة ملف Excel للحضور
initialize_excel()

# تهيئة الكاميرا
camera = cv2.VideoCapture(0)

# مجموعة لتتبع الأسماء التي تم تسجيل حضورها
recorded_names = set()

# تحديد العتبة للمطابقة
threshold = 0.6  # يمكن تغيير القيمة حسب دقة التعرف المطلوبة

# بدء الدورة لالتقاط الصور من الكاميرا
while True:
    # التقاط الإطار من الكاميرا
    ret, frame = camera.read()
    if not ret:
        print("فشل في التقاط الإطار من الكاميرا")
        break

    # تحويل الصورة إلى RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # استخدام MTCNN لاكتشاف الوجوه
    detector = MTCNN()
    faces = detector.detect_faces(np.array(rgb_frame))

    # إعداد مواقع الوجوه
    face_locations = [(face['box'][1], face['box'][0] + face['box'][2], face['box'][1] + face['box'][3], face['box'][0]) for face in faces]

    # إيجاد تشفيرات الوجوه في الإطار
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # إنشاء كائن للرسم على الصورة
    pil_image = Image.fromarray(rgb_frame)
    draw = ImageDraw.Draw(pil_image)

    # حلقة للتعامل مع الوجوه المكتشفة
    for face_location, face_encoding in zip(face_locations, face_encodings):
        # مقارنة تشفيرات الوجوه مع الوجوه المعروفة
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=threshold)
        name = "Unknown"
        
        # حساب مسافة الوجه وإيجاد أفضل تطابق
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        # التحقق مما إذا كانت المطابقة صحيحة بناءً على العتبة
        if matches[best_match_index] and face_distance[best_match_index] < threshold:
            name = known_face_names[best_match_index]

            # تسجيل الحضور مرة واحدة فقط
            log_attendance(name, recorded_names)

        # رسم مربع وتسمية على الوجه المكتشف
        top, right, bottom, left = face_location
        draw.rectangle([(left, top), (right, bottom)], outline=(0, 0, 255), width=2)
        draw.rectangle([(left, bottom - 20), (right, bottom)], fill=(0, 0, 255))
        draw.text((left + 6, bottom - 20), name, fill=(255, 255, 255))

    # عرض الصورة
    cv2.imshow('Live Video', cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR))

    # إنهاء البرنامج عند الضغط على مفتاح 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# تحرير الكاميرا وإنهاء البرنامج
camera.release()
cv2.destroyAllWindows()
