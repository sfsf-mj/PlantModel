from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

# تحميل النموذج
model = tf.keras.models.load_model("plant_model.keras")

# def preprocess_image(image):
#     # تغيير حجم الصورة ليناسب النموذج
#     image = image.resize((256, 256))  # التعديل على حسب حجم إدخال النموذج
#     image = np.array(image) / 255.0  # تحويل الصورة إلى مصفوفة وتطبيعها
#     image = np.expand_dims(image, axis=0)
#     return image

# كود المعالجة السابقه مفروض يتعدل على حسب أخر نموذج (معلوووومه مهههههمه) لازم تتعدل
def preprocess_image(image):
    image = image.resize((224, 224))  # تغيير حجم الصورة
    image_array = np.array(image) / 255.0  # تطبيع القيم
    return image_array.reshape(1, 224, 224, 3)  # إعادة تشكيل البيانات


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "لم يتم إرسال ملف"}), 400
    
    file = request.files['file']

    if not file.filename.endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({"error": "نوع الملف غير مدعوم. يرجى تحميل صورة."}), 400

    image = Image.open(file.stream)  # فتح الصورة
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    
    # استخراج النتيجة
    class_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    # إرسال النتيجة
    response = {
        "class_index": int(class_index),
        "confidence": float(confidence)
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True) # لمن يترفع مفروض اشيل debug=True
