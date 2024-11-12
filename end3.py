import os
import cv2
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score

# إعداد حجم الصورة
img_size = 256

# تحديد مسارات بيانات التدريب والتحقق والاختبار
train_path = 'C:\\Users\\salma\\Desktop\\enddata\\training'
val_path = 'C:\\Users\\salma\\Desktop\\enddata\\val'
test_path = 'C:\\Users\\salma\\Desktop\\enddata\\test'

# تصنيفات الفئات
categories = ['Apple___healthy', 'Apple___Black_rot']

# تحميل بيانات
def load_data(path):
    data = []
    for category in categories:
        category_path = os.path.join(path, category)
        class_num = categories.index(category)
        for img in os.listdir(category_path):
            img_array = cv2.imread(os.path.join(category_path, img), cv2.IMREAD_COLOR)
            if img_array is None:
                print(f"Error loading image: {os.path.join(category_path, img)}")
                continue
            img_resized = cv2.resize(img_array, (img_size, img_size))
            data.append([img_resized, class_num])
    random.shuffle(data)
    X = np.array([features for features, label in data]).reshape(-1, img_size, img_size, 3) / 255.0
    y = np.array([label for features, label in data])
    return X, y

# تحميل البيانات
X_train, y_train = load_data(train_path)
X_val, y_val = load_data(val_path)
X_test, y_test = load_data(test_path)

# بناء النموذج
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# تجميع النموذج
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# تدريب النموذج
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val), batch_size=32)

# عدد الصور في كل مجموعة بعد التحميل
print(f"Total images in training set: {X_train.shape[0]}")
print(f"Total images in validation set: {X_val.shape[0]}")
print(f"Total images in test set: {X_test.shape[0]}")

# رسم الدقة والخسارة مع زيادة المسافة بين النقاط
plt.figure(figsize=(12, 5))

# رسم دقة التدريب والتحقق
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.xticks(np.arange(0, len(history.history['accuracy']), step=1))  # زيادة المسافة بين النقاط
plt.legend()
plt.grid()

# رسم خسارة التدريب والتحقق
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(np.arange(0, len(history.history['loss']), step=1))  # زيادة المسافة بين النقاط
plt.legend()
plt.grid()

# عرض الرسوم البيانية
plt.tight_layout()
plt.show()

# تقييم النموذج على مجموعة الاختبار
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# تقييم النموذج على مجموعة التدريب
train_loss, train_acc = model.evaluate(X_train, y_train)
print(f"Train accuracy: {train_acc}")

# تقييم النموذج على مجموعة التحقق
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Validation accuracy: {val_acc}")

# دمج جميع البيانات
X_all = np.concatenate((X_train, X_val, X_test))
y_all = np.concatenate((y_train, y_val, y_test))

# التنبؤ بالفئات على البيانات المدمجة
y_pred_all = (model.predict(X_all) > 0.5).astype("int32")

# حساب مصفوفة التشويش لجميع البيانات
cm_all = confusion_matrix(y_all, y_pred_all)

# رسم مصفوفة التشويش
plt.figure(figsize=(6, 6))
sns.heatmap(cm_all, annot=True, fmt='d', cmap="Blues", xticklabels=categories, yticklabels=categories)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Combined Dataset')
plt.show()

# حساب مصفوفة التشويش كنسبة مئوية لجميع البيانات
cm_all_percentage = cm_all.astype('float') / cm_all.sum(axis=1)[:, np.newaxis] * 100

# رسم مصفوفة التشويش بالنسب المئوية
plt.figure(figsize=(6, 6))
sns.heatmap(cm_all_percentage, annot=True, fmt='.2f', cmap="Blues", xticklabels=categories, yticklabels=categories)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Combined Dataset (Percentage)')
plt.show()
# حساب مقاييس الأداء
accuracy_all = accuracy_score(y_all, y_pred_all)
TN_all, FP_all, FN_all, TP_all = cm_all.ravel()
recall_all = TP_all / (TP_all + FN_all) if (TP_all + FN_all) > 0 else 0
error_rate_all = (FP_all + FN_all) / (TP_all + TN_all + FP_all + FN_all)

TP_healthy = cm_all[0, 0]  # True Positives for Healthy
FP_healthy = cm_all[1, 0]  # False Positives for Healthy

TP_disease = cm_all[1, 1]  # True Positives for Diseased
FP_disease = cm_all[0, 1]  # False Positives for Diseased

# حساب Precision لكل فئة
precision_healthy = TP_healthy / (TP_healthy + FP_healthy) if (TP_healthy + FP_healthy) > 0 else 0
precision_disease = TP_disease / (TP_disease + FP_disease) if (TP_disease + FP_disease) > 0 else 0
# التوقعات الحقيقية والتوقعات التي حصلت عليها من النموذج
y_true = [1, 0, 1, 1, 0, 1]  # ضع القيم الحقيقية للفئات هنا
y_pred = [1, 0, 0, 1, 0, 1]  # ضع القيم المتوقعة من النموذج هنا

# حساب Total Precision
TP_all = cm_all[0, 0] + cm_all[1, 1]  # True Positives للفئات
FP_all = cm_all[0, 1] + cm_all[1, 0]  # False Positives للفئات

# طباعة مقاييس الأداء
print(f"Combined Accuracy: {accuracy_all:.4f}")
print(f"Combined Recall: {recall_all:.4f}")
print(f"Combined Error Rate: {error_rate_all:.4f}")
print(f"Precision (Apple___healthy): {precision_healthy:.4f}")
print(f"Precision (Apple___Black_rot): {precision_disease:.4f}")

# حساب Precision الكلي
precision_total = TP_all / (TP_all + FP_all) if (TP_all + FP_all) > 0 else 0

# طباعة Total Precision
print(f"Total Precision: {precision_total:.4f}")

# إضافة جزء التنبؤ
# تحديد عدد الصور للتنبؤ

# دالة لتحميل صور التنبؤ من مسار مخصص
def load_prediction_images(prediction_path):
    data = []
    for img in os.listdir(prediction_path):
        img_array = cv2.imread(os.path.join(prediction_path, img), cv2.IMREAD_COLOR)
        if img_array is None:
            print(f"Error loading image: {os.path.join(prediction_path, img)}")
            continue
        img_resized = cv2.resize(img_array, (img_size, img_size))
        data.append(img_resized)
    X_pred = np.array(data).reshape(-1, img_size, img_size, 3) / 255.0
    return X_pred

# مسار الصور الجديدة للتنبؤ
prediction_path = 'C:\\users\\salma\\desktop\\i'

# تحميل الصور من مسار التنبؤ
X_pred = load_prediction_images(prediction_path)

# إجراء التنبؤات على الصور الجديدة
predictions = (model.predict(X_pred) > 0.5).astype("int32")

# طباعة التنبؤات للصور الجديدة
print("\nPredictions on New Images:")
for i, pred in enumerate(predictions):
    predicted_label = categories[pred[0]]
    print(f"Image {i + 1}: Predicted label = {predicted_label}")
model.save("end3.keras")

