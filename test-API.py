from PIL import Image
import requests

# حدد مسار الصورة
image_path = 'C:/Users/workstation/Desktop/plantModel/Dataset/Single_prediction/b2.jpg'

# افتح الصورة واطبعها
image = Image.open(image_path)
image.show()

# حدد عنوان URL لواجهة برمجة التطبيقات
url = 'http://127.0.0.1:5000/predict'

# قم بإرسال الصورة
with open(image_path, 'rb') as image_file:
    response = requests.post(url, files={'file': image_file})

# اطبع الاستجابة
print(response.json())


# import requests

# url = 'http://127.0.0.1:5000/predict'
# files = {'file': open('C:/Users/workstation/Desktop/plantModel/Dataset/Single_prediction/b2.jpg', 'rb')}

# response = requests.post(url, files=files)
# # print(response.json())
# print(response.text)