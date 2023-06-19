Git for Age and Gender Estimation from Cctv Camera,

1) Install tensorflow,

pip install -r requirements.txt

2) and then

from main import age_gender_obj

img = cv2.imread(img_path)
result = age_gender_obj.run(img)

The result will be,
{'age':int, 'gender':str}