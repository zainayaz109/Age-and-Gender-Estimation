import os, cv2, traceback
import numpy as np
from PIL import Image
from gender_model import MobileNetGender
from age_model import SSR_net
from tensorflow.keras.optimizers import Adam
root_dir = os.path.split(os.path.abspath(__file__))[0]

class AgeGenderDetection:
    def __init__(self):
        input_shape = (256, 128, 3)
        image_size = (256,128)
        stage_num = [3,3,3]
        lambda_local = 1
        lambda_d = 1

        self.age_model = SSR_net(image_size,stage_num,1,1)()
        self.gender_model = MobileNetGender(input_shape)()
        
        self.age_model.compile(
            optimizer=Adam(0.0001), 
            loss=["mae"],
            metrics=['mae']
        )
        self.gender_model.compile(
            optimizer=Adam(0.00001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        self.age_model.load_weights(root_dir+'/weights/best_weight_age.h5')
        self.gender_model.load_weights(root_dir+'/weights/best_weight_gender.h5')
        
    def run(self, img):
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.asarray(Image.fromarray(img).resize((128,256)))/255.0
            img = np.expand_dims(img, axis=0)

            age = self.age_model.predict(img)
            gender = np.argmax(self.gender_model.predict(img))
            gender = 'M' if gender == 1 else 'F'

            result = {'age': int(age), 'gender': gender}

            return result
        except:
            print(traceback.print_exc())
            return {'age': None, 'gender': None}
        
age_gender_obj = AgeGenderDetection()