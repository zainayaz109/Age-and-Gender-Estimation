
# Project Title

The model follows the Soft-Stagewise Regression approach that does classification at stage 1 followed by regression at stage 2 for getting state-of-art results on age and gender estimation on cctv camera.


## Installation

For Setting up the Environment, I have used python==3.8 (recommended)

```bash
  git clone https://github.com/zainayaz109/Age-and-Gender-Estimation.git
  cd Age-and-Gender-Estimation
  conda create --name age-gender python=3.8
  conda activate age-gender
  pip install -r requirements.txt
```
## Weight Files
Download weights from
[link](https://drive.google.com/drive/folders/1giqgDQXnosl1iZaQlOs67YYa3RB4Wtfm?usp=sharing)
and put that in weights folder.

## How to use
At first, import the service_obj from main.py

```bash
  from main import service_obj
  import cv2

  img = cv2.imread(path)
  result = age_gender_obj.run(img)

```
The result will be,
{'age':int, 'gender':str}

    
## Authors

- [Zain Ayaz](https://sites.google.com/view/zainayaz)

