from io import BytesIO
import string
import numpy as np
import cv2
from PIL import Image

from django.core.files.uploadedfile import InMemoryUploadedFile

from tensorflow.keras.models import load_model

from ijazahpy.preprocessing import crop_ijazah
from ijazahpy.preprocessing import to_mnist_ar, to_mnist
from ijazahpy.preprocessing import remove_noise_bin
from ijazahpy.preprocessing import prepare_ws_image
from ijazahpy.preprocessing import prepare_for_tr
from ijazahpy.preprocessing import preprocess_for_tesseract
from ijazahpy.segmentation import DotsSegmentation
from ijazahpy.segmentation import WordSegmentation
from ijazahpy.segmentation import segment_characters
from ijazahpy.unit_test import process_label

import pytesseract

def decode_file(file):
    return cv2.imdecode(np.fromstring(file.read(), np.uint8),
                        cv2.IMREAD_COLOR)

def numpy_to_djfile(img_array, file=None):
    pil_img = Image.fromarray(img_array)
    thumb_io = BytesIO()
    pil_img.save(thumb_io, format='JPEG')
    file_ = InMemoryUploadedFile(thumb_io, None, file.name, 'image/jpeg', thumb_io.tell, None)
    return file_

def crop(img):
    return crop_ijazah(img)

def segment_dot_ijazah(og, val=47, dot_size=3, min_width=32):
    img = og.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    dot = DotsSegmentation(rlsa_val=val)
    
    rects = dot.segment(gray, dot_size=dot_size, min_width=min_width)
    segmented_imgs = []
    model = load_model('trained_models/engchars-sgd-100-90.h5')
    for i, rect in enumerate(rects):
        x,y,w,h = rect
        segmented_img = gray[y:y+h, x:x+w]
        label = ''

        cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 0), 2)
        
        # get label
        if x > 200 and x < 400:
            # segment from colored image. for detailEnhance process.
            label_img = og[y:y+h+10, 0:x]
            
            label_img = cv2.cvtColor(cv2.detailEnhance(label_img, sigma_s=10, sigma_r=0.15),
                                     cv2.COLOR_BGR2GRAY)
            
            chars = segment_characters(label_img)
            test_set = []
            for j, entry in enumerate(chars):
                box, char_img = entry[0], entry[1]
                mnist_like = to_mnist(char_img, aspect_ratio=False)
                
                test_set.append(mnist_like)

            test_set = np.asarray(test_set).reshape(-1, 28, 28, 1)
            predicted_y = model.predict(test_set)
        
            for prediction in predicted_y:
                label += string.ascii_letters[prediction.argmax()]
                
        segmented_imgs.append((segmented_img,
                               process_label(label, metrics='ratio', tolerance=0.4),
                               rect))

    return img, segmented_imgs

def segment_char(url, walk=False):
    img = cv2.imread(url[1:], cv2.IMREAD_GRAYSCALE)
    
    char_entries = segment_characters(img, walking_kernel=walk)
    res = []
    for entry in char_entries:
        try:
            mnist_like = to_mnist_ar(entry[1])
            res.append(mnist_like)
        except:
            continue
    return res

def segment_word(url):
    ws = WordSegmentation()
    img = cv2.imread(url[1:], cv2.IMREAD_GRAYSCALE)
    prepared_img = prepare_ws_image(img, 50)
    words = ws.segment(img)

    return words

def recognize_text(url, tr):
    ws = WordSegmentation()
    img = cv2.imread(url[1:], cv2.IMREAD_GRAYSCALE)
    
    prepared_img = prepare_ws_image(img, 50)
    _, prepared_img = cv2.threshold(prepared_img, 
                                128, 
                                255, 
                                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    prepared_img = remove_noise_bin(prepared_img, 30)
    prepared_img = cv2.subtract(255,prepared_img)
    
    words = ws.segment(prepared_img)

    res = []
    for entry in words:
        curr_box, curr_img = entry

        curr_img = cv2.subtract(255,curr_img)

        if curr_img.shape[0] < 40 and curr_img.shape[1] < 40:
            continue
        
        curr_img = prepare_for_tr(curr_img, thresh=False)
        res.append(tr.recognize(curr_img))
        
    return res

def recognize_with_tesseract(url):
    img = cv2.imread(url[1:], cv2.IMREAD_GRAYSCALE)
    return pytesseract.image_to_string(
        preprocess_for_tesseract(img), config='--psm 7')

if __name__ == '__main__':
    print(cv2.__version__)
    img = cv2.imread('G:\\Kuliah\\skripsi\\Project\\Ijazah\\ijazah3.jpg')
    
    entries = segment_dot_ijazah(crop_ijazah(img))
    for e in entries:
        word = e[1]
        if word != '':
            print(word)
