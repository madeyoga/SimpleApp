import numpy as np
import cv2
from PIL import Image

from ijazahpy.preprocessing import crop_ijazah
from ijazahpy.preprocessing import to_mnist_ar
from ijazahpy.preprocessing import remove_noise_bin
from ijazahpy.preprocessing import prepare_ws_image
from ijazahpy.preprocessing import prepare_text_image
from ijazahpy.segmentation import DotsSegmentation
from ijazahpy.segmentation import WordSegmentation
from ijazahpy.segmentation import segment_characters

from django.core.files.uploadedfile import InMemoryUploadedFile

from io import BytesIO

def decode_file(file):
    return cv2.imdecode(np.fromstring(file.read(), np.uint8),
                        cv2.IMREAD_GRAYSCALE)

def numpy_to_djfile(img_array, file=None):
    pil_img = Image.fromarray(img_array)
    thumb_io = BytesIO()
    pil_img.save(thumb_io, format='JPEG')
    file_ = InMemoryUploadedFile(thumb_io, None, file.name, 'image/jpeg', thumb_io.tell, None)
    return file_

def crop(img):
    return  crop_ijazah(img)

def segment(og, val=47):
    img = og.copy()
    dot = DotsSegmentation(rlsa_val=val)

    rects = dot.segment(img)
    segmented_imgs = []
    for i, rect in enumerate(rects):
        x,y,w,h = rect
        segmented_img = img[y:y+h, x:x+w]
        segmented_imgs.append(segmented_img)

    return segmented_imgs

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

def recognize_text(url, tr):
    word = WordSegmentation()
    img = cv2.imread(url[1:], cv2.IMREAD_GRAYSCALE)
    
    prepared_img = prepare_ws_image(img, 50)
    words = ws.segment(img)

    res = []
    for entry in words:
        curr_box, curr_img = entry
        curr_img = remove_noise_bin(curr_img, 10)
        if curr_img.shape[0] < 40 and curr_img.shape[1] < 40:
            continue
        curr_img = prepare_text_image(curr_img, thresh=False)
        res.append(tr.recognize(curr_img))
        
    return res
