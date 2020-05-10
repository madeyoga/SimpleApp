from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from django.conf import settings

from .helper import *

from ijazahpy.pretrained import CharacterRecognizer, TextRecognizer
from ijazahpy.preprocessing import crop_ijazah, remove_noise_bin, prepare_for_tr

import json

char_recognizer = CharacterRecognizer()
text_recognizer = TextRecognizer()

# Create your views here.
def index(request):
    if request.method == 'POST' and request.FILES['gambar']:
        file = request.FILES['gambar']
        rlsa_value = int(request.POST['rlsa-value'])
        minimum_width = int(request.POST['minimum-width'])
        dot_size = int(request.POST['dot-size'])

        print(rlsa_value, minimum_width, dot_size)
        
        # decodes a file to a color image
        img = decode_file(file)
        img = crop_ijazah(img)
        
        visual, entries = segment_dot_ijazah(img,
                                             val=rlsa_value,
                                             dot_size=dot_size,
                                             min_width=minimum_width)

        fs = FileSystemStorage()
        filename = fs.save(file.name, numpy_to_djfile(visual, file))
        main_file_url = fs.url(filename)
        
        res = []
        for i, entry in enumerate(entries):
            img = entry[0]
            predicted_label = entry[1]
            (x,y,w,h) = entry[2]
            
            filename = fs.save(str(i)+file.name, numpy_to_djfile(img, file))
            uploaded_file_url = fs.url(filename)
            res.append((uploaded_file_url, predicted_label))
            
        data = {
            'entries': res,
            'main_file_url': main_file_url,
            'rlsa_value': rlsa_value,
            'dot_size': dot_size,
            'minimum_width': minimum_width
            }
        return render(request,
                      'ijazah/index.html',
                      data
                      )
    
    return render(request, 'ijazah/index.html')

def recognize(request):
    url = request.GET['url']
    method = request.GET['method']
    walk = False
    print(request.GET['walk'], bool(int(request.GET['walk'])))
    
    if method == 'Text':
        letters = ' '.join(recognize_text(url, text_recognizer))
    elif method == 'Tesseract':
        letters = recognize_with_tesseract(url)
        if letters == '':
            letters='~Unrecognized~'
    else:
        char_imgs = segment_char(url, walk=walk)    
        letters = ''
        for mnist_like in char_imgs:
            if method=='Character':
                pred = char_recognizer.recognize_char(mnist_like)
                letters += char_recognizer.prediction_to_char(pred)
            elif method=='Digit':
                pred = char_recognizer.recognize_digit(mnist_like)
                letters += char_recognizer.prediction_to_char(pred)
    
    data={
        'url': url,
        'method': method,
        'walk': walk,
        'result': letters
    }
    return HttpResponse(json.dumps(data), content_type='application/json')
    
