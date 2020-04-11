from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from django.conf import settings

from .helper import decode_file, crop, segment, numpy_to_djfile, segment_char
from ijazahpy.pretrained import CharacterRecognizer, TextRecognizer
from ijazahpy.preprocessing import crop_ijazah
import json

char_recognizer = CharacterRecognizer()
text_recognizer = TextRecognizer()
# Create your views here.
def index(request):
    if request.method == 'POST' and request.FILES['gambar']:
        file = request.FILES['gambar']
        
        img = decode_file(file)
        img = crop_ijazah(img)
        segmented_imgs = segment(img)

        urls = []
        for i, img in enumerate(segmented_imgs):
            djfile = numpy_to_djfile(img, file)
            
            fs = FileSystemStorage()
            filename = fs.save(str(i)+file.name, djfile)
            uploaded_file_url = fs.url(filename)
            urls.append(uploaded_file_url)
            
        return render(request,
                      'ijazah/index.html',
                      {'segmented_image_urls': urls})
    
    return render(request, 'ijazah/index.html')

def recognize(request):
    url = request.GET['url']
    method = request.GET['method']
    walk = bool(request.GET['walk'])
    
    if method == 'text':
        letters = ' '.join(recognize_text(url))
    else:
        char_imgs = segment_char(url, walk=walk)    
        letters = ''
        for mnist_like in char_imgs:
            if method=='character':
                pred = char_recognizer.recognize_char(mnist_like)
                letters += recognizer.prediction_to_char(pred)
            elif method=='digit':
                pred = char_recognizer.recognize_digit(mnist_like)
                letters += recognizer.prediction_to_char(pred)
            
    data={
        'url': url,
        'method': method,
        'walk': walk,
        'result': letters
    }
    return HttpResponse(json.dumps(data), content_type='application/json')
    
