from django.urls import path
from django.conf.urls import url

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    url(r'^ajax/recognize/', views.recognize, name='recognize')
##    url(r'^ajax/recognize/(?P<url>\w{0,50})/(?P<method>\w{0,50})', views.recognize, name='recognize'),
]
