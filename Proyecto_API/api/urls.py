from django.urls import path
from .views import DataView
urlpatterns = [
    path('data/',DataView.as_view(),name='data_list'),
    path('data/<int:id>',DataView.as_view(),name='data_list'),
    path('getKmean/',DataView.getKmean,name='getKmean'),
    path('getTdData/',DataView.getTdData,name='getTdData')
]
