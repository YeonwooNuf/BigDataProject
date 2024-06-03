from django.urls import path
from . import views

urlpatterns = [
    path('endpoint/', views.endpoint_function, name='endpoint-name'),
    path('predict/', views.backend_view),
]
