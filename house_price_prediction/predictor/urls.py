from django.urls import path
import predictor.views as views
urlpatterns = [
    path('', views.predict_price, name='predict_price'),
]
