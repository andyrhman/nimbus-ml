from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from .views import RecommendDestinationsAPIView

urlpatterns = [
    path('recommend-destinations', RecommendDestinationsAPIView.as_view())
]