from django.urls import path
from django.conf import settings
from .views import ItineraryGeneratorAPIView, MostPopularDestination, RecommendDestinationsAPIView, TopFiveSimiliarDestinationAPIView

urlpatterns = [
    path('recommend-destinations', RecommendDestinationsAPIView.as_view()),
    path('top-five-similar', TopFiveSimiliarDestinationAPIView.as_view()),
    path('most-popular', MostPopularDestination.as_view()),
    path('itinerary-generator', ItineraryGeneratorAPIView.as_view())
]