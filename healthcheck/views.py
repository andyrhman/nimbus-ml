from rest_framework.views import APIView
from django.shortcuts import render

class HealthCheckView(APIView):
    def get(self, request):
        return render(request, "index.html")
