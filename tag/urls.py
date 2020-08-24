from django.urls import path
from .views import(
    GetList,
)

urlpatterns = [
    path('/',GetList.as_view())

]
