from django.urls import path
from . import views

urlpatterns = [
    path('notes/', views.note_list),
    path('notes/<uuid:note_id>/', views.note_detail),
    path('notes/<uuid:note_id>/react/', views.note_reaction),
]