from django.urls import path
from . import views

urlpatterns = [
    # Public portfolio
    path('user/<str:username>/', views.public_portfolio, name='public_portfolio'),
    path('user/<str:username>/activity/', views.activity_graph, name='activity_graph'),
    path('user/<str:username>/embed/', views.embed_widget, name='embed_widget'),

    # Authenticated
    path('me/', views.my_portfolio, name='my_portfolio'),
    path('me/items/', views.add_showcase_item, name='add_showcase_item'),
    path('me/items/<uuid:item_id>/', views.remove_showcase_item, name='remove_item'),
    path('me/items/<uuid:item_id>/pin/', views.toggle_pin_item, name='toggle_pin'),
    path('me/widget/', views.create_widget, name='create_widget'),

    # Endorsements
    path('user/<str:username>/endorse/', views.endorse_skill, name='endorse'),
    path('user/<str:username>/endorse/<str:skill>/', views.remove_endorsement, name='remove_endorsement'),
]