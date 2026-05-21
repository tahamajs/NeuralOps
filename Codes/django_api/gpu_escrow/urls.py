# gpu_escrow/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('balance/', views.balance, name='escrow_balance'),
    path('deposit/', views.deposit, name='escrow_deposit'),
    path('withdraw/', views.withdraw, name='escrow_withdraw'),
    path('transactions/', views.transaction_list, name='escrow_transactions'),
]