from django.urls import path

from . import marketplace_views as views


urlpatterns = [
    path('listings/', views.listings),
    path('listings/<uuid:listing_id>/', views.listing_detail),
    path('listings/<uuid:listing_id>/history/', views.listing_history),
    path('listings/<uuid:listing_id>/price_history/', views.listing_price_history),
    path('listings/<uuid:listing_id>/depth/', views.listing_depth),
    path('listings/<uuid:listing_id>/order_book/', views.listing_order_book),
    path('listings/<uuid:listing_id>/place_order/', views.place_order),
    path('depth/', views.depth),
    path('watchlist/', views.watchlist),
    path('watchlist/<uuid:listing_id>/', views.remove_watchlist),
    path('orders/', views.orders),
    path('orders/mine/', views.my_orders),
    path('orders/<uuid:order_id>/cancel/', views.cancel_order),
    path('jobs/', views.jobs),
    path('jobs/<uuid:job_id>/', views.job_detail),
    path('balance/', views.balance),
    path('invoices/', views.invoices),
    path('usage/', views.usage),
    path('leaderboard/', views.leaderboard),
    path('reputation/', views.reputation),
    path('provider/stats/', views.provider_stats),
    path('provider/usage/', views.provider_usage),
    path('provider/revenue/', views.provider_revenue),
]
