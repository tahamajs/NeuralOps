from django.urls import path
from . import views

urlpatterns = [
    # ── Reference Data ────────────────────────────────────────
    path('gpu-models/', views.gpu_model_list, name='gpu_model_list'),

    # ── Listings (Offers) ─────────────────────────────────────
    path('listings/', views.listing_list, name='listing_list'),
    path('listings/mine/', views.my_listings, name='my_listings'),
    path('listings/<uuid:listing_id>/', views.listing_detail, name='listing_detail'),
    path('listings/<uuid:listing_id>/toggle/', views.listing_toggle_online, name='listing_toggle'),
    path('listings/<uuid:listing_id>/favorite/', views.favorite_listing, name='favorite_listing'),
    path('listings/<uuid:listing_id>/unfavorite/', views.unfavorite_listing, name='unfavorite_listing'),

    # ── Orders ────────────────────────────────────────────────
    path('orders/', views.order_list, name='order_list'),
    path('orders/mine/', views.my_orders, name='my_orders'),
    path('orders/<uuid:order_id>/', views.order_detail, name='order_detail'),
    path('orders/<uuid:order_id>/bids/', views.order_bids, name='order_bids'),
    path('orders/<uuid:order_id>/bid/', views.place_bid, name='place_bid'),

    # ── Jobs ──────────────────────────────────────────────────
    path('jobs/', views.job_list, name='job_list'),
    path('jobs/<uuid:job_id>/', views.job_detail, name='job_detail'),
    path('jobs/<uuid:job_id>/complete/', views.job_complete, name='job_complete'),
    path('jobs/<uuid:job_id>/fail/', views.job_fail, name='job_fail'),
    path('jobs/<uuid:job_id>/dispute/', views.job_dispute, name='job_dispute'),
    path('jobs/<uuid:job_id>/provision/', views.provision_job_view, name='job_provision'),

    # ── Favorites ─────────────────────────────────────────────
    path('favorites/', views.my_favorites, name='my_favorites'),

    # ── Pricing ───────────────────────────────────────────────
    path('pricing/', views.pricing_rule_list, name='pricing_rule_list'),

    # ── Stats ─────────────────────────────────────────────────
    path('stats/', views.marketplace_stats, name='marketplace_stats'),
    path('dashboard/provider/', views.provider_dashboard, name='provider_dashboard'),



    # ── Cross-app integration ──────────────────────────────────
    path('listings/<uuid:listing_id>/experiments/', views.listing_linked_experiments),
    path('listings/<uuid:listing_id>/notes/', views.listing_linked_notes),
    path('listings/<uuid:listing_id>/portfolio/', views.provider_portfolio),
    path('notifications/', views.marketplace_notifications),

    
    # ── Provider Agent Endpoints ────────────────────────────────
    path('listings/register/', views.register_listing, name='register_listing'),
    path('listings/update-status/', views.agent_update_status, name='agent_update_status'),
    path('listings/heartbeat/', views.agent_heartbeat, name='agent_heartbeat'),
    path('jobs/<uuid:job_id>/status/', views.job_update_status, name='job_update_status'),
    path('jobs/<uuid:job_id>/logs/', views.job_update_logs, name='job_update_logs'),
    path('agent/check-update/', views.agent_check_update, name='agent_check_update'),
    path('agent/latest/', views.agent_latest_version, name='agent_latest_version'),



    
]


from .trading_views import TradeOrderViewSet, PortfolioViewSet
from rest_framework.routers import DefaultRouter

router = DefaultRouter()
router.register(r'trading/orders', TradeOrderViewSet, basename='trading-order')
router.register(r'trading/portfolio', PortfolioViewSet, basename='trading-portfolio')
urlpatterns += router.urls