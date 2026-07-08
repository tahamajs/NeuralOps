from django.urls import path

from . import views

urlpatterns = [
    # ── Compute Indices ──────────────────────────────────────
    path("indices/", views.index_list, name="index_list"),
    path("indices/<uuid:index_id>/", views.index_detail, name="index_detail"),
    path("indices/<uuid:index_id>/chart/", views.index_chart, name="index_chart"),
    path("indices/recalculate/", views.recalculate_indices, name="recalculate_indices"),
    # ── Futures Contracts ────────────────────────────────────
    path("futures/", views.futures_list, name="futures_list"),
    path("futures/<uuid:contract_id>/", views.futures_detail, name="futures_detail"),
    path(
        "futures/<uuid:contract_id>/order/",
        views.place_futures_order,
        name="place_futures_order",
    ),
    # ── Futures Orders & Positions ───────────────────────────
    path("futures/orders/mine/", views.my_futures_orders, name="my_futures_orders"),
    path(
        "futures/positions/mine/",
        views.my_futures_positions,
        name="my_futures_positions",
    ),
    path(
        "futures/orders/<uuid:order_id>/cancel/",
        views.cancel_futures_order,
        name="cancel_futures_order",
    ),
    # ── Invoices ─────────────────────────────────────────────
    path("invoices/", views.invoice_list, name="invoice_list"),
    path("invoices/<uuid:invoice_id>/", views.invoice_detail, name="invoice_detail"),
    # ── Payouts ──────────────────────────────────────────────
    path("payouts/", views.payout_list, name="payout_list"),
    path("payouts/request/", views.request_payout, name="request_payout"),
    # ── Payment Methods ──────────────────────────────────────
    path("payment-methods/", views.payment_method_list, name="payment_method_list"),
    path(
        "payment-methods/<uuid:method_id>/",
        views.payment_method_delete,
        name="payment_method_delete",
    ),
    # ── Subscriptions & Plans ────────────────────────────────
    path("plans/", views.plan_list, name="plan_list"),
    path("subscription/", views.my_subscription, name="my_subscription"),
    path("subscription/change/<uuid:plan_id>/", views.change_plan, name="change_plan"),
    # ── Usage ────────────────────────────────────────────────
    path("usage/", views.my_usage, name="my_usage"),
    # ── Dashboards ───────────────────────────────────────────
    path(
        "dashboard/consumer/",
        views.consumer_finance_dashboard,
        name="consumer_finance_dashboard",
    ),
    path(
        "dashboard/provider/",
        views.provider_finance_dashboard,
        name="provider_finance_dashboard",
    ),
]
