from django.contrib import admin
from .models import SupportTicket, TicketMessage, Solution, Feedback

admin.site.register(SupportTicket)
admin.site.register(TicketMessage)
admin.site.register(Solution)
admin.site.register(Feedback)