from django.contrib import admin

from .models import Feedback, Solution, SupportTicket, TicketMessage

admin.site.register(SupportTicket)
admin.site.register(TicketMessage)
admin.site.register(Solution)
admin.site.register(Feedback)
