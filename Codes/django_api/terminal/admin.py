from django.contrib import admin
from .models import TerminalSession


@admin.register(TerminalSession)
class TerminalSessionAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'is_active', 'connected_at', 'last_activity_at']
    list_filter = ['is_active']
    search_fields = ['user__username']
    readonly_fields = ['id', 'user', 'pid', 'fd',
                       'cols', 'rows', 'is_active', 'connected_at', 'last_activity_at']
    actions = ['kill_sessions']

    def kill_sessions(self, request, queryset):
        import os, signal
        for session in queryset.filter(is_active=True):
            try:
                if session.pid:
                    os.kill(session.pid, signal.SIGKILL)
                if session.fd:
                    os.close(session.fd)
            except Exception:
                pass
            session.mark_inactive()
        self.message_user(request, f"Killed selected sessions.")
    kill_sessions.short_description = "Kill selected terminal sessions"