from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from .models import TerminalSession


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_sessions(request):
    """List active terminal sessions for the current user."""
    sessions = TerminalSession.objects.filter(
        user=request.user, is_active=True
    ).order_by('-last_activity_at')
    data = [{
        'id': str(s.id),
        'connected_at': s.connected_at.isoformat(),
        'last_activity_at': s.last_activity_at.isoformat(),
        'pid': s.pid,
        'cols': s.cols,
        'rows': s.rows,
    } for s in sessions]
    return Response(data)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def kill_session(request, session_id):
    """Kill a specific terminal session (own or admin)."""
    try:
        session = TerminalSession.objects.get(id=session_id)
    except TerminalSession.DoesNotExist:
        return Response({'error': 'Session not found'}, status=404)

    if session.user != request.user and not request.user.is_staff:
        return Response({'error': 'Not allowed'}, status=403)

    # Send signal to kill the child process
    if session.pid:
        try:
            import signal
            os.kill(session.pid, signal.SIGKILL)
        except OSError:
            pass

    # Close file descriptor
    if session.fd:
        try:
            import os
            os.close(session.fd)
        except OSError:
            pass

    session.mark_inactive()
    return Response({'status': 'killed'})