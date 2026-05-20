import os
import signal
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from .models import TerminalSession


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_sessions(request):
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
    try:
        session = TerminalSession.objects.get(id=session_id)
    except TerminalSession.DoesNotExist:
        return Response({'error': 'Session not found'}, status=status.HTTP_404_NOT_FOUND)

    if session.user != request.user and not request.user.is_staff:
        return Response({'error': 'Not allowed'}, status=status.HTTP_403_FORBIDDEN)

    if session.pid:
        try:
            os.kill(session.pid, signal.SIGKILL)
        except OSError:
            pass
    if session.fd:
        try:
            os.close(session.fd)
        except OSError:
            pass

    session.mark_inactive()
    return Response({'status': 'killed'})