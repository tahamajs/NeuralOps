@api_view(["POST"])
@permission_classes([IsAuthenticated])
def start_sandbox(request):
    """Launch a new sandbox container."""
    serializer = SandboxSessionSerializer(data=request.data)
    if serializer.is_valid():
        session = serializer.save(user=request.user, status="created")
        # Start Docker container via Celery
        from integrations.tasks import provision_sandbox

        provision_sandbox.delay(str(session.id))
        return Response(SandboxSessionSerializer(session).data, status=201)
    return Response(serializer.errors, status=400)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def sandbox_status(request, session_id):
    session = get_object_or_404(SandboxSession, id=session_id, user=request.user)
    return Response(SandboxSessionSerializer(session).data)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def execute_in_sandbox(request, session_id):
    """Execute a command in the sandbox and return output."""
    session = get_object_or_404(
        SandboxSession, id=session_id, user=request.user, status="running"
    )
    command = request.data.get("command", "")

    # Check if command is allowed
    if session.allowed_commands and command.split()[0] not in session.allowed_commands:
        return Response({"error": "Command not allowed"}, status=403)

    import docker

    client = docker.from_env()
    try:
        container = client.containers.get(session.container_id)
        exit_code, output = container.exec_run(command)
        output_str = output.decode("utf-8", errors="replace")
        SandboxAuditLog.objects.create(
            session=session, command=command, output=output_str, exit_code=exit_code
        )
        return Response({"exit_code": exit_code, "output": output_str})
    except Exception as e:
        return Response({"error": str(e)}, status=500)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def stop_sandbox(request, session_id):
    session = get_object_or_404(SandboxSession, id=session_id, user=request.user)
    import docker

    client = docker.from_env()
    try:
        container = client.containers.get(session.container_id)
        container.stop(timeout=10)
        session.status = "stopped"
        session.stopped_at = timezone.now()
        session.save()
        return Response({"status": "stopped"})
    except Exception as e:
        return Response({"error": str(e)}, status=500)


@shared_task
def provision_sandbox(session_id: str):
    session = SandboxSession.objects.get(id=session_id)
    session.status = "running"
    session.save()
    import docker

    client = docker.from_env()
    container = client.containers.run(
        session.docker_image,
        detach=True,
        name=f"sandbox-{session.id.hex[:12]}",
        mem_limit=f"{session.memory_limit_mb}m",
        nano_cpus=int(session.cpu_limit * 1e9),
        network_mode="none" if not session.network_access else "bridge",
        command="sleep infinity",
    )
    session.container_id = container.id
    session.save()
