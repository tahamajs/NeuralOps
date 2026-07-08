@api_view(["GET"])
@permission_classes([IsAuthenticated])
def plugin_list(request):
    """List available plugins."""
    plugins = PluginDefinition.objects.filter(is_active=True)
    return Response(PluginDefinitionSerializer(plugins, many=True).data)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def enable_plugin(request, plugin_slug):
    """Enable a plugin for the current user."""
    plugin = get_object_or_404(PluginDefinition, slug=plugin_slug, is_active=True)
    activation, created = PluginActivation.objects.get_or_create(
        user=request.user,
        plugin=plugin,
        defaults={"config": request.data.get("config", {})},
    )
    if not created:
        activation.is_enabled = True
        activation.save()
    return Response(PluginActivationSerializer(activation).data)


@api_view(["DELETE"])
@permission_classes([IsAuthenticated])
def disable_plugin(request, plugin_slug):
    plugin = get_object_or_404(PluginDefinition, slug=plugin_slug)
    PluginActivation.objects.filter(user=request.user, plugin=plugin).update(
        is_enabled=False
    )
    return Response(status=204)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def plugin_manifest(request, plugin_slug):
    """Return the plugin manifest (for UI injection)."""
    plugin = get_object_or_404(PluginDefinition, slug=plugin_slug, is_active=True)
    activation = PluginActivation.objects.filter(
        user=request.user, plugin=plugin, is_enabled=True
    ).first()
    if not activation:
        return Response({"error": "Plugin not enabled"}, status=403)
    return Response(plugin.manifest)
