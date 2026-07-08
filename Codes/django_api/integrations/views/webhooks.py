@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def webhook_list(request):
    if request.method == "GET":
        subs = WebhookSubscription.objects.filter(user=request.user)
        return Response(WebhookSubscriptionSerializer(subs, many=True).data)
    elif request.method == "POST":
        serializer = WebhookSubscriptionSerializer(data=request.data)
        if serializer.is_valid():
            sub = serializer.save(user=request.user)
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)


@api_view(["DELETE"])
@permission_classes([IsAuthenticated])
def webhook_delete(request, sub_id):
    sub = get_object_or_404(WebhookSubscription, id=sub_id, user=request.user)
    sub.delete()
    return Response(status=204)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def webhook_test(request, sub_id):
    """Send a test event to the webhook URL."""
    sub = get_object_or_404(WebhookSubscription, id=sub_id, user=request.user)
    success = dispatch_webhook(sub, "test", {"message": "Test event from NeuralOps"})
    return Response({"success": success})
