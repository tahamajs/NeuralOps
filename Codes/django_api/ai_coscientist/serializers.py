from rest_framework import serializers

from .models import (Hypothesis, HypothesisReview, MetaReview, ResearchSession,
                     TournamentMatch)


class HypothesisReviewSerializer(serializers.ModelSerializer):
    class Meta:
        model = HypothesisReview
        fields = "__all__"


class MetaReviewSerializer(serializers.ModelSerializer):
    class Meta:
        model = MetaReview
        fields = "__all__"


class HypothesisSerializer(serializers.ModelSerializer):
    reviews = HypothesisReviewSerializer(many=True, read_only=True)
    meta_review = MetaReviewSerializer(read_only=True)

    class Meta:
        model = Hypothesis
        fields = "__all__"


class ResearchSessionSerializer(serializers.ModelSerializer):
    hypotheses = HypothesisSerializer(many=True, read_only=True)
    hypothesis_count = serializers.SerializerMethodField()

    class Meta:
        model = ResearchSession
        fields = "__all__"

    def get_hypothesis_count(self, obj):
        return obj.hypotheses.count()


class TournamentMatchSerializer(serializers.ModelSerializer):
    class Meta:
        model = TournamentMatch
        fields = "__all__"
