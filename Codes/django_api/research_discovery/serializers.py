from rest_framework import serializers
from .models import (
    LiteratureReview,
    LiteratureSection,
    GeneratedHypothesis,
    HypothesisSource,
    CodeExperiment,
    CodeIteration,
    LifeScienceDatabase,
    DatabaseQuery,
    ExperimentDatabaseLink,
    ResearchGapAnalysis,
    SideBySideComparison,
    GrantOpportunity,
    GrantMatch,
    ResearcherGrantProfile,
)


class LiteratureSectionSerializer(serializers.ModelSerializer):
    class Meta:
        model = LiteratureSection
        fields = "__all__"


class LiteratureReviewSerializer(serializers.ModelSerializer):
    sections = LiteratureSectionSerializer(many=True, read_only=True)

    class Meta:
        model = LiteratureReview
        fields = "__all__"


class HypothesisSourceSerializer(serializers.ModelSerializer):
    class Meta:
        model = HypothesisSource
        fields = "__all__"


class GeneratedHypothesisSerializer(serializers.ModelSerializer):
    sources = HypothesisSourceSerializer(many=True, read_only=True)

    class Meta:
        model = GeneratedHypothesis
        fields = "__all__"


class CodeIterationSerializer(serializers.ModelSerializer):
    class Meta:
        model = CodeIteration
        fields = "__all__"


class CodeExperimentSerializer(serializers.ModelSerializer):
    iterations = CodeIterationSerializer(many=True, read_only=True)

    class Meta:
        model = CodeExperiment
        fields = "__all__"


class LifeScienceDatabaseSerializer(serializers.ModelSerializer):
    class Meta:
        model = LifeScienceDatabase
        fields = "__all__"


class DatabaseQuerySerializer(serializers.ModelSerializer):
    class Meta:
        model = DatabaseQuery
        fields = "__all__"


class ExperimentDatabaseLinkSerializer(serializers.ModelSerializer):
    class Meta:
        model = ExperimentDatabaseLink
        fields = "__all__"


class ResearchGapAnalysisSerializer(serializers.ModelSerializer):
    class Meta:
        model = ResearchGapAnalysis
        fields = "__all__"


class SideBySideComparisonSerializer(serializers.ModelSerializer):
    class Meta:
        model = SideBySideComparison
        fields = "__all__"


class GrantOpportunitySerializer(serializers.ModelSerializer):
    class Meta:
        model = GrantOpportunity
        fields = "__all__"


class GrantMatchSerializer(serializers.ModelSerializer):
    grant = GrantOpportunitySerializer(read_only=True)

    class Meta:
        model = GrantMatch
        fields = "__all__"


class ResearcherGrantProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = ResearcherGrantProfile
        fields = "__all__"
