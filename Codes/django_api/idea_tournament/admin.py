from django.contrib import admin
from .models import IdeaTournament, Hypothesis, HypothesisReview, LiteratureCitation, TournamentLog

admin.site.register(IdeaTournament)
admin.site.register(Hypothesis)
admin.site.register(HypothesisReview)
admin.site.register(LiteratureCitation)
admin.site.register(TournamentLog)