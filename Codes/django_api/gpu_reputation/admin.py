from django.contrib import admin
from .models import *

for m in [ReputationScore, Stake, SlashingCondition, SlashingEvent, Dispute,
           JuryMember, JuryVote, ReputationGraph, TrustEdge, EigenTrustResult,
           ReputationBadge, UserReputationBadge]:
    admin.site.register(m)