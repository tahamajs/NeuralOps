from django.contrib import admin
from .models import Publication, Authorship, Citation, Review

admin.site.register(Publication)
admin.site.register(Authorship)
admin.site.register(Citation)
admin.site.register(Review)