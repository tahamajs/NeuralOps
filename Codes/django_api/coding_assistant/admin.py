from django.contrib import admin
from .models import CodeGenerationSession, CodeMessage, GeneratedCode, CodeReview, TestGeneration

admin.site.register(CodeGenerationSession)
admin.site.register(CodeMessage)
admin.site.register(GeneratedCode)
admin.site.register(CodeReview)
admin.site.register(TestGeneration)