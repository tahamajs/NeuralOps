import uuid
from django.db import models
from django.conf import settings

User = settings.AUTH_USER_MODEL


class Connection(models.Model):
    """用户之间的连接关系"""
    CONNECTION_TYPES = [
        ('FOLLOW', '关注'),
        ('COLLEAGUE', '同行'),
        ('MENTOR', '指导关系'),
        ('COLLABORATOR', '合作者'),
    ]
    STATUS_CHOICES = [
        ('PENDING', '待确认'),
        ('ACCEPTED', '已接受'),
        ('REJECTED', '已拒绝'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    from_user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='following_connections')
    to_user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='follower_connections')
    connection_type = models.CharField(max_length=20, choices=CONNECTION_TYPES, default='FOLLOW')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = [['from_user', 'to_user', 'connection_type']]

    def __str__(self):
        return f"{self.from_user.username} → {self.to_user.username} ({self.connection_type})"


class ResearchGroup(models.Model):
    """研究小组"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    slug = models.SlugField(unique=True)
    description = models.TextField()
    avatar = models.ImageField(upload_to='group_avatars/', null=True, blank=True)
    members = models.ManyToManyField(User, through='GroupMembership', related_name='research_groups')
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='created_groups')
    is_private = models.BooleanField(default=False)
    member_count = models.PositiveIntegerField(default=0)
    discussion_count = models.PositiveIntegerField(default=0)
    tags = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-member_count', '-created_at']

    def __str__(self):
        return self.name


class GroupMembership(models.Model):
    """用户与研究组的关系"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    group = models.ForeignKey(ResearchGroup, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    role = models.CharField(max_length=20, choices=[('MEMBER', '成员'), ('MODERATOR', '管理员'), ('OWNER', '创建者')], default='MEMBER')
    joined_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [['group', 'user']]

    def __str__(self):
        return f"{self.user.username} in {self.group.name} ({self.role})"


class Discussion(models.Model):
    """小组讨论"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    group = models.ForeignKey(ResearchGroup, on_delete=models.CASCADE, related_name='discussions')
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='discussions')
    title = models.CharField(max_length=300)
    content = models.TextField(blank=True)
    is_pinned = models.BooleanField(default=False)
    is_closed = models.BooleanField(default=False)
    view_count = models.PositiveIntegerField(default=0)
    reply_count = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-is_pinned', '-updated_at']

    def __str__(self):
        return self.title


class DiscussionReply(models.Model):
    """讨论回复"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    discussion = models.ForeignKey(Discussion, on_delete=models.CASCADE, related_name='replies')
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='discussion_replies')
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"Reply by {self.author.username}"


class ActivityFeed(models.Model):
    """动态订阅"""
    ACTIVITY_TYPES = [
        ('PAPER_PUBLISHED', '论文发布'),
        ('PAPER_SAVED', '论文收藏'),
        ('EXPERIMENT_COMPLETED', '实验完成'),
        ('NOTE_CREATED', '笔记创建'),
        ('CONNECTION', '建立联系'),
        ('GROUP_JOINED', '加入小组'),
        ('DISCUSSION_STARTED', '发起讨论'),
        ('REPUTATION_MILESTONE', '声望里程碑'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='activities')
    activity_type = models.CharField(max_length=30, choices=ACTIVITY_TYPES)
    title = models.CharField(max_length=300)
    description = models.TextField(blank=True)
    link = models.URLField(blank=True, max_length=500)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.username}: {self.title}"


class PublicationAlert(models.Model):
    """发布提醒"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='publication_alerts')
    follower = models.ForeignKey(User, on_delete=models.CASCADE, related_name='received_alerts')
    alert_type = models.CharField(max_length=30, choices=ActivityFeed.ACTIVITY_TYPES)
    content = models.TextField()
    link = models.URLField(blank=True, max_length=500)
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Alert for {self.follower.username}"