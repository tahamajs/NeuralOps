from django.test import TestCase

# Create your tests here.
def test_scheduler():
    sched = GPUScheduler(total_gpus=2)
    sched.start()
    # Submit jobs with different priorities
    sched.add_job('job1', priority=5, estimated_duration=30)
    sched.add_job('job2', priority=2, estimated_duration=1200)
    sched.add_job('job3', priority=8, estimated_duration=45)
    # Let the scheduler run for a while
    time.sleep(10)
    # Check logs