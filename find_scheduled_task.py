import subprocess

cmd = ['powershell', '-Command', """
Get-ScheduledTask | ForEach-Object {
    $task = $_
    $act = $task.Actions | Out-String
    if ($act -match "start_ppl_loop" -or $act -match "eval_perplexity" -or $act -match "ppl_loop") {
        [PSCustomObject]@{
            TaskName = $task.TaskName
            TaskPath = $task.TaskPath
            State = $task.State
        }
    }
}
"""]

res = subprocess.run(cmd, capture_output=True, text=True)
print("STDOUT:\n", res.stdout)
print("STDERR:\n", res.stderr)
