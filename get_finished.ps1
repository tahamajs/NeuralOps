$finished = Get-Content "D:\SVD_Project\results\full_benchmark\queue_stdout.log" | Select-String -Pattern "FINISHED BENCHMARK"
$finished
