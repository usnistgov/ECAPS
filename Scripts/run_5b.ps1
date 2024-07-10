Remove-Item -Path 5b_stdout.txt -ErrorAction SilentlyContinue
Remove-Item -Path 5b_stderr.txt -ErrorAction SilentlyContinue


$ProcessInfo = Start-Process -FilePath 'python' -ArgumentList 'Scripts/5b-ECAPS_classification-AlgorithmSelection.py' -RedirectStandardOutput '5b_stdout.txt' -RedirectStandardError '5b_stderr.txt' -NoNewWindow
$ProcessInfo.Id | Out-File -FilePath 'run_5b.pid'
