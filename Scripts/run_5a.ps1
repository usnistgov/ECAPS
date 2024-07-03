Remove-Item -Path 5a_stdout.txt -ErrorAction SilentlyContinue
Remove-Item -Path 5a_stderr.txt -ErrorAction SilentlyContinue


$ProcessInfo = Start-Process -FilePath 'python' -ArgumentList '5a-ECAPS_classification-PreProcessingSelection.py' -RedirectStandardOutput '5a_stdout.txt' -RedirectStandardError '5a_stderr.txt' -NoNewWindow
$ProcessInfo.Id | Out-File -FilePath 'run_5a.pid'
