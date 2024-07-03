Remove-Item -Path 3b_stdout.txt -ErrorAction SilentlyContinue
Remove-Item -Path 3b_stderr.txt -ErrorAction SilentlyContinue

$ProcessInfo = Start-Process -FilePath 'python' -ArgumentList '3b-ECAPS_classification-NoTuningSCRIPT.py' -RedirectStandardOutput '3b_stdout.txt' -RedirectStandardError '3b_stderr.txt' -NoNewWindow
$ProcessInfo.Id | Out-File -FilePath 'run_3b.pid'
