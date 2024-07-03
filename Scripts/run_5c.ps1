Remove-Item -Path 5c_stdout.txt -ErrorAction SilentlyContinue
Remove-Item -Path 5c_stderr.txt -ErrorAction SilentlyContinue


$ProcessInfo = Start-Process -FilePath 'python' -ArgumentList './Scripts/5c-ECAPS_classification-LSVC_LOGO.py' -RedirectStandardOutput '5c_stdout.txt' -RedirectStandardError '5c_stderr.txt' -NoNewWindow
$ProcessInfo.Id | Out-File -FilePath 'run_5c.pid'
