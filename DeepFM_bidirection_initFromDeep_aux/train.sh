nohup python -u train.py > train.log 2>&1 &
echo $! > save_pid.txt
