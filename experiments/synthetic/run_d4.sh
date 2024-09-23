python experiments/synthetic/benchmark_wmc.py -s log -b d4 -r 10 -v 30 35 40 45 50 55 60 65 70 75 80 -t jax -d cpu
python experiments/synthetic/benchmark_wmc.py -s log -b d4 -r 10 -v 30 35 40 45 50 55 60 65 70 75 80 -t jax -d cuda
python experiments/synthetic/benchmark_wmc.py -s log -b d4 -r 10 -v 30 35 40 45 50 55 60 65 70 75 80 -t torch -d cpu
python experiments/synthetic/benchmark_wmc.py -s log -b d4 -r 10 -v 30 35 40 45 50 55 60 65 70 75 80 -t torch -d cuda
