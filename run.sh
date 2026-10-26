for seed in 42 123 2026 1026 3600
do
    python main.py -mode E -type 0 -seed $seed
    python main.py -mode E -type 1 -seed $seed
    python main.py -mode E -type 2 -seed $seed
    python main.py -mode E -type 3 -seed $seed
    python main.py -mode E -type 4 -seed $seed
    python main.py -mode D -seed $seed
done