for seed in 42 123 2026 1026 3600
do
    python main.py -mode E -type 10 -seed "$seed"  # relu, dropout=0.35, classifier max-norm=1.0
    # python main.py -mode E -type 9 -seed "$seed"  # relu, dropout=0.35, classifier max-norm=0.25
    # python main.py -mode E -type 8 -seed $seed -ks 63 # relu with dropout=0.35
    # python main.py -mode E -type 7 -seed $seed  # relu with dropout=0.35
    # python main.py -mode E -type 6 -seed $seed  # relu with dropout=0.1
    # python main.py -mode E -type 5 -seed $seed  # relu with dropout=0.5
    # python main.py -mode E -type 0 -seed $seed # elu
    # python main.py -mode E -type 1 -seed $seed # relu
    # python main.py -mode E -type 2 -seed $seed # leaky relu
    # python main.py -mode E -type 3 -seed $seed # elu with alpha=0.5
    # python main.py -mode E -type 4 -seed $seed # elu with dropout=0.5
    # python main.py -mode D -seed $seed
done
