for seed in 42 123 2026 1026 3600
do
    # python main.py -mode E -type 32 -seed "$seed" -lr 0.005  # E27 with temporal kernel shortened to 15
    python main.py -mode E -type 30 -seed "$seed" -lr 0.005  # E25 with temporal kernel=51
    python main.py -mode E -type 31 -seed "$seed" -lr 0.005  # E25 with temporal kernel=63
    python main.py -mode E -type 29 -seed "$seed" -lr 0.005  # E25 with ReLU activation
    python main.py -mode E -type 27 -seed "$seed" -lr 0.005  # E25 + 4 Hz highpass FIR
    python main.py -mode E -type 28 -seed "$seed" -lr 0.005  # E25 + 30 Hz lowpass FIR
    python main.py -mode E -type 26 -seed "$seed" -lr 0.005  # case 20 architecture, cosine LR decay to 0.0001
    python main.py -mode E -type 0 -seed "$seed" -lr 0.005  # spec convolution with E25 settings
    python main.py -mode E -type 25 -seed "$seed" -lr 0.005  # E25 reference: depthwise + pointwise
    # python main.py -mode E -type 24 -seed "$seed" -lr 0.001  # case 20 architecture, cosine LR decay to 0.0001
    # python main.py -mode E -type 23 -seed "$seed" -lr 0.003  # case 20 architecture, cosine LR decay to 0.0001
    # python main.py -mode E -type 22 -seed "$seed" -lr 0.01  # case 20 architecture, cosine LR decay to 0.0001
    # python main.py -mode E -type 21 -seed "$seed"  # leaky_relu(slope=0.01), dropout=0.25, kernel=31, BN momentum=0.7, classifier max-norm=1.0
    # python main.py -mode E -type 20 -seed "$seed"  # leaky_relu(slope=0.01), dropout=0.35, kernel=31, BN momentum=0.7, classifier max-norm=1.0
    # python main.py -mode E -type 19 -seed "$seed"  # leaky_relu(slope=0.1), dropout=0.35, BN momentum=0.7, classifier max-norm=1.0
    # python main.py -mode E -type 18 -seed "$seed"  # leaky_relu(slope=0.01), dropout=0.35, BN momentum=0.7, classifier max-norm=1.0
    # python main.py -mode E -type 17 -seed "$seed"  # relu, dropout=0.35, BN momentum=0.7, classifier max-norm=1.0
    # python main.py -mode E -type 16 -seed "$seed"  # relu, dropout=0.35, BN momentum=0.5, classifier max-norm=1.0
    # python main.py -mode E -type 15 -seed "$seed"  # relu, dropout=0.35, BN momentum=0.3, classifier max-norm=1.0
    # python main.py -mode E -type 14 -seed "$seed"  # relu, dropout=0.35, BN momentum=0.01, classifier max-norm=1.0
    # python main.py -mode E -type 13 -seed "$seed"  # relu, dropout=0.35, F1=8, D=2, F2=16, classifier max-norm=1.0
    # python main.py -mode E -type 12 -seed "$seed"  # relu, dropout=0.25, classifier max-norm=1.0
    # python main.py -mode E -type 11 -seed "$seed"  # relu, dropout=0.35, classifier max-norm=2.0
    # python main.py -mode E -type 10 -seed "$seed"  # relu, dropout=0.35, classifier max-norm=1.0
    # python main.py -mode E -type 9 -seed "$seed"  # relu, dropout=0.35, classifier max-norm=0.25
    # python main.py -mode E -type 8 -seed $seed -ks 63 # relu with dropout=0.35
    # python main.py -mode E -type 7 -seed $seed  # relu with dropout=0.35
    # python main.py -mode E -type 6 -seed $seed  # relu with dropout=0.1
    # python main.py -mode E -type 5 -seed $seed  # relu with dropout=0.5
    # Case 0 uses spec convolution with E25 settings.
    # python main.py -mode E -type 1 -seed $seed # relu
    # python main.py -mode E -type 2 -seed $seed # leaky relu
    # python main.py -mode E -type 3 -seed $seed # elu with alpha=0.5
    # python main.py -mode E -type 4 -seed $seed # elu with dropout=0.5
    python main.py -mode D -seed $seed
done
