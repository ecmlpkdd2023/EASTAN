# Efficient Adaptive Spatial-Temporal Attention Network for Traffic Flow Forecasting
    # data
        # PEMS0X
            # PEMS0X.npz
            # PEMS0X.csv
    # experiments
        # PEMS0X
            # pre-trained model -> EAST(PEMS0X-params)
            # log file -> EAST-logPEMS0X-params.log
    # scripts
        # run.sh
    # gen_traffic_data.py
    # model.py
    # train.py
    # utils.py
    # readme

## How to run EAST
    # step 0 : unzip data file and put PEMS0X.npz into ./data/PEMS0X

    # step 1 : run gen_traffic_data.py to generate the traffic data as ./data/PEMS0X/PEMS0X.csv
        # command :
            # python3 gen_traffic_data.py

    # step 2 : run train.py for EAST
        # command :
            # python3 train.py --dataset PEMS08 --K 4 --max_epoch 1000 --batch_size 32 --node_num 170 --learning_rate 0.001 --patience 20 --emb_dim 128 --encoder_layer 1 --decoder_layer 1

    # or just run the run.sh after step 1
        # command :
            # sh scripts/run.sh

## Where is the log file
    # The log file and pre-trained model in ./experiments/PEMS0X

## How to fine-tune
    # see train.py for more hyper parameters
