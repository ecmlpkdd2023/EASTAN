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
('--node_num',type = int, default=307, help = 'Node number')
('--emb_dim', type = int, default=128, help = 'Adaptive spatial embedding dimension')
('--time_slot', type = int, default = 5, help = 'One time-step length [default: 5mins]')
('--P', type = int, default = 12, help = 'History steps')
('--Q', type = int, default = 12, help = 'Prediction steps')
('--K', type = int, default = 4, help = 'Number of attention heads')
('--d', type = int, default = 8, help = 'Dims of each head attention outputs')
('--attn_mask', default=True, help= 'Attention mask operation')
('--train_ratio', type = float, default = 0.6, help = 'Training set [default : 0.6]')
('--val_ratio', type = float, default = 0.2, help = 'Validation set [default : 0.2]')
('--test_ratio', type = float, default = 0.2, help = 'Testing set [default : 0.2]')
('--batch_size', type = int, default = 32, help = 'Batch size')
('--max_epoch', type = int, default = 1000, help = 'Total epoch')
('--patience', type = int, default = 20, help = 'Patience for early stop')
('--learning_rate', type=float, default = 0.001, help = 'Initial learning rate')
('--decay_epoch', type=int, default = 5, help = 'Decay epoch')
('--path', default = './', help = 'Traffic file')
('--dataset', default = 'PEMS04', help = 'Traffic dataset name [PEMS03 || PEMS04 || PEMS07 || PEMS08]')
('--load_model', default = 'F', help = 'Set T if pretrained model is to be loaded before training start')
('--encoder_layer', type = int, default = 2, help = 'Encoder layer num')
('--decoder_layer', type = int, default = 3, help = 'Decoder layer num')
('--spatial_c', type = int, default=5, help = 'Spatial sampling factor [min area: 5-25(PEMS08), max area: 5-125(PEMS07)]')
('--temporal_c', type = int, default=2, help = 'Temporal sampling factor [1,2,3,4,5]')
