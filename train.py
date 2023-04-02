import math
import argparse
import utils, model
import time, datetime
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--node_num',type = int, default=307, help = 'Node number')
parser.add_argument('--emb_dim', type = int, default=128, help = 'Adaptive spatial embedding dimension')
parser.add_argument('--time_slot', type = int, default = 5, help = 'One time-step length [default: 5mins]')
parser.add_argument('--P', type = int, default = 12, help = 'History steps')
parser.add_argument('--Q', type = int, default = 12, help = 'Prediction steps')
parser.add_argument('--K', type = int, default = 4, help = 'Number of attention heads')
parser.add_argument('--d', type = int, default = 8, help = 'Dims of each head attention outputs')
parser.add_argument('--attn_mask', default=True, help= 'Attention mask operation')
parser.add_argument('--train_ratio', type = float, default = 0.6, help = 'Training set [default : 0.6]')
parser.add_argument('--val_ratio', type = float, default = 0.2, help = 'Validation set [default : 0.2]')
parser.add_argument('--test_ratio', type = float, default = 0.2, help = 'Testing set [default : 0.2]')
parser.add_argument('--batch_size', type = int, default = 32, help = 'Batch size')
parser.add_argument('--max_epoch', type = int, default = 1000, help = 'Total epoch')
parser.add_argument('--patience', type = int, default = 20, help = 'Patience for early stop')
parser.add_argument('--learning_rate', type=float, default = 0.001, help = 'Initial learning rate')
parser.add_argument('--decay_epoch', type=int, default = 5, help = 'Decay epoch')
parser.add_argument('--path', default = './', help = 'Traffic file')
parser.add_argument('--dataset', default = 'PEMS04', help = 'Traffic dataset name [PEMS03 || PEMS04 || PEMS07 || PEMS08]')
parser.add_argument('--load_model', default = 'F', help = 'Set T if pretrained model is to be loaded before training start')

parser.add_argument('--encoder_layer', type = int, default = 2, help = 'Encoder layer num')
parser.add_argument('--decoder_layer', type = int, default = 3, help = 'Decoder layer num')
parser.add_argument('--spatial_c', type = int, default=5, help = 'Spatial sampling factor [min area: 5-25(PEMS08), max area: 5-125(PEMS07)]')
parser.add_argument('--temporal_c', type = int, default=2, help = 'Temporal sampling factor [1,2,3,4,5]')


args = parser.parse_args()
log_time = time.time()
LOG_FILE = args.path+'experiments/'+args.dataset+'/EAST-log'+args.dataset+'-N_'+str(args.node_num)+'-el_'+str(args.encoder_layer)+'-dl_'+str(args.decoder_layer)+'-sc_'+str(args.spatial_c)+'-tc_'+str(args.temporal_c)+'-bs_'+str(args.batch_size)+'-ed_'+str(args.emb_dim)+'-K_'+str(args.K)+'-lr_'+str(args.learning_rate)+'-P_'+str(args.P)+'-Q_'+str(args.Q)+'-d_'+str(args.d)+'-time_'+str(log_time)+'.log'
MODEL_FILE = args.path+'experiments/'+args.dataset+'/EAST('+args.dataset+'-N_'+str(args.node_num)+'-el_'+str(args.encoder_layer)+'-dl_'+str(args.decoder_layer)+'-sc_'+str(args.spatial_c)+'-tc_'+str(args.temporal_c)+'-bs_'+str(args.batch_size)+'-ed_'+str(args.emb_dim)+'-K_'+str(args.K)+'-lr_'+str(args.learning_rate)+'-P_'+str(args.P)+'-Q_'+str(args.Q)+'-d_'+str(args.d)+'-time_'+str(log_time)+')'

start = time.time()

log = open(LOG_FILE, 'w')
utils.log_string(log, str(args)[10 : -1])

# load data
utils.log_string(log, 'loading data...')
(trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, mean, std) = utils.loadData(args)
utils.log_string(log, 'trainX: %s\ttrainY: %s' % (trainX.shape, trainY.shape))
utils.log_string(log, 'valX:   %s\t\tvalY:   %s' % (valX.shape, valY.shape))
utils.log_string(log, 'testX:  %s\t\ttestY:  %s' % (testX.shape, testY.shape))
utils.log_string(log, 'data loaded!')

# use a single gpu 0 for training and testing
device = torch.device(0 if torch.cuda.is_available() else 'cpu')

# for just a single GPU device
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#transform data to tensors
trainX = torch.FloatTensor(trainX).to(device)
trainTE = torch.LongTensor(trainTE).to(device)
trainY = torch.FloatTensor(trainY).to(device)
valX = torch.FloatTensor(valX).to(device)
valTE = torch.LongTensor(valTE).to(device)
valY = torch.FloatTensor(valY).to(device)
testX = torch.FloatTensor(testX).to(device)
testTE = torch.LongTensor(testTE).to(device)
testY = torch.FloatTensor(testY).to(device)

# Gloabal Temporal embedding size
TEmbsize = (24*60//args.time_slot)+7  # number of slots in a day (time-of-day) + number of days in a week (day-of-week)

east = model.EAST(args.K, args.d, args.emb_dim, TEmbsize, args.P, device, args.node_num, args.emb_dim, args.spatial_c, args.temporal_c, args.encoder_layer, args.decoder_layer).to(device)

optimizer = torch.optim.Adam(east.parameters(), lr=args.learning_rate, weight_decay=0.00001)

# Total parameters for EAST
utils.log_string(log, "Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in east.parameters())))


utils.log_string(log, '**** training model ****')
if args.load_model == 'T':
    utils.log_string(log, 'loading pretrained model from %s' % MODEL_FILE)
    east.load_state_dict(torch.load(MODEL_FILE))

num_train, _, N = trainX.shape
num_val = valX.shape[0]
wait = 0
val_loss_min = np.inf

for epoch in range(args.max_epoch):
    if wait >= args.patience:
        utils.log_string(log, 'early stop at epoch: %04d' % (epoch))
        break

    # shuffle
    permutation = np.random.permutation(num_train)
    trainX = trainX[permutation]
    trainTE = trainTE[permutation]
    trainY = trainY[permutation]

    # train loss
    start_train = time.time()
    train_loss = 0
    num_batch = math.ceil(num_train / args.batch_size)
    for batch_idx in range(num_batch):
        east.train()
        optimizer.zero_grad()
        start_idx = batch_idx * args.batch_size
        end_idx = min(num_train, (batch_idx + 1) * args.batch_size)
        batchX = trainX[start_idx : end_idx]
        batchTE = trainTE[start_idx : end_idx]
        batchlabel = trainY[start_idx : end_idx]
        batchpred = east(batchX, batchTE)[0]
        batchloss = model.mae_loss(batchpred, batchlabel, device)
        if (batch_idx+1) % 100 == 0:
            print("Batch: ", batch_idx+1, "out of", num_batch, end=" | ")
            print("Loss: ", batchloss.item(), flush=True)
        batchloss.backward()
        optimizer.step()
        train_loss += batchloss.item() * (end_idx - start_idx)
    train_loss /= num_train
    end_train = time.time()

    # val loss
    start_val = time.time()
    val_loss = 0
    num_batch = math.ceil(num_val / args.batch_size)
    for batch_idx in range(num_batch):
        east.eval()
        start_idx = batch_idx * args.batch_size
        end_idx = min(num_val, (batch_idx + 1) * args.batch_size)
        batchX = valX[start_idx : end_idx]
        batchTE = valTE[start_idx : end_idx]
        batchlabel = valY[start_idx : end_idx]
        batchpred = east(batchX, batchTE)[0] # do not use attn results
        batchloss = model.mae_loss(batchpred, batchlabel, device)
        val_loss += batchloss.item() * (end_idx - start_idx)
    val_loss /= num_val
    end_val = time.time()
    utils.log_string(
        log,
        '%s | epoch: %04d/%d, training time: %.1fs, inference time: %.1fs' %
        (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1,
         args.max_epoch, end_train - start_train, end_val - start_val))
    utils.log_string(
        log, 'train loss: %.4f, val_loss: %.4f' % (train_loss, val_loss))
    if val_loss <= val_loss_min:
        utils.log_string(
            log,
            'val loss decrease from %.4f to %.4f, saving model to %s' %
            (val_loss_min, val_loss, MODEL_FILE))
        wait = 0
        val_loss_min = val_loss
        torch.save(east.state_dict(), MODEL_FILE)
    else:
        wait += 1

# test model
utils.log_string(log, '**** testing model ****')
utils.log_string(log, 'loading model from %s' % MODEL_FILE)
east.load_state_dict(torch.load(MODEL_FILE))
utils.log_string(log, 'model restored!')
utils.log_string(log, 'evaluating...')

num_test = testX.shape[0]

trainPred = []
num_batch = math.ceil(num_train / args.batch_size)
for batch_idx in range(num_batch):
    start_idx = batch_idx * args.batch_size
    end_idx = min(num_train, (batch_idx + 1) * args.batch_size)
    batchX = trainX[start_idx : end_idx]
    batchTE = trainTE[start_idx : end_idx]
    batchlabel = trainY[start_idx : end_idx]
    batchpred,train_spatial_attn, train_temporal_attn, train_fusion_attn = east(batchX, batchTE) # do not use the attn results
    trainPred.append(batchpred.detach().cpu().numpy())
trainPred = np.concatenate(trainPred, axis = 0)

# Total Valid
valPred = []
num_batch = math.ceil(num_val / args.batch_size)
for batch_idx in range(num_batch):
    start_idx = batch_idx * args.batch_size
    end_idx = min(num_val, (batch_idx + 1) * args.batch_size)
    batchX = valX[start_idx : end_idx]
    batchTE = valTE[start_idx : end_idx]
    batchlabel = valY[start_idx : end_idx]
    batchpred,val_spatial_attn,val_temporal_attn, val_fusion_attn = east(batchX, batchTE)
    valPred.append(batchpred.detach().cpu().numpy())
valPred = np.concatenate(valPred, axis = 0)

# Total Test
testPred = []
num_batch = math.ceil(num_test / args.batch_size)
start_test = time.time()
for batch_idx in range(num_batch):
    start_idx = batch_idx * args.batch_size
    end_idx = min(num_test, (batch_idx + 1) * args.batch_size)
    batchX = testX[start_idx : end_idx]
    batchTE = testTE[start_idx : end_idx]
    batchlabel = testY[start_idx : end_idx]
    batchpred,test_spatial_attn,test_temporal_attn, test_fusion_attn = east(batchX, batchTE)
    testPred.append(batchpred.detach().cpu().numpy())
end_test = time.time()
testPred = np.concatenate(testPred, axis = 0)

trainY = trainY.cpu().numpy()
valY = valY.cpu().numpy()
testY = testY.cpu().numpy()

train_mae, train_rmse, train_mape = utils.metric(trainPred, trainY)
val_mae, val_rmse, val_mape = utils.metric(valPred, valY)
test_mae, test_rmse, test_mape = utils.metric(testPred, testY)
utils.log_string(log, 'testing time: %.1fs' % (end_test - start_test))
utils.log_string(log, '                MAE\t\tRMSE\t\tMAPE')
utils.log_string(log, 'train            %.2f\t\t%.2f\t\t%.2f%%' %
                 (train_mae, train_rmse, train_mape * 100))
utils.log_string(log, 'val              %.2f\t\t%.2f\t\t%.2f%%' %
                 (val_mae, val_rmse, val_mape * 100))
utils.log_string(log, 'test             %.2f\t\t%.2f\t\t%.2f%%' %
                 (test_mae, test_rmse, test_mape * 100))
utils.log_string(log, 'performance in each prediction step')

MAE, RMSE, MAPE = [], [], []
for q in range(args.Q):
    mae, rmse, mape = utils.metric(testPred[:, q], testY[:, q])
    MAE.append(mae)
    RMSE.append(rmse)
    MAPE.append(mape)
    utils.log_string(log, 'step: %02d         %.2f\t\t%.2f\t\t%.2f%%' %
                     (q + 1, mae, rmse, mape * 100))
average_mae = np.mean(MAE)
average_rmse = np.mean(RMSE)
average_mape = np.mean(MAPE)
utils.log_string(
    log, 'average:         %.2f\t\t%.2f\t\t%.2f%%' %
    (average_mae, average_rmse, average_mape * 100))
end = time.time()

utils.log_string(log, 'total time: %.1fmin' % ((end - start) / 60))

log.close()
