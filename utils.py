import numpy as np
import pandas as pd


# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)


# metric
def metric(pred, label):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        # mape = np.divide(mae, label)
        mape = np.abs(np.divide(np.subtract(pred, label), label)).astype(np.float32)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return mae, rmse, mape


def mask_np(array, null_val):
    if np.isnan(null_val):
        return (~np.isnan(null_val)).astype('float32')
    else:
        return np.not_equal(array, null_val).astype('float32')


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mask_np(y_true, null_val)
        mask /= mask.mean()
        mape = np.abs((y_pred - y_true) / y_true)
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)


def masked_mse_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mse = (y_true - y_pred) ** 2
    return np.mean(np.nan_to_num(mask * mse))


def masked_mae_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mae = np.abs(y_true - y_pred)
    return np.mean(np.nan_to_num(mask * mae))


def seq2instance(data, P, Q):
    num_step, dims = data.shape
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape=(num_sample, P, dims))
    y = np.zeros(shape=(num_sample, Q, dims))
    for i in range(num_sample):
        x[i] = data[i: i + P]
        y[i] = data[i + P: i + P + Q]
    return x, y


def loadData(args):
    # Traffic
    if args.dataset == 'PEMS04':  # 307 nodes for pems04
        TRAFFIC_FILE = args.path + 'data/' + args.dataset + '/' + args.dataset + '.csv'
        df = pd.read_csv(TRAFFIC_FILE, index_col='date', parse_dates=True)
        Traffic = df.values
        print("Initial loaded traffic Shape is: ", Traffic.shape)

    elif args.dataset == 'PEMS08':  # 170 nodes for pems08
        TRAFFIC_FILE = args.path + 'data/' + args.dataset + '/' + args.dataset + '.csv'
        df = pd.read_csv(TRAFFIC_FILE, index_col='date', parse_dates=True)
        Traffic = df.values
        print("Initial loaded traffic Shape is: ", Traffic.shape)

    elif args.dataset == 'PEMS07':  # 883 nodes for pems07
        TRAFFIC_FILE = args.path + 'data/' + args.dataset + '/' + args.dataset + '.csv'
        df = pd.read_csv(TRAFFIC_FILE, index_col='date', parse_dates=True)
        Traffic = df.values
        print("Initial loaded traffic Shape is: ", Traffic.shape)

    elif args.dataset == 'PEMS03':  # 358 nodes for pems03
        TRAFFIC_FILE = args.path + 'data/' + args.dataset + '/' + args.dataset + '.csv'
        df = pd.read_csv(TRAFFIC_FILE, index_col='date', parse_dates=True)
        Traffic = df.values
        print("Initial loaded traffic Shape is: ", Traffic.shape)

    # train/val/test
    num_step = df.shape[0]
    train_steps = round(args.train_ratio * num_step)
    test_steps = round(args.test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps
    train = Traffic[: train_steps]
    val = Traffic[train_steps: train_steps + val_steps]
    test = Traffic[-test_steps:]

    # X, Y
    trainX, trainY = seq2instance(train, args.P, args.Q)
    valX, valY = seq2instance(val, args.P, args.Q)
    testX, testY = seq2instance(test, args.P, args.Q)

    print("trainX Shape is: ", trainX.shape)
    print("trainY Shape is: ", trainY.shape)
    print("valX Shape is: ", valX.shape)
    print("valY Shape is: ", valY.shape)

    # Standard-Normalization
    mean, std = np.mean(trainX), np.std(trainX)
    trainX = (trainX - mean) / std
    valX = (valX - mean) / std
    testX = (testX - mean) / std


    # Global temporal embedding (Global Time Stamp with day_ID and week_ID)
    '''
    The sampling interval of each dataset is known (5mins), and the target of prediction
        is also based on the sampling interval, which is the essence of temporal embedding.

    For global temporal embedding, We encode the time element of day + week, 
        and our goal is to use the data of the past hour(seq_len=12) to predict the next hour(horizon=12).
    '''

    Time = df.index
    dayofweek = np.reshape(Time.weekday, newshape=(-1, 1))
    timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) \
                // 300  # with 5 minutes window
    timeofday = np.reshape(timeofday, newshape=(-1, 1))
    Time = np.concatenate((dayofweek, timeofday), axis=-1)

    # train/val/test
    train = Time[: train_steps]
    val = Time[train_steps: train_steps + val_steps]
    test = Time[-test_steps:]

    # shape = (num_sample, P + Q, 2)
    trainTE = seq2instance(train, args.P, args.Q)
    trainTE = np.concatenate(trainTE, axis=1).astype(np.int32)
    valTE = seq2instance(val, args.P, args.Q)
    valTE = np.concatenate(valTE, axis=1).astype(np.int32)
    testTE = seq2instance(test, args.P, args.Q)
    testTE = np.concatenate(testTE, axis=1).astype(np.int32)

    print("train Shape is: ", train.shape)
    print("trainTE Shape is: ", trainTE.shape)
    print("valTE Shape is: ", valTE.shape)
    return (trainX, trainTE, trainY, valX, valTE, valY,
            testX, testTE, testY, mean, std)
