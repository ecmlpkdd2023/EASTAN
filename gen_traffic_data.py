import pandas as pd
import numpy as np

'''
    Note: Due to upload file size limitations, we can only upload PEMS04. 
'''
def gen_pems_data(dataset):
    if dataset == 'PEMS03': # 9/1/2018-11/30/2018 26208 time-steps
        date_index = pd.date_range(start='20180901', end='20181201', freq='5Min')[:-1]

    elif dataset == 'PEMS04': # 1/1/2018-2/28/2018 16992 time-steps
        date_index = pd.date_range(start='20180101',end='20180301',freq='5Min')[:-1]

    elif dataset == 'PEMS07': # 5/1/2017 28224 time-steps
        date_index = pd.date_range(start='20170501', periods=28225,freq='5Min')[:-1]

    elif dataset == 'PEMS08': # 7/1/2016-8/31/2016 17856 time-steps
        date_index = pd.date_range(start='20160701',end='20160901',freq='5Min')[:-1]
        # print(date_index)

    pems_data = np.load('./data/' + dataset + '/' + dataset + '.npz')['data'][:, :, 0]

    df_pems_data = pd.DataFrame(pems_data,index=date_index).reset_index()

    df_col = df_pems_data.columns.tolist()

    df_new_col = []
    for col_name in df_col:
        if col_name == 'index':
            col_name_new = 'date'
        else:
            col_name_new = 'S_' + str(col_name)
        df_new_col.append(col_name_new)

    df_pems_data.columns = df_new_col
    print(df_pems_data)
    df_pems_data.to_csv('./data/' + dataset + '/' + dataset + '.csv', index=None)
    print( 'Traffic data:{} write success'.format(dataset))


if __name__ == '__main__':
    # PEMS03 || PEMS04 || PEMS07 || PEMS08
    Dataset = 'PEMS04'
    gen_pems_data(Dataset)