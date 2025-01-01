import numpy as np


def process_data(args, dataset):
    train, val, test = dict(), dict(), dict()
    column_names = dataset.columns
    for col in column_names:
        close_arr = []
        for idx in range(len(dataset)-args.close_size - args.out_dim):
            data = dataset[col].iloc[idx:idx+args.close_size+args.out_dim]
            close_arr.append(data.values)
        cell_arr = np.array(close_arr)
        cell_arr = cell_arr[:, :, np.newaxis]
        test_len = args.test_days * 24 * args.granularity
        val_len = args.val_days * 24 * args.granularity
        train_len = len(cell_arr) - test_len - val_len

        train[col] = cell_arr[:train_len]
        val[col] = cell_arr[train_len:train_len + val_len]
        test[col] = cell_arr[-test_len:]
    return train, val, test


def process_isolated(args, dataset):
    train, val, test = dict(), dict(), dict()
    column_names = dataset.columns

    for col in column_names:
        close_arr, label_arr = [], []
        # close_ext_arr = []

        cell_traffic = dataset[col]
        start_idx = args.close_size
        for idx in range(start_idx, len(dataset) - args.out_dim + 1):
            y_ = [cell_traffic.iloc[idx + i] for i in range(args.out_dim)]
            label_arr.append(y_)

            if args.close_size > 0:
                x_close = [cell_traffic.iloc[idx - c] for c in range(1, args.close_size + 1)]
                close_arr.append(x_close)
                # for t_idx in range(1, args.close_size + 1):
                #     time_index = cell_traffic.index[idx - t_idx]
                #     day = time_index.weekday()
                #     hour = time_index.hour
                #     minute = time_index.minute // 10
                #     weekend = 1 if day >= 5 else 0
                #     close_ext_arr.append((day, hour, minute, weekend))

        cell_arr_close = np.array(close_arr)
        cell_arr_close = cell_arr_close[:, :, np.newaxis]
        cell_label = np.array(label_arr)

        test_len = args.test_days * 24 * args.granularity
        val_len = args.val_days * 24 * args.granularity
        train_len = len(cell_arr_close) - test_len - val_len

        train_x_close = cell_arr_close[:train_len]
        val_x_close = cell_arr_close[train_len:train_len + val_len]
        test_x_close = cell_arr_close[-test_len:]

        train_label = cell_label[:train_len]
        val_label = cell_label[train_len:train_len + val_len]
        test_label = cell_label[-test_len:]

        train[col] = (train_x_close, train_label)
        val[col] = (val_x_close, val_label)
        test[col] = (test_x_close, test_label)

    return train, val, test


def get_warm_up_data(args, data):
    close_arr, label_arr = [], []
    start_idx = args.close_size
    for idx in range(start_idx, len(data) - args.out_dim + 1):
        y_ = [data.iloc[idx + i] for i in range(args.out_dim)]
        label_arr.append(y_)

        if args.close_size > 0:
            x_close = [data.iloc[idx - c] for c in range(1, args.close_size + 1)]
            close_arr.append(x_close)

    cell_arr_close = np.array(close_arr)
    cell_label = np.array(label_arr)
    return cell_arr_close, cell_label