import pandas
import sklearn.preprocessing
import random
import numpy
import re
import lightgbm
import sys
import os
import tensorflow
import argparse

def size_common(last_or_shoe):
    if last_or_shoe == 'shoe':
        select = 'sku'
    elif last_or_shoe == 'last':
        select = 'styleno'
    all_data = pandas.read_pickle('../data/size_data')
    all_data['sku'] = all_data['sku'].apply(lambda x:x[-3])
    gb_sex = all_data.groupby(by='sex')
    for one_sex in gb_sex:
        if one_sex[0] == 1:
            continue
        positive = one_sex[1][one_sex[1]['pick'] == 1]
        negative = one_sex[1][one_sex[1]['pick'] == 0]
        negative = negative.sample(n=positive.shape[0])
        data = positive.append(negative)
        all_phone = data['phone'].drop_duplicates().tolist()
        all_select = data[select].drop_duplicates().tolist()
        if last_or_shoe == 'last':
            test_select = random.sample(all_select, 2)
            test_phone = random.sample(all_phone, data['phone'].drop_duplicates().shape[0] // 10)
        elif last_or_shoe == 'shoe':
            test_select = random.sample(all_select, 1)
            test_phone = random.sample(all_phone, data['phone'].drop_duplicates().shape[0] // 10)
        train_data = pandas.DataFrame(columns=data.columns)
        test_data = pandas.DataFrame(columns=data.columns)
        test_data = test_data.append(data[data[select].isin(test_select)])
        data = data[~data[select].isin(test_select)]
        test_data = test_data.append(data[data['phone'].isin(test_phone)])
        data = data[~data['phone'].isin(test_phone)]
        test_data = test_data.drop_duplicates()
        test_data = test_data[test_data['pick'] == 1]
        train_data = data
        train_data = train_data.sample(n=train_data.shape[0])
        test_data = test_data.sample(n=test_data.shape[0])
        train_label = train_data['pick'].astype(int)
        train_data = train_data.drop(columns=['sku', 'phone', 'styleno', 'pick', 'sex'])
        test_label = test_data['pick'].astype(int)
        test_data = test_data.drop(columns=['sku', 'phone', 'styleno', 'pick', 'sex'])

        pandas.to_pickle(train_data.columns.values.tolist(), '../train_size/intermediate/size_dimension_name')

    return train_data, train_label, test_data, test_label

def size_common_ditermine(last_or_shoe, man_or_woman, train_phone, validate_phone, unselect):
    if last_or_shoe == 'shoe':
        select = 'sku'
    elif last_or_shoe == 'last':
        select = 'styleno'
    if man_or_woman=='man':
        sex=1
    elif man_or_woman=='woman':
        sex=2
    all_data = pandas.read_pickle('../data/size_data')
    test_phone = pandas.read_pickle('intermediate/test_phone_%s_%s'%(last_or_shoe,man_or_woman))
    all_data = all_data[~all_data['phone'].isin(test_phone)]
    train2_phone = pandas.read_pickle('intermediate/train2_phone_%s_%s'%(last_or_shoe,man_or_woman))
    all_data = all_data[~all_data['phone'].isin(train2_phone)]
    all_data['sku'] = all_data['sku'].apply(lambda x:x[-3])
    gb_sex = all_data.groupby(by='sex')
    for one_sex in gb_sex:
        if one_sex[0] != sex:
            continue
        positive = one_sex[1][one_sex[1]['pick'] == 1]
        negative = one_sex[1][one_sex[1]['pick'] == 0]
        negative = negative.sample(n=positive.shape[0])
        data = positive.append(negative)
        all_phone = data['phone'].drop_duplicates().tolist()
        all_select = data[select].drop_duplicates().tolist()
        train_data = pandas.DataFrame(columns=data.columns)
        test_data = pandas.DataFrame(columns=data.columns)
        test_data = test_data.append(data[data[select].isin(unselect)])
        data = data[~data[select].isin(unselect)]
        test_data = test_data.append(data[data['phone'].isin(validate_phone)])
        data = data[data['phone'].isin(train_phone)]
        train_data = data
        train_data = train_data.sample(n=train_data.shape[0])
        test_data = test_data.sample(n=test_data.shape[0])
        train_label = train_data['pick'].astype(int)
        train_data = train_data.drop(columns=['sku', 'phone', 'styleno', 'pick', 'sex'])
        test_label = test_data['pick'].astype(int)
        test_data = test_data.drop(columns=['sku', 'phone', 'styleno', 'pick', 'sex'])
        pandas.to_pickle(train_data.columns.values.tolist(), '../train_size/intermediate/size_dimension_name')
        return train_data, train_label, test_data, test_label

def size_1d(last_or_shoe):
    train_data, train_label, test_data, test_label = size_common(last_or_shoe)

    OneHotEncoder = sklearn.preprocessing.OneHotEncoder()
    OneHotEncoder.fit(train_label.as_matrix().reshape(-1,1))
    train_label = OneHotEncoder.transform(train_label.as_matrix().reshape(-1,1)).toarray().astype(numpy.float32)
    test_label = OneHotEncoder.transform(test_label.as_matrix().reshape(-1,1)).toarray().astype(numpy.float32)
    StandardScaler = sklearn.preprocessing.StandardScaler()
    StandardScaler.fit(train_data)
    train_data = StandardScaler.transform(train_data).reshape(-1,147,1).astype(numpy.float32)
    test_data = StandardScaler.transform(test_data).reshape(-1,147,1).astype(numpy.float32)

    return train_data, train_label, test_data, test_label

def size_2d(last_or_shoe):
    train_data, train_label, test_data, test_label = size_common(last_or_shoe)

    OneHotEncoder = sklearn.preprocessing.OneHotEncoder()
    OneHotEncoder.fit(train_label.as_matrix().reshape(-1,1))
    train_label = OneHotEncoder.transform(train_label.as_matrix().reshape(-1,1)).toarray().astype(numpy.float32)
    test_label = OneHotEncoder.transform(test_label.as_matrix().reshape(-1,1)).toarray().astype(numpy.float32)
    StandardScaler = sklearn.preprocessing.StandardScaler()
    StandardScaler.fit(train_data)
    train_data = StandardScaler.transform(train_data)
    train_data = numpy.concatenate((train_data, numpy.zeros((train_data.shape[0], 22))), axis=1).reshape(-1,13,13,1).astype(numpy.float32)
    test_data = StandardScaler.transform(test_data)
    test_data = numpy.concatenate((test_data,numpy.zeros((test_data.shape[0],22))), axis=1).reshape(-1,13,13,1).astype(numpy.float32)

    return train_data, train_label, test_data, test_label

def size_3d(last_or_shoe):
    train_data, train_label, test_data, test_label = size_common(last_or_shoe)

    OneHotEncoder = sklearn.preprocessing.OneHotEncoder()
    OneHotEncoder.fit(train_label.as_matrix().reshape(-1,1))
    train_label = OneHotEncoder.transform(train_label.as_matrix().reshape(-1,1)).toarray().astype(numpy.float32)
    test_label = OneHotEncoder.transform(test_label.as_matrix().reshape(-1,1)).toarray().astype(numpy.float32)
    StandardScaler = sklearn.preprocessing.StandardScaler()
    StandardScaler.fit(train_data)
    train_data = StandardScaler.transform(train_data)
    train_data = numpy.concatenate((train_data, numpy.zeros((train_data.shape[0], 69))), axis=1).reshape(-1,6,6,6,1).astype(numpy.float32)
    test_data = StandardScaler.transform(test_data)
    test_data = numpy.concatenate((test_data,numpy.zeros((test_data.shape[0],69))), axis=1).reshape(-1,6,6,6,1).astype(numpy.float32)

    return train_data, train_label, test_data, test_label

def size_mlp(last_or_shoe):
    train_data, train_label, test_data, test_label = size_common(last_or_shoe)

    OneHotEncoder = sklearn.preprocessing.OneHotEncoder()
    OneHotEncoder.fit(train_label.as_matrix().reshape(-1,1))
    train_label = OneHotEncoder.transform(train_label.as_matrix().reshape(-1,1)).toarray().astype(numpy.float32)
    test_label = OneHotEncoder.transform(test_label.as_matrix().reshape(-1,1)).toarray().astype(numpy.float32)
    StandardScaler = sklearn.preprocessing.StandardScaler()
    StandardScaler.fit(train_data)
    pandas.to_pickle(StandardScaler, 'intermediate/StandardScaler_mlp_%s' % last_or_shoe)
    train_data = StandardScaler.transform(train_data).reshape(-1,147).astype(numpy.float32)
    test_data = StandardScaler.transform(test_data).reshape(-1,147).astype(numpy.float32)

    return train_data, train_label, test_data, test_label

def size_mlp_ditermine(last_or_shoe, man_or_woman, train_phone, validate_phone, unselect, times):
    train_data, train_label, test_data, test_label = size_common_ditermine(last_or_shoe, man_or_woman, train_phone, validate_phone, unselect)

    OneHotEncoder = sklearn.preprocessing.OneHotEncoder()
    OneHotEncoder.fit(train_label.as_matrix().reshape(-1,1))
    train_label = OneHotEncoder.transform(train_label.as_matrix().reshape(-1,1)).toarray().astype(numpy.float32)
    test_label = OneHotEncoder.transform(test_label.as_matrix().reshape(-1,1)).toarray().astype(numpy.float32)
    StandardScaler = sklearn.preprocessing.StandardScaler()
    StandardScaler.fit(train_data)
    pandas.to_pickle(StandardScaler, 'intermediate/StandardScaler_mlp_%s_%s_%s' % (last_or_shoe, times,man_or_woman))
    train_data = StandardScaler.transform(train_data).reshape(-1,147).astype(numpy.float32)
    test_data = StandardScaler.transform(test_data).reshape(-1,147).astype(numpy.float32)

    return train_data, train_label, test_data, test_label

def size_stack_mlp(last_or_shoe):
    all_data = pandas.read_pickle('../data/size_data')
    all_data['sku'] = all_data['sku'].apply(lambda x:x[-3])
    gb_sex = all_data.groupby(by='sex')
    for one_sex in gb_sex:
        if one_sex[0] == 1:
            continue
        positive = one_sex[1][one_sex[1]['pick'] == 1]
        negative = one_sex[1][one_sex[1]['pick'] == 0]
        negative = negative.sample(n=positive.shape[0])
        data = positive.append(negative)

        phone=pandas.read_pickle('intermediate/train2_phone_%s'%last_or_shoe)
        data=data[data['phone'].isin(phone)]
        data = data.sample(n=data.shape[0])
        label = data['pick'].astype(int)
        data = data.drop(columns=['sku', 'phone', 'styleno', 'pick', 'sex'])
        k = numpy.zeros([data.shape[0],1])
        for times in range(4):
            flag = None
            parser = argparse.ArgumentParser()
            parser.add_argument('--device', default='/cpu:0')
            parser.add_argument('--summary_dir', default='summary/')
            parser.add_argument('--model_dir', default='model/')
            parser.add_argument('--input_dimension', default=147)
            parser.add_argument('--fc1_dimension', default=147)
            parser.add_argument('--output_dimension', default=2)
            flag, unparsed = parser.parse_known_args()

            StandardScaler = pandas.read_pickle('intermediate/StandardScaler_mlp_%s_%s' % (last_or_shoe, times))
            temp = StandardScaler.transform(data).astype(numpy.float32)
            with tensorflow.device(flag.device):
                w1 = tensorflow.contrib.eager.Variable(tensorflow.truncated_normal([flag.input_dimension, flag.fc1_dimension], stddev=0.1), name='w1')
                b1 = tensorflow.contrib.eager.Variable(tensorflow.constant(0.1, shape = flag.fc1_dimension), name='b1')
                w6 = tensorflow.contrib.eager.Variable(tensorflow.truncated_normal([flag.fc1_dimension, flag.output_dimension], stddev=0.1), name='w6')
                b6 = tensorflow.contrib.eager.Variable(tensorflow.constant(0.1, shape = flag.output_dimension), name='b6')

            def model(x):
                h1 = tensorflow.nn.relu(tensorflow.matmul(x, w1) + b1)
                h6 = tensorflow.matmul(h1, w6) + b6
                return h6

            Saver = tensorflow.contrib.eager.Saver([w1, w6, b1, b6])
            Saver.restore('model/mlp_%s_%s'%(last_or_shoe, times))
            y = model(temp)
            y=tensorflow.nn.softmax(y).numpy()
            k = numpy.concatenate([k,y[:,1].reshape(-1,1)],axis=1)

        for times in range(4):
            model = pandas.read_pickle('model/lgbm_%s_%s' % (last_or_shoe, times))
            StandardScaler = pandas.read_pickle('intermediate/StandardScaler_lgbm_%s_%s' % (last_or_shoe, times))
            temp = StandardScaler.transform(data)
            y = model.predict_proba(temp)
            k = numpy.concatenate([k,y[:,1].reshape(-1,1)],axis=1)

        data = numpy.delete(k,0,1)
        train_data = data[:int(data.shape[0] * 0.9),:].astype(numpy.float32)
        train_label = label[:int(data.shape[0] * 0.9)]
        test_data = data[int(data.shape[0] * 0.9):,:].astype(numpy.float32)
        test_label = label[int(data.shape[0] * 0.9):]
        OneHotEncoder = sklearn.preprocessing.OneHotEncoder()
        OneHotEncoder.fit(train_label.as_matrix().reshape(-1,1))
        train_label = OneHotEncoder.transform(train_label.as_matrix().reshape(-1,1)).toarray().astype(numpy.float32)
        test_label = OneHotEncoder.transform(test_label.as_matrix().reshape(-1,1)).toarray().astype(numpy.float32)

    return train_data, train_label, test_data, test_label

def size_stack_mlp_eager(last_or_shoe, man_or_woman):
    if man_or_woman=='man':
        sex=1
        n=3
    elif man_or_woman=='woman':
        sex=2
        n=4
    all_data = pandas.read_pickle('../data/size_data')
    all_data['sku'] = all_data['sku'].apply(lambda x:x[-3])
    gb_sex = all_data.groupby(by='sex')
    for one_sex in gb_sex:
        if one_sex[0] != sex:
            continue
        positive = one_sex[1][one_sex[1]['pick'] == 1]
        negative = one_sex[1][one_sex[1]['pick'] == 0]
        negative = negative.sample(n=positive.shape[0])
        data = positive.append(negative)

        phone=pandas.read_pickle('intermediate/train2_phone_%s_%s'%(last_or_shoe,man_or_woman))
        data=data[data['phone'].isin(phone)]
        data = data.sample(n=data.shape[0])
        label = data['pick'].astype(int)
        data = data.drop(columns=['sku', 'phone', 'styleno', 'pick', 'sex'])
        k = numpy.zeros([data.shape[0],1])
        for times in range(n):
            input_dimension=147
            fc1_dimension=147
            output_dimension=2
            StandardScaler = pandas.read_pickle('intermediate/StandardScaler_mlp_%s_%s_%s' % (last_or_shoe, times, man_or_woman))
            temp = StandardScaler.transform(data).astype(numpy.float32)

            w1=tensorflow.get_variable('w1_%s'%times, [input_dimension, fc1_dimension], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
            b1=tensorflow.get_variable('b1_%s'%times, [fc1_dimension], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
            w6=tensorflow.get_variable('w6_%s'%times, [fc1_dimension, output_dimension], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
            b6=tensorflow.get_variable('b6_%s'%times, [output_dimension], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
            def model(x):
                h1 = tensorflow.nn.relu(tensorflow.matmul(x, w1) + b1)
                h6 = tensorflow.matmul(h1, w6) + b6
                return h6
            Saver = tensorflow.contrib.eager.Saver([w1, w6, b1, b6])
            Saver.restore('model/mlp_%s_%s_%s'%(last_or_shoe, times, man_or_woman))
            y = model(temp)
            y=tensorflow.nn.softmax(y).numpy()
            k = numpy.concatenate([k,y[:,1].reshape(-1,1)],axis=1)

        for times in range(n):
            model = pandas.read_pickle('model/lgbm_%s_%s_%s' % (last_or_shoe, times, man_or_woman))
            StandardScaler = pandas.read_pickle('intermediate/StandardScaler_lgbm_%s_%s_%s' % (last_or_shoe, times, man_or_woman))
            temp = StandardScaler.transform(data)
            y = model.predict_proba(temp)
            k = numpy.concatenate([k,y[:,1].reshape(-1,1)],axis=1)

        data = numpy.delete(k,0,1)
        train_data = data[:int(data.shape[0] * 0.9),:].astype(numpy.float32)
        train_label = label[:int(data.shape[0] * 0.9)]
        test_data = data[int(data.shape[0] * 0.9):,:].astype(numpy.float32)
        test_label = label[int(data.shape[0] * 0.9):]
        OneHotEncoder = sklearn.preprocessing.OneHotEncoder()
        OneHotEncoder.fit(train_label.as_matrix().reshape(-1,1))
        train_label = OneHotEncoder.transform(train_label.as_matrix().reshape(-1,1)).toarray().astype(numpy.float32)
        test_label = OneHotEncoder.transform(test_label.as_matrix().reshape(-1,1)).toarray().astype(numpy.float32)

    return train_data, train_label, test_data, test_label

def size_askl(last_or_shoe):
    train_data, train_label, test_data, test_label = size_common(last_or_shoe)

    StandardScaler = sklearn.preprocessing.StandardScaler()
    StandardScaler.fit(train_data)
    pandas.to_pickle(StandardScaler, 'intermediate/StandardScaler_askl_%s' % last_or_shoe)
    train_data = StandardScaler.transform(train_data).reshape(-1,147).astype(numpy.float32)
    test_data = StandardScaler.transform(test_data).reshape(-1,147).astype(numpy.float32)
    train_label = train_label.astype(int)
    test_label = test_label.astype(int)

    return train_data, train_label, test_data, test_label

def size_gbm(last_or_shoe):
    train_data, train_label, test_data, test_label = size_common(last_or_shoe)

    StandardScaler = sklearn.preprocessing.StandardScaler()
    StandardScaler.fit(train_data)
    pandas.to_pickle(StandardScaler, 'intermediate/StandardScaler_gbm_%s' % last_or_shoe)
    train_data = StandardScaler.transform(train_data).reshape(-1,147).astype(numpy.float32)
    test_data = StandardScaler.transform(test_data).reshape(-1,147).astype(numpy.float32)

    return train_data, train_label, test_data, test_label

def size_xgb(last_or_shoe):
    train_data, train_label, test_data, test_label = size_common(last_or_shoe)

    StandardScaler = sklearn.preprocessing.StandardScaler()
    StandardScaler.fit(train_data)
    pandas.to_pickle(StandardScaler, 'intermediate/StandardScaler_xgb_%s' % last_or_shoe)
    train_data = StandardScaler.transform(train_data).reshape(-1,147).astype(numpy.float32)
    test_data = StandardScaler.transform(test_data).reshape(-1,147).astype(numpy.float32)

    return train_data, train_label, test_data, test_label

def size_lgbm(last_or_shoe):
    train_data, train_label, test_data, test_label = size_common(last_or_shoe)

    StandardScaler = sklearn.preprocessing.StandardScaler()
    StandardScaler.fit(train_data)
    pandas.to_pickle(StandardScaler, 'intermediate/StandardScaler_lgbm_%s' % last_or_shoe)
    train_data = StandardScaler.transform(train_data).reshape(-1,147).astype(numpy.float32)
    test_data = StandardScaler.transform(test_data).reshape(-1,147).astype(numpy.float32)

    return train_data, train_label, test_data, test_label

def size_1shoe(man_or_woman, shoe):
    if man_or_woman=='man':
        sex=1
    elif man_or_woman=='woman':
        sex=2
    all_data = pandas.read_pickle('../data/size_data')
    all_data['sku'] = all_data['sku'].apply(lambda x:x[-2])
    all_data=all_data[all_data['sku']==shoe]
    gb_sex = all_data.groupby(by='sex')
    for one_sex in gb_sex:
        if one_sex[0] != sex:
            continue
        positive = one_sex[1][one_sex[1]['pick'] == 1]
        negative = one_sex[1][one_sex[1]['pick'] == 0]
        negative = negative.sample(n=positive.shape[0])
        data = positive.append(negative)
        all_phone = data['phone'].drop_duplicates().tolist()
        test_phone = random.sample(all_phone, data['phone'].drop_duplicates().shape[0] // 10)
        train_data = pandas.DataFrame(columns=data.columns)
        test_data = pandas.DataFrame(columns=data.columns)
        test_data = test_data.append(data[data['phone'].isin(test_phone)])
        data = data[~data['phone'].isin(test_phone)]
        train_data = data
        train_data = train_data.sample(n=train_data.shape[0])
        test_data = test_data.sample(n=test_data.shape[0])
        train_label = train_data['pick'].astype(int)
        train_data = train_data.drop(columns=['sku', 'phone', 'styleno', 'pick', 'sex'])
        test_label = test_data['pick'].astype(int)
        test_data = test_data.drop(columns=['sku', 'phone', 'styleno', 'pick', 'sex'])

        pandas.to_pickle(train_data.columns.values.tolist(), '../train_size/intermediate/size_dimension_name')

    StandardScaler = sklearn.preprocessing.StandardScaler()
    StandardScaler.fit(train_data)
    pandas.to_pickle(StandardScaler, 'intermediate/StandardScaler_lgbm_%s_%s' % (man_or_woman,shoe))
    train_data = StandardScaler.transform(train_data).reshape(-1,147).astype(numpy.float32)
    test_data = StandardScaler.transform(test_data).reshape(-1,147).astype(numpy.float32)

    return train_data, train_label, test_data, test_label

def size_lgbm_ditermine(last_or_shoe, man_or_woman, train_phone, validate_phone, unselect, times):
    train_data, train_label, test_data, test_label = size_common_ditermine(last_or_shoe, man_or_woman, train_phone, validate_phone, unselect)

    StandardScaler = sklearn.preprocessing.StandardScaler()
    StandardScaler.fit(train_data)
    pandas.to_pickle(StandardScaler, 'intermediate/StandardScaler_lgbm_%s_%s_%s' % (last_or_shoe, times, man_or_woman))
    train_data = StandardScaler.transform(train_data).reshape(-1,147).astype(numpy.float32)
    test_data = StandardScaler.transform(test_data).reshape(-1,147).astype(numpy.float32)

    return train_data, train_label, test_data, test_label

def size_stack_lgbm(last_or_shoe):
    all_data = pandas.read_pickle('../data/size_data')
    all_data['sku'] = all_data['sku'].apply(lambda x:x[-3]*3)
    gb_sex = all_data.groupby(by='sex')
    for one_sex in gb_sex:
        if one_sex[0] == 1:
            continue
        positive = one_sex[1][one_sex[1]['pick'] == 1]
        negative = one_sex[1][one_sex[1]['pick'] == 0]
        negative = negative.sample(n=positive.shape[0])
        data = positive.append(negative)

        phone=pandas.read_pickle('intermediate/train2_phone_%s'%last_or_shoe)
        data=data[data['phone'].isin(phone)]
        data = data.sample(n=data.shape[0])
        label = data['pick'].astype(int)
        data = data.drop(columns=['sku', 'phone', 'styleno', 'pick', 'sex'])
        k = numpy.zeros([data.shape[0],1])
        for times in range(4):
            flag = None
            parser = argparse.ArgumentParser()
            parser.add_argument('--device', default='/cpu:0')
            parser.add_argument('--summary_dir', default='summary/')
            parser.add_argument('--model_dir', default='model/')
            parser.add_argument('--input_dimension', default=147)
            parser.add_argument('--fc1_dimension', default=147)
            parser.add_argument('--output_dimension', default=2)
            flag, unparsed = parser.parse_known_args()

            StandardScaler = pandas.read_pickle('intermediate/StandardScaler_mlp_%s_%s' % (last_or_shoe, times))
            temp = StandardScaler.transform(data).astype(numpy.float32)
            with tensorflow.device(flag.device):
                w1 = tensorflow.contrib.eager.Variable(tensorflow.truncated_normal([flag.input_dimension, flag.fc1_dimension], stddev=0.1), name='w1')
                b1 = tensorflow.contrib.eager.Variable(tensorflow.constant(0.1, shape = flag.fc1_dimension), name='b1')
                w6 = tensorflow.contrib.eager.Variable(tensorflow.truncated_normal([flag.fc1_dimension, flag.output_dimension], stddev=0.1), name='w6')
                b6 = tensorflow.contrib.eager.Variable(tensorflow.constant(0.1, shape = flag.output_dimension), name='b6')

            def model(x):
                h1 = tensorflow.nn.relu(tensorflow.matmul(x, w1) + b1)
                h6 = tensorflow.matmul(h1, w6) + b6
                return h6

            Saver = tensorflow.contrib.eager.Saver([w1, w6, b1, b6])
            Saver.restore('model/mlp_%s_%s'%(last_or_shoe, times))
            y = model(temp)
            y=tensorflow.nn.softmax(y).numpy()
            k = numpy.concatenate([k,y[:,1].reshape(-1,1)],axis=1)

        for times in range(4):
            model = pandas.read_pickle('model/lgbm_%s_%s' % (last_or_shoe, times))
            StandardScaler = pandas.read_pickle('intermediate/StandardScaler_lgbm_%s_%s' % (last_or_shoe, times))
            temp = StandardScaler.transform(data)
            y = model.predict_proba(temp)
            k = numpy.concatenate([k,y[:,1].reshape(-1,1)],axis=1)

        data = numpy.delete(k,0,1)
        train_data = data[:int(data.shape[0] * 0.9),:]
        train_label = label[:int(data.shape[0] * 0.9)]
        test_data = data[int(data.shape[0] * 0.9):,:]
        test_label = label[int(data.shape[0] * 0.9):]

    return train_data, train_label, test_data, test_label

def suit_common(question_number, man_or_woman):
    if man_or_woman=='man':
        sex=1
    elif man_or_woman=='woman':
        sex=2

    # all_data=pandas.read_pickle('../data/data')
    # def select_question(text):
    #     return question_number in text.keys()
    # all_data['answer'] = all_data['answer'].apply(lambda x:eval(x))
    # all_data['contain_answer'] = all_data['answer'].apply(select_question)
    # all_data = all_data[all_data['contain_answer'] == True]
    # all_data['answer_left'] = all_data['answer'].apply(lambda x:x[question_number][0])
    # all_data['answer_right'] = all_data['answer'].apply(lambda x:x[question_number][2])
    # if question_number == 12:
    #     all_data=all_data[all_data['comments']!='1']
    #     all_data=all_data[all_data['answer_left']=='1']
    #     all_data=all_data[all_data['answer_right']=='1']
    # else:
    #     all_data=all_data[all_data['answer_left']=='3']
    #     all_data=all_data[all_data['answer_right']=='3']
    # all_data = all_data.drop(columns=['contain_answer', 'answer'])
    # def test1(x):
    #     pick=x[x['pick']==1]
    #     bsize=pick['basicsize'].values[0]
    #     unpick=x[x['pick']!=1]
    #     unpick.loc[unpick['basicsize']<bsize,['answer_left','answer_right']]='1'
    #     unpick.loc[unpick['basicsize']>bsize,['answer_left','answer_right']]='5'
    #     return pick.append(unpick)
    # all_data=all_data.groupby(by=['phone','styleno']).apply(test1)
    # all_data=pandas.DataFrame(all_data.values,columns=all_data.columns)
    # all_data['basicsize']=all_data['basicsize'].astype(int)
    # man=all_data[all_data['sex']==1]
    # woman=all_data[all_data['sex']==2]
    # def testman(x):
    #     pick=x[x['pick']==1]
    #     value=pick['basicsize'].values[0]
    #     unpick=x[x['pick']==0]
    #     if value==240:
    #         return pick.append(unpick[unpick['basicsize']==245])
    #     elif value==270:
    #         return pick.append(unpick[unpick['basicsize']==265])
    #     else:
    #         return pick.append(unpick[unpick['basicsize']==value+5]).append(unpick[unpick['basicsize']==value-5])
    # def testwoman(x):
    #     pick=x[x['pick']==1]
    #     value=pick['basicsize'].values[0]
    #     unpick=x[x['pick']==0]
    #     if value==215:
    #         return pick.append(unpick[unpick['basicsize']==220])
    #     elif value==250:
    #         return pick.append(unpick[unpick['basicsize']==245])
    #     else:
    #         return pick.append(unpick[unpick['basicsize']==value+5]).append(unpick[unpick['basicsize']==value-5])    
    # man=man.groupby(by=['phone','styleno']).apply(testman)
    # woman=woman.groupby(by=['phone','styleno']).apply(testwoman)
    # man=pandas.DataFrame(man.values,columns=man.columns)
    # woman=pandas.DataFrame(woman.values,columns=woman.columns)
    # all_data=man.append(woman)
    # all_data.to_pickle('../data/all_data_%s'%question_number)
    # return 0

    all_data=pandas.read_pickle('../data/all_data_%s'%question_number)
    left_foot = []
    right_foot = []
    for name in all_data.columns.values:
        if re.search('_left',name):
            left_foot.append(name)
        if re.search('_right',name):
            right_foot.append(name)
    left_data = all_data.drop(columns=right_foot)
    right_data = all_data.drop(columns=left_foot)
    pandas.to_pickle(left_data.drop(columns=['phone', 'styleno', 'answer_left', 'comments', 'sex', 'basicsize', 'pick', 'sku']).columns.values.tolist(), 'intermediate/suit_left')
    pandas.to_pickle(right_data.drop(columns=['phone', 'styleno', 'answer_right', 'comments', 'sex', 'basicsize', 'pick', 'sku']).columns.values.tolist(), 'intermediate/suit_right')
    left_foot_new = []
    right_foot_new = []
    for name in left_foot:
        left_foot_new.append(re.sub('_left','',name))
    for name in right_foot:
        right_foot_new.append(re.sub('_right','',name))
    left_data = left_data.rename(columns=dict(zip(left_foot,left_foot_new)))
    right_data = right_data.rename(columns=dict(zip(right_foot,right_foot_new)))
    all_data = left_data.append(right_data)
    gb_sex = all_data.groupby(by='sex')
    for one_sex in gb_sex:
        if one_sex[0] != sex:
            continue
        data = one_sex[1]
        data=data.sample(n=data.shape[0])
        test_data=data.iloc[:int(data.shape[0]*0.1),:]
        train_data=data.iloc[int(data.shape[0]*0.1):,:]
        train_label = train_data['answer']
        train_data = train_data.drop(columns=['phone', 'styleno', 'answer', 'comments', 'sex', 'basicsize', 'pick', 'sku'])
        test_label = test_data['answer']
        test_data = test_data.drop(columns=['phone', 'styleno', 'answer', 'comments', 'sex', 'basicsize', 'pick', 'sku'])

        return train_data, train_label, test_data, test_label

def suit_common_ditermine(question_number, test_phone, test_styleno):
    all_data = pandas.read_pickle('../data/suit_data')
    def select_question(text):
        return question_number in text.keys()
    all_data['answer'] = all_data['answer'].apply(lambda x:eval(x))
    all_data['contain_answer'] = all_data['answer'].apply(select_question)
    all_data = all_data[all_data['contain_answer'] == True]
    all_data['answer_left'] = all_data['answer'].apply(lambda x:x[question_number][0])
    all_data['answer_right'] = all_data['answer'].apply(lambda x:x[question_number][2])
    if question_number == 12:
        all_data.loc[all_data['comments'] == '1', 'answer_left'] = '3'
        all_data.loc[all_data['comments'] == '1', 'answer_right'] = '3'
        all_data['answer_left'] = all_data['answer_left'].replace('3','2')
        all_data['answer_right'] = all_data['answer_right'].replace('3','2')
    else:
        all_data['answer_left'] = all_data['answer_left'].replace('2','1')
        all_data['answer_left'] = all_data['answer_left'].replace('4','5')
        all_data['answer_right'] = all_data['answer_right'].replace('2','1')
        all_data['answer_right'] = all_data['answer_right'].replace('4','5')
    all_data = all_data.drop(columns=['contain_answer', 'answer'])
    left_foot = []
    right_foot = []
    for name in all_data.columns.values:
        if re.search('_left',name):
            left_foot.append(name)
        if re.search('_right',name):
            right_foot.append(name)
    left_data = all_data.drop(columns=right_foot)
    right_data = all_data.drop(columns=left_foot)
    pandas.to_pickle(left_data.drop(columns=['phone', 'styleno', 'answer_left', 'comments', 'sex']).columns.values, '../train_suit/intermediate/suit_left')
    pandas.to_pickle(right_data.drop(columns=['phone', 'styleno', 'answer_right', 'comments', 'sex']).columns.values, '../train_suit/intermediate/suit_right')
    left_foot_new = []
    right_foot_new = []
    for name in left_foot:
        left_foot_new.append(re.sub('_left','',name))
    for name in right_foot:
        right_foot_new.append(re.sub('_right','',name))
    left_data = left_data.rename(columns=dict(zip(left_foot,left_foot_new)))
    right_data = right_data.rename(columns=dict(zip(right_foot,right_foot_new)))
    all_data = left_data.append(right_data)
    gb_sex = all_data.groupby(by='sex')
    for one_sex in gb_sex:
        if one_sex[0] == 1:
            continue
        data = one_sex[1]
        all_phone = data['phone'].drop_duplicates().tolist()
        all_styleno = data['styleno'].drop_duplicates().tolist()
        train_data = pandas.DataFrame(columns=data.columns)
        test_data = pandas.DataFrame(columns=data.columns)
        test_data = test_data.append(data[data['styleno'].isin(test_styleno)])
        data = data[~data['styleno'].isin(test_styleno)]
        test_data = test_data.append(data[data['phone'].isin(test_phone)])
        data = data[~data['phone'].isin(test_phone)]
        test_data = test_data.drop_duplicates()
        train_data = data
        train_data = train_data.sample(n=train_data.shape[0])
        test_data = test_data.sample(n=test_data.shape[0])
        train_label = train_data['answer']
        train_data = train_data.drop(columns=['phone', 'styleno', 'answer', 'comments', 'sex'])
        test_label = test_data['answer']
        test_data = test_data.drop(columns=['phone', 'styleno', 'answer', 'comments', 'sex'])

    return train_data, train_label, test_data, test_label

def suit_all(question_number):
    all_data = pandas.read_pickle('../data/suit_data')
    def select_question(text):
        return question_number in text.keys()
    all_data['answer'] = all_data['answer'].apply(lambda x:eval(x))
    all_data['contain_answer'] = all_data['answer'].apply(select_question)
    all_data = all_data[all_data['contain_answer'] == True]
    all_data['answer_left'] = all_data['answer'].apply(lambda x:x[question_number][0])
    all_data['answer_right'] = all_data['answer'].apply(lambda x:x[question_number][2])
    if question_number == 12:
        all_data.loc[all_data['comments'] == '1', 'answer_left'] = '3'
        all_data.loc[all_data['comments'] == '1', 'answer_right'] = '3'
        all_data['answer_left'] = all_data['answer_left'].replace('3','2')
        all_data['answer_right'] = all_data['answer_right'].replace('3','2')
    else:
        all_data['answer_left'] = all_data['answer_left'].replace('2','1')
        all_data['answer_left'] = all_data['answer_left'].replace('4','5')
        all_data['answer_right'] = all_data['answer_right'].replace('2','1')
        all_data['answer_right'] = all_data['answer_right'].replace('4','5')
    all_data = all_data.drop(columns=['contain_answer', 'answer'])
    left_foot = []
    right_foot = []
    for name in all_data.columns.values:
        if re.search('_left',name):
            left_foot.append(name)
        if re.search('_right',name):
            right_foot.append(name)
    left_data = all_data.drop(columns=right_foot)
    right_data = all_data.drop(columns=left_foot)
    left_foot_new = []
    right_foot_new = []
    for name in left_foot:
        left_foot_new.append(re.sub('_left','',name))
    for name in right_foot:
        right_foot_new.append(re.sub('_right','',name))
    left_data = left_data.rename(columns=dict(zip(left_foot,left_foot_new)))
    right_data = right_data.rename(columns=dict(zip(right_foot,right_foot_new)))
    all_data = left_data.append(right_data)
    gb_sex = all_data.groupby(by='sex')
    for one_sex in gb_sex:
        if one_sex[0] == 1:
            continue
        data = one_sex[1]
        data = data.sample(n=data.shape[0])
        label = data['answer']
        data = data.drop(columns=['phone', 'styleno', 'answer', 'comments', 'sex'])

    return data, label

def suit_stack_randomsearch_lgbm(question_number):
    data, label = suit_all(question_number)
    k = numpy.zeros((data.shape[0],1))
    if question_number == 12:
        for times in range(4):
            model = pandas.read_pickle('model/lgbm_%s_%s' % (question_number, times))
            StandardScaler = pandas.read_pickle('intermediate/StandardScaler_lgbm_%s_%s' % (question_number, times))
            temp = StandardScaler.transform(data)
            y = model.predict_proba(temp)
            k = numpy.concatenate([k,y[:,1].reshape(-1,1)],axis=1)
    else:
        for times in range(4):
            model = pandas.read_pickle('model/lgbm_%s_%s' % (question_number, times))
            StandardScaler = pandas.read_pickle('intermediate/StandardScaler_lgbm_%s_%s' % (question_number, times))
            temp = StandardScaler.transform(data)
            y = model.predict_proba(temp)
            k = numpy.concatenate([k,y[:,0:2]],axis=1)
    data = numpy.delete(k,0,1)
    train_data = data[:int(data.shape[0] * 0.8),:]
    train_label = label[:int(data.shape[0] * 0.8)]
    test_data = data[int(data.shape[0] * 0.8):,:]
    test_label = label[int(data.shape[0] * 0.8):]

    return train_data, train_label, test_data, test_label

def suit_mlp(question_number):
    train_data, train_label, test_data, test_label = suit_common(question_number)

    OneHotEncoder = sklearn.preprocessing.OneHotEncoder()
    OneHotEncoder.fit(train_label.as_matrix().reshape(-1,1))
    train_label = OneHotEncoder.transform(train_label.as_matrix().reshape(-1,1)).toarray().astype(numpy.float32)
    test_label = OneHotEncoder.transform(test_label.as_matrix().reshape(-1,1)).toarray().astype(numpy.float32)
    StandardScaler = sklearn.preprocessing.StandardScaler()
    StandardScaler.fit(train_data)
    pandas.to_pickle(StandardScaler, 'intermediate/StandardScaler_mlp_%s' % question_number)
    train_data = StandardScaler.transform(train_data).reshape(-1,92).astype(numpy.float32)
    test_data = StandardScaler.transform(test_data).reshape(-1,92).astype(numpy.float32)

    return train_data, train_label, test_data, test_label

def suit_1d(question_number):
    train_data, train_label, test_data, test_label = suit_common(question_number)

    OneHotEncoder = sklearn.preprocessing.OneHotEncoder()
    OneHotEncoder.fit(train_label.as_matrix().reshape(-1,1))
    train_label = OneHotEncoder.transform(train_label.as_matrix().reshape(-1,1)).toarray().astype(numpy.float32)
    test_label = OneHotEncoder.transform(test_label.as_matrix().reshape(-1,1)).toarray().astype(numpy.float32)
    StandardScaler = sklearn.preprocessing.StandardScaler()
    StandardScaler.fit(train_data)
    train_data = StandardScaler.transform(train_data).reshape(-1,92,1).astype(numpy.float32)
    test_data = StandardScaler.transform(test_data).reshape(-1,92,1).astype(numpy.float32)

    return train_data, train_label, test_data, test_label

def suit_2d(question_number):
    train_data, train_label, test_data, test_label = suit_common(question_number)

    OneHotEncoder = sklearn.preprocessing.OneHotEncoder()
    OneHotEncoder.fit(train_label.as_matrix().reshape(-1,1))
    train_label = OneHotEncoder.transform(train_label.as_matrix().reshape(-1,1)).toarray().astype(numpy.float32)
    test_label = OneHotEncoder.transform(test_label.as_matrix().reshape(-1,1)).toarray().astype(numpy.float32)
    StandardScaler = sklearn.preprocessing.StandardScaler()
    StandardScaler.fit(train_data)
    train_data = StandardScaler.transform(train_data)
    train_data = numpy.concatenate((train_data,numpy.zeros((train_data.shape[0],8))), axis=1).reshape(-1,10,10,1).astype(numpy.float32)
    test_data = StandardScaler.transform(test_data)
    test_data = numpy.concatenate((test_data,numpy.zeros((test_data.shape[0],8))), axis=1).reshape(-1,10,10,1).astype(numpy.float32)

    return train_data, train_label, test_data, test_label

def suit_askl(question_number):
    train_data, train_label, test_data, test_label = suit_common(question_number)

    StandardScaler = sklearn.preprocessing.StandardScaler()
    StandardScaler.fit(train_data)
    pandas.to_pickle(StandardScaler, 'intermediate/StandardScaler_askl_%s' % question_number)
    train_data = StandardScaler.transform(train_data).reshape(-1,92).astype(numpy.float32)
    test_data = StandardScaler.transform(test_data).reshape(-1,92).astype(numpy.float32)
    train_label = train_label.astype(int)
    test_label = test_label.astype(int)

    return train_data, train_label, test_data, test_label

def suit_gbm(question_number):
    train_data, train_label, test_data, test_label = suit_common(question_number)

    StandardScaler = sklearn.preprocessing.StandardScaler()
    StandardScaler.fit(train_data)
    pandas.to_pickle(StandardScaler, 'intermediate/StandardScaler_gbm_%s' % question_number)
    train_data = StandardScaler.transform(train_data).reshape(-1,92).astype(numpy.float32)
    test_data = StandardScaler.transform(test_data).reshape(-1,92).astype(numpy.float32)

    return train_data, train_label, test_data, test_label

def suit_xgb(question_number):
    train_data, train_label, test_data, test_label = suit_common(question_number)

    StandardScaler = sklearn.preprocessing.StandardScaler()
    StandardScaler.fit(train_data)
    pandas.to_pickle(StandardScaler, 'intermediate/StandardScaler_xgb_%s' % question_number)
    train_data = StandardScaler.transform(train_data).reshape(-1,92).astype(numpy.float32)
    test_data = StandardScaler.transform(test_data).reshape(-1,92).astype(numpy.float32)

    return train_data, train_label, test_data, test_label

def suit_lgbm(question_number,man_or_woman):
    train_data, train_label, test_data, test_label = suit_common(question_number,man_or_woman)

    StandardScaler = sklearn.preprocessing.StandardScaler()
    StandardScaler.fit(train_data)
    pandas.to_pickle(StandardScaler, 'intermediate/StandardScaler_lgbm_%s_%s' % (question_number,man_or_woman))
    train_data = StandardScaler.transform(train_data).reshape(-1,92).astype(numpy.float32)
    test_data = StandardScaler.transform(test_data).reshape(-1,92).astype(numpy.float32)

    return train_data, train_label, test_data, test_label

def suit_1shoe(question_number,man_or_woman,shoe):
    if man_or_woman=='man':
        sex=1
    elif man_or_woman=='woman':
        sex=2

    all_data=pandas.read_pickle('../data/all_data_%s'%question_number)
    all_data['sku'] = all_data['sku'].apply(lambda x:x[-2])
    all_data=all_data[all_data['sku']==shoe]
    left_foot = []
    right_foot = []
    for name in all_data.columns.values:
        if re.search('_left',name):
            left_foot.append(name)
        if re.search('_right',name):
            right_foot.append(name)
    left_data = all_data.drop(columns=right_foot)
    right_data = all_data.drop(columns=left_foot)
    pandas.to_pickle(left_data.drop(columns=['phone', 'styleno', 'answer_left', 'comments', 'sex', 'basicsize', 'pick', 'sku']).columns.values.tolist(), 'intermediate/suit_left')
    pandas.to_pickle(right_data.drop(columns=['phone', 'styleno', 'answer_right', 'comments', 'sex', 'basicsize', 'pick', 'sku']).columns.values.tolist(), 'intermediate/suit_right')
    left_foot_new = []
    right_foot_new = []
    for name in left_foot:
        left_foot_new.append(re.sub('_left','',name))
    for name in right_foot:
        right_foot_new.append(re.sub('_right','',name))
    left_data = left_data.rename(columns=dict(zip(left_foot,left_foot_new)))
    right_data = right_data.rename(columns=dict(zip(right_foot,right_foot_new)))
    all_data = left_data.append(right_data)
    gb_sex = all_data.groupby(by='sex')
    for one_sex in gb_sex:
        if one_sex[0] != sex:
            continue
        data = one_sex[1]
        data=data.sample(n=data.shape[0])
        test_data=data.iloc[:int(data.shape[0]*0.1),:]
        train_data=data.iloc[int(data.shape[0]*0.1):,:]
        train_label = train_data['answer']
        train_data = train_data.drop(columns=['phone', 'styleno', 'answer', 'comments', 'sex', 'basicsize', 'pick', 'sku'])
        test_label = test_data['answer']
        test_data = test_data.drop(columns=['phone', 'styleno', 'answer', 'comments', 'sex', 'basicsize', 'pick', 'sku'])

    StandardScaler = sklearn.preprocessing.StandardScaler()
    StandardScaler.fit(train_data)
    pandas.to_pickle(StandardScaler, 'intermediate/StandardScaler_lgbm_%s_%s_%s' % (question_number,man_or_woman,shoe))
    train_data = StandardScaler.transform(train_data).reshape(-1,92).astype(numpy.float32)
    test_data = StandardScaler.transform(test_data).reshape(-1,92).astype(numpy.float32)

    return train_data, train_label, test_data, test_label   

def suit_lgbm_ditermine(question_number, phone, select, times):
    train_data, train_label, test_data, test_label = suit_common_ditermine(question_number, phone, select)

    StandardScaler = sklearn.preprocessing.StandardScaler()
    StandardScaler.fit(train_data)
    pandas.to_pickle(StandardScaler, 'intermediate/StandardScaler_lgbm_%s_%s' % (question_number, times))
    train_data = StandardScaler.transform(train_data).reshape(-1,92).astype(numpy.float32)
    test_data = StandardScaler.transform(test_data).reshape(-1,92).astype(numpy.float32)

    return train_data, train_label, test_data, test_label

def suit_all_lgbm(question_number):
    train_data, train_label = suit_all(question_number)

    StandardScaler = sklearn.preprocessing.StandardScaler()
    StandardScaler.fit(train_data)
    pandas.to_pickle(StandardScaler, 'intermediate/StandardScaler_lgbm_final_%s' % question_number)
    train_data = StandardScaler.transform(train_data).reshape(-1,92).astype(numpy.float32)

    return train_data, train_label

def regress_common(train_phone, test_phone):
    metric=['footshoelastcalculatelength', 'footbasicwidth', 'footmetatarsalgirth', 'foottarsalgirth', 'footaround']
    all_data=pandas.read_pickle('../data/size_data_with_size')
    gb_sex=all_data.groupby(by='sex')
    for one_sex in gb_sex:
        if one_sex[0]==1:
            continue
        target=one_sex[1][one_sex[1]['pick']==1]
        target=target[['phone','styleno']+metric]
        basic=one_sex[1][one_sex[1]['basicsize']==230]
        data=pandas.merge(basic, target, how='inner', on=['phone', 'styleno'])
        data=data.rename(columns={'footshoelastcalculatelength_x':'footshoelastcalculatelength',
            'footbasicwidth_x':'footbasicwidth',
            'footmetatarsalgirth_x':'footmetatarsalgirth',
            'foottarsalgirth_x':'foottarsalgirth',
            'footaround_x':'footaround'})
        data=data.sample(n=data.shape[0])
        train =data[data['phone'].isin(train_phone)]
        test=data[data['phone'].isin(test_phone)]
        train_y=train[['footshoelastcalculatelength_y',
            'footbasicwidth_y',
            'footmetatarsalgirth_y',
            'foottarsalgirth_y',
            'footaround_y']]
        test_y=test[['footshoelastcalculatelength_y',
            'footbasicwidth_y',
            'footmetatarsalgirth_y',
            'foottarsalgirth_y',
            'footaround_y']]
        train_x=train.drop(columns=['footshoelastcalculatelength_y',
            'footbasicwidth_y',
            'footmetatarsalgirth_y',
            'foottarsalgirth_y',
            'footaround_y'])
        test_x=test.drop(columns=['footshoelastcalculatelength_y',
            'footbasicwidth_y',
            'footmetatarsalgirth_y',
            'foottarsalgirth_y',
            'footaround_y'])
        pandas.to_pickle(train_x.columns.values.tolist(), 'intermediate/regress_dimension_name')
        train_x=train_x.drop(columns=['sex','phone','styleno','basicsize','sku','pick'])
        test_x=test_x.drop(columns=['sex','phone','styleno','basicsize','sku','pick'])   

    return train_x, train_y, test_x, test_y    

def regress_mlp(train_phone, test_phone):
    train_x, train_y, test_x, test_y = regress_common(train_phone, test_phone)

    StandardScaler = sklearn.preprocessing.StandardScaler()
    StandardScaler.fit(train_x)
    pandas.to_pickle(StandardScaler, 'intermediate/StandardScaler_regress_mlp')
    train_x = StandardScaler.transform(train_x).reshape(-1,147).astype(numpy.float32)
    test_x = StandardScaler.transform(test_x).reshape(-1,147).astype(numpy.float32)
    train_y=train_y.as_matrix().astype(numpy.float32)
    test_y=test_y.as_matrix().astype(numpy.float32)

    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    regress_mlp(train_phone, test_phone)