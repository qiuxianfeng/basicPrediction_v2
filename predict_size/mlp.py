import pandas
import tensorflow
import numpy
import json

import time

def predict(data, last_or_shoe):
    StandardScaler=pandas.read_pickle('../train_size/intermediate/StandardScaler_mlp_%s_0'%last_or_shoe)
    name=pandas.read_pickle('../train_size/intermediate/size_dimension_name')
    predict_data=data[name]
    predict_data = StandardScaler.transform(predict_data.values.reshape(1,-1)).astype(numpy.float32)

    with tensorflow.Session() as sess:
        Saver =tensorflow.train.import_meta_graph('../train_size/model/mlp_%s_0.meta'%last_or_shoe)
        Saver.restore(sess, '../train_size/model/mlp_%s_0'%last_or_shoe)
        x = tensorflow.get_default_graph().get_tensor_by_name("x:0")
        keep_prob = tensorflow.get_default_graph().get_tensor_by_name("keep_prob:0")
        y = tensorflow.get_default_graph().get_tensor_by_name("y:0")
        probability=sess.run(y, feed_dict={x:predict_data, keep_prob:1.0})

    result=json.dumps({'size':float(probability[0][1])})
    return result

if __name__ == '__main__':
    size_data=pandas.read_pickle('../data/size_data')
    data=size_data.iloc[0]
    predict(data, 'shoe')