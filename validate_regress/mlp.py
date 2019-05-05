import pandas
import tensorflow
import numpy
import json
import sys
sys.path.append("..")
import preprocess_data

def validate():
    test_phone=pandas.read_pickle('../train_regress/intermediate/test_phone')
    metric=['footshoelastcalculatelength', 'footbasicwidth', 'footmetatarsalgirth', 'foottarsalgirth', 'footaround']
    all_data=pandas.read_pickle('../data/size_data_with_size')

    gb_sex=all_data.groupby(by='sex')
    for one_sex in gb_sex:
        if one_sex[0]==1:
            continue
        test=one_sex[1][one_sex[1]['phone'].isin(test_phone)]
        test=test.drop(columns=['sex','sku'])  
    label=test[['phone','styleno','basicsize','pick']]
    true_y=test[metric]
    test=test.drop(columns=['phone','styleno','basicsize','pick'])
    columns=test.columns.values.tolist()
    StandardScaler=pandas.read_pickle('../train_regress/intermediate/StandardScaler_regress_mlp')
    test = StandardScaler.transform(test).reshape(-1,147).astype(numpy.float32)
    k=numpy.concatenate([test,label.as_matrix(),true_y.as_matrix()], 1)
    test=pandas.DataFrame(k,columns=columns+['phone', 'styleno', 'basicsize', 'pick', 'footshoelastcalculatelength_true', 'footbasicwidth_true', 'footmetatarsalgirth_true', 'foottarsalgirth_true', 'footaround_true'])
    bg_phone_styleno=test.groupby(by=['phone', 'styleno'])
    with tensorflow.Session() as sess:
        Saver =tensorflow.train.import_meta_graph('../train_regress/model/mlp.meta')
        Saver.restore(sess, '../train_regress/model/mlp')
        x = tensorflow.get_default_graph().get_tensor_by_name("x:0")
        y_ = tensorflow.get_default_graph().get_tensor_by_name("y_:0")
        y = tensorflow.get_default_graph().get_tensor_by_name("y:0")
        loss=tensorflow.reduce_sum(tensorflow.abs(y_-y),1)+tensorflow.sqrt(tensorflow.reduce_sum(tensorflow.square(y_-y),1))
        a=tensorflow.get_variable('a',[8])
        b=tensorflow.get_variable('b',[8])
        acc=tensorflow.cast(tensorflow.equal(tensorflow.argmin(a),tensorflow.argmax(b)),tensorflow.float32)
        minloss=tensorflow.reduce_min(a)
        all_answer=[]
        all_minloss=[]
        all_info=[]
        k=numpy.zeros([1,14])
        t=0
        for one in bg_phone_styleno:
            one1=one[1].sort_values(by='basicsize')
            b230=one1[one1['basicsize']==230]
            predict=sess.run(y, feed_dict={x:b230.drop(columns=['phone', 'styleno', 'basicsize', 'pick', 'footshoelastcalculatelength_true', 'footbasicwidth_true', 'footmetatarsalgirth_true', 'foottarsalgirth_true', 'footaround_true'])})
            predict=numpy.tile(predict,(8,1))
            losses=sess.run(loss,feed_dict={y_:one1[['footshoelastcalculatelength_true', 'footbasicwidth_true', 'footmetatarsalgirth_true', 'foottarsalgirth_true', 'footaround_true']],y:predict})
            answer=sess.run(acc,feed_dict={a:losses,b:one1['pick']})
            min1=sess.run(minloss,feed_dict={a:losses})
            all_answer.append(answer)
            all_minloss.append(min1)
            kk=numpy.concatenate([predict,one1[['footshoelastcalculatelength_true', 'footbasicwidth_true', 'footmetatarsalgirth_true', 'foottarsalgirth_true', 'footaround_true']],one1[['phone', 'styleno', 'basicsize', 'pick']]],1)
            k=numpy.concatenate([k,kk])
            print(t)
            t=t+1
    k=numpy.delete(k,0,0)
    k=pandas.DataFrame(k,columns=['footshoelastcalculatelength_pre', 'footbasicwidth_pre', 'footmetatarsalgirth_pre', 'foottarsalgirth_pre', 'footaround_pre', 'footshoelastcalculatelength_true', 'footbasicwidth_true', 'footmetatarsalgirth_true', 'foottarsalgirth_true', 'footaround_true', 'phone', 'styleno', 'basicsize', 'pick'])
    k.to_csv('all_result.csv',index=False)
    answer=numpy.array(all_answer).reshape(-1,1)
    minloss=numpy.array(all_minloss).reshape(-1,1)
    result=numpy.concatenate([minloss,answer],1)
    result=pandas.DataFrame(result,columns=['min_loss','answer'])
    result.to_csv('result.csv',index=False)
    print(result)
    
if __name__ == '__main__':
    validate()