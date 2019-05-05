import pandas
import numpy

result=pandas.read_csv('result.csv')
all_result=pandas.read_csv('all_result.csv')
pre=all_result[['footshoelastcalculatelength_pre', 'footbasicwidth_pre',
    'footmetatarsalgirth_pre', 'foottarsalgirth_pre', 'footaround_pre',]]
true=all_result[['footshoelastcalculatelength_true', 'footbasicwidth_true',
    'footmetatarsalgirth_true', 'foottarsalgirth_true', 'footaround_true',]]
k=pre.as_matrix()-true.as_matrix()
kk=all_result[['phone', 'styleno', 'basicsize', 'pick']].as_matrix()
k=numpy.concatenate([k,kk],1)
k=pandas.DataFrame(k,columns=['footshoelastcalculatelength_minus', 'footbasicwidth_minus', 'footmetatarsalgirth_minus', 'foottarsalgirth_minus', 'footaround_minus', 'phone', 'styleno', 'basicsize', 'pick'])

k['L1d2']=k[['footshoelastcalculatelength_minus', 'footbasicwidth_minus', 'footmetatarsalgirth_minus', 'foottarsalgirth_minus', 'footaround_minus']].abs().sum(1)/2
k['L2d2']=k[['footshoelastcalculatelength_minus', 'footbasicwidth_minus', 'footmetatarsalgirth_minus', 'foottarsalgirth_minus', 'footaround_minus']].applymap(numpy.square).sum(1).map(numpy.sqrt)/2
k['loss']=k['L1d2']+k['L2d2']
gb=k.groupby(by=['phone', 'styleno'])
k.to_csv('result1.csv',index=False)
m=[]
t=0
for one in gb:
    m1=one[1].loc[one[1]['loss'].idxmin()][['footshoelastcalculatelength_minus', 'footbasicwidth_minus', 'footmetatarsalgirth_minus', 'foottarsalgirth_minus', 'footaround_minus', 'loss']].as_matrix().tolist()
    m2=one[1][one[1]['pick']==1][['footshoelastcalculatelength_minus', 'footbasicwidth_minus', 'footmetatarsalgirth_minus', 'foottarsalgirth_minus', 'footaround_minus', 'loss']].as_matrix().tolist()
    m1.extend(m2[0])
    m.append(m1)
    print(t)
    t=t+1
m=pandas.DataFrame(m,columns=['footshoelastcalculatelength_min', 'footbasicwidth_min', 'footmetatarsalgirth_min', 'foottarsalgirth_min', 'footaround_min', 'loss_min', 'footshoelastcalculatelength_real', 'footbasicwidth_real', 'footmetatarsalgirth_real', 'foottarsalgirth_real', 'footaround_real', 'loss_real'])
m['label']=m['loss_min']==m['loss_real']
m['label']=m['label'].map({True:1,False:0})
m.to_csv('result2.csv',index=False)