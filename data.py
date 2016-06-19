import tushare as ts
import string
aa=ts.get_hist_data('600848')
#aa.to_csv('~/Desktop/test/cnn/data/600848.csv')
da=open("data/600848.csv")
ff=da.readlines()
RawData=[]
for sgl in ff:
    sgld=sgl.split(',')
    sglss=[]
    sgld[0]=string.replace(sgld[0],'-','')
    for sgls in sgld:
        try:
            sglss.append(float(sgls))
        except:
            pass
    RawData.append(sglss)
RawData.remove([])
print RawData[0]
