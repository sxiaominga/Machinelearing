import pandas as pd
import warnings
from matplotlib import pyplot as plt
from matplotlib import pyplot
warnings.filterwarnings('ignore')
file_name = 'scores_u.xlsx'
df = pd.read_excel(file_name,header=1)
# (3)预测2019年各批次文科与理科的分数线
# 1.划分训练集和测试集
from statsmodels.tsa.holtwinters import (ExponentialSmoothing,
                                         SimpleExpSmoothing,
                                         Holt)
df.columns = ['年份','文科' ,'理科' ,'文科.1' ,'理科.1' ]
df.set_index('年份', inplace=True)
order=list(range(2006,2019,1)) #指定输出的 index顺序
df=df.loc[order]


train = df[0:9]
test = df[9:13]

test_length = len(test)
# 2.绘制图像
pyplot.plot(df)
plt.show()
# 3.拟合简单指数平滑（SES）模型并为它们创建预测：
# 1）文科
ses = SimpleExpSmoothing(train['文科']).fit(smoothing_level=0.8)
ses_forecast = ses.predict(start=2014, end=2018)

df.plot(color='#F0F0FF',
          title='Simple Exponential Smoothing',
          label='Actual',
          legend=True)
ses_forecast.plot(color='#FF69B4', legend=True,
                    label='$alpha=0.8$')
ses.fittedvalues.plot(color='#FF69B4')
# 2）理科
ses1 = SimpleExpSmoothing(train['理科']).fit(smoothing_level=0.8)
ses_forecast1 = ses1.predict(start=2014, end=2018)
ses_forecast1.plot(color='#FFD700', legend=True,
                    label='$alpha=0.8$')
ses1.fittedvalues.plot(color='#FFD700')
# 3）文科.1
ses2 = SimpleExpSmoothing(train['文科.1']).fit(smoothing_level=0.8)
ses_forecast2 = ses2.predict(start=2014, end=2018)
ses_forecast2.plot(color='#DC143C', legend=True,
                    label='$alpha=0.8$')
ses2.fittedvalues.plot(color='#DC143C')
# 4）理科.1
ses3 = SimpleExpSmoothing(train['理科.1']).fit(smoothing_level=0.8)
ses_forecast3 = ses3.predict(start=2014, end=2018)
ses_forecast3.plot(color='#008000', legend=True,
                    label='$alpha=0.8$')
ses3.fittedvalues.plot(color='#008000')

plt.axis([2006,2022,400,600])
plt.tight_layout()
plt.show()