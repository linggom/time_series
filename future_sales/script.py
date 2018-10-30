import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

sales = pd.read_csv('./data/sales_train.csv.gz')


def get_label(sales, lag=1):
    max_num = sales['date_block_num'].max()
    curr = max_num - lag + 1
    labels = sales[sales['date_block_num'] == curr]['item_cnt_day'].values
    return labels

def create_data_lag(sales, lag=1, test_data=None):    
    max_num = sales['date_block_num'].max()
    curr = max_num - lag + 1    
    print(max_num, curr)
    if test_data is None:
        last_sales =  sales[sales['date_block_num'] == curr]
        last_sales = last_sales.groupby(['shop_id','item_id']).agg({"item_price":'mean',"item_cnt_day":"sum"})
        last_sales['shop_id'] = [i[0] for i in last_sales.index]
        last_sales['item_id'] = [i[1] for i in last_sales.index]
        label = last_sales.item_cnt_day.values
        curr = curr - 1
    else:
        last_sales = test_data
        label = None
    for i in range(0, max_num - lag + 1):
        previous_sales = sales.copy()
        previous_sales = previous_sales[['date_block_num', 'shop_id', 'item_id', 'item_price','item_cnt_day']]
        previous_sales = previous_sales[previous_sales['date_block_num'] == curr]
        col_item_price = 'block_%s_lag_%s_item_price' % (curr, i)
        col_item_cnt_day = 'block_%s_lag_%s_item_cnt_day' % (curr, i)
        previous_sales[col_item_price] = previous_sales['item_price']
        previous_sales[col_item_cnt_day] = previous_sales['item_cnt_day']

        previous_sales= previous_sales.groupby(['shop_id','item_id']).agg({col_item_price:'mean', col_item_cnt_day:"sum"})
        previous_sales['shop_id'] = [i[0] for i in previous_sales.index]
        previous_sales['item_id'] = [i[1] for i in previous_sales.index]

        last_sales = pd.merge(last_sales, previous_sales, on=['shop_id', 'item_id'], how='left')
        curr = curr - 1
    cols = []
    for col in last_sales.columns:
        if 'tuple' in str(type(col)):
            cols.append(col[0]+"_"+col[1])
        else:
            cols.append(col)
    last_sales.columns = cols
    last_sale_clear_nan = last_sales.fillna(0)

    if 'item_price' in last_sale_clear_nan.columns:
        del last_sale_clear_nan['item_price']

    if 'item_cnt_day' in last_sale_clear_nan.columns:
        del last_sale_clear_nan['item_cnt_day']

    return last_sale_clear_nan, label

eval_feature, eval_label = create_data_lag(sales, lag=1)
test_feature,test_label = create_data_lag(sales, lag=2)
train_feature,train_label = create_data_lag(sales, lag=3)

def get_column_buy_count(sales):
    col_cnt = []
    for c in sales.columns:
        if 'item_cnt_day' in c:
            col_cnt.append(c)
    return col_cnt


def get_last_month(sales):
    cnt_columns = get_column_buy_count(sales)
    sales['fe_last_month_count_sum'] = sales[cnt_columns[0]]
    sales['fe_last_month_count_mean'] = sales[cnt_columns[0]]
    return sales

def get_last_3_month(sales):
    cnt_columns = get_column_buy_count(sales)
    sales['fe_last_3_months_count_sum'] = sales[cnt_columns[:3]].sum(axis=1)
    sales['fe_last_3_months_count_mean'] = sales[cnt_columns[:3]].mean(axis=1)
    return sales

def get_last_6_month(sales):
    cnt_columns = get_column_buy_count(sales)
    sales['fe_last_6_months_count_sum'] = sales[cnt_columns[:6]].sum(axis=1)
    sales['fe_last_6_months_count_mean'] = sales[cnt_columns[:6]].mean(axis=1)
    return sales

def get_last_12_month(sales):
    cnt_columns = get_column_buy_count(sales)
    sales['fe_last_12_months_count_sum'] = sales[cnt_columns[:12]].sum(axis=1)
    sales['fe_last_12_months_count_mean'] = sales[cnt_columns[:12]].mean(axis=1)
    return sales


def get_rest_month(sales):
    cnt_columns = get_column_buy_count(sales)
    sales['fe_rest_months_count_sum'] = sales[cnt_columns[12:]].sum(axis=1)
    sales['fe_rest_months_count_mean'] = sales[cnt_columns[12:]].mean(axis=1)
    return sales

def get_grouped_month_summary(sales):
    cnt_columns = get_column_buy_count(sales)
    for month in (1,3,6,9):
        
        sales['fe_group_%s_months_count_sum' % month] = sales[cnt_columns[:month]].sum(axis=1)
        sales['fe_group_%s_months_count_mean' % month] = sales[cnt_columns[:month]].mean(axis=1)
        print("get_grouped_month_summary : ", month)
        
    return sales

def get_the_3_6_9_12_24_month(sales):
    cnt_columns = get_column_buy_count(sales)
    for i in (3,6,9):        
        print("get_the_3_6_9_12_24_month ", i)
        sales['fe_%s_month' % i] = sales[cnt_columns[i]]
    return sales

def get_last_buy(sales):
    col_cnt = get_column_buy_count(sales)
    last_buy = []
        
    for items in sales[col_cnt].values:
        month = -1
        for idx,row in enumerate(items):
            if idx > 0 and row > 0:
                if month == -1:
                    month = idx
                    break
        last_buy.append(month)
    sales['fe_last_buy'] = last_buy
    return sales

def get_average_interval(sales):
    col_cnt = get_column_buy_count(sales)
    average_interval = []
    for items in sales[col_cnt].values:
        month_interval = []
        last_buy_month = 0
        for idx,row in enumerate(items):
            if idx > 0 and row > 0:
                month_interval.append(idx - last_buy_month)
                last_buy_month = idx
        average_interval.append(np.mean(month_interval))
    sales['fe_average_interval'] = average_interval
    return sales

def get_avg_buy_rate(sales):  
    col_cnt = get_column_buy_count(sales)
    sales['fe_avg_buy_rate'] = sales[col_cnt].mean(axis=1)
    return sales

def get_std_buy_rate(sales):
    col_cnt = get_column_buy_count(sales)
    sales['fe_std_buy_rate'] = sales[col_cnt].std(axis=1)
    return sales

def generate_feature(last_sale_clear_nan):
    last_sale_clear_nan = get_last_buy(last_sale_clear_nan)
    print("get_last_buy")
    
    # last_sale_clear_nan = get_average_interval(last_sale_clear_nan)
    print("get_average_interval")
    
    last_sale_clear_nan = get_avg_buy_rate(last_sale_clear_nan)
    print("get_avg_buy_rate")
    
    last_sale_clear_nan = get_std_buy_rate(last_sale_clear_nan)
    print("get_std_buy_rate")
    
    last_sale_clear_nan = get_grouped_month_summary(last_sale_clear_nan)
    print("get grouped month summary")
    last_sale_clear_nan = get_the_3_6_9_12_24_month(last_sale_clear_nan)
    
    
    print('filter the features')
    return last_sale_clear_nan

test_feature = generate_feature(test_feature)
eval_feature = generate_feature(eval_feature)

train_feature = generate_feature(train_feature)

selected_feature = []
for fe in train_feature.columns:
    if 'fe_' in fe:
        print(fe)
        selected_feature.append(fe)
print(selected_feature)
train_feature = train_feature[selected_feature]
eval_feature_filtered = eval_feature[selected_feature]
test_feature = test_feature[selected_feature]



import lightgbm as lgb
from lightgbm import plot_importance

lr = lgb.LGBMRegressor(num_leaves=31, learning_rate=0.3, n_estimators=1000, colsample_bytree=0.8,
                       min_child_weight=300,
                       max_depth=8,
                       subsample=0.8,
                       seed=42
                      )

lr.fit(train_feature, train_label)

print("train")
print(mean_absolute_error(train_label, lr.predict(train_feature)))
print(math.sqrt(mean_squared_error(train_label, lr.predict(train_feature))))


print("test")
print(mean_absolute_error(test_label, lr.predict(test_feature)))
print(math.sqrt(mean_squared_error(test_label, lr.predict(test_feature))))


print("eval")
print(mean_absolute_error(eval_label, lr.predict(eval_feature_filtered)))
print(math.sqrt(mean_squared_error(eval_label, lr.predict(eval_feature_filtered))))
eval_feature['predictions'] = lr.predict(eval_feature_filtered)


data_test = pd.read_csv('data/test.csv.gz')
sample_submission = pd.read_csv('data/sample_submission.csv.gz')

submission_feature, _= create_data_lag(sales, lag=0, test_data = data_test)
submission_feature_filtered = generate_feature(submission_feature)
submission_feature_filtered = submission_feature_filtered[selected_feature]


submission_feature['predictions'] = lr.predict(submission_feature_filtered)


# import ipdb;ipdb.set_trace()

submission = pd.merge(data_test, submission_feature, on=['shop_id', 'item_id'], how='left')
submission['item_cnt_month'] = submission['predictions']
submission['ID'] = submission['ID_x']
submission[['ID','item_cnt_month']].to_csv('data/submission_5.csv', index=False)
