# LightGBM
ltr = lgb.Dataset(X_tr,label=X_tr.SalePrice)
param = {
'bagging_freq': 5,
'bagging_fraction': 0.6,
'bagging_seed': 123,
'boost_from_average':'false',
'boost': 'gbdt',
'feature_fraction': 0.3,
'learning_rate': .01,
'max_depth': 3,
'metric':'rmse',
'min_data_in_leaf': 128,
'min_sum_hessian_in_leaf': 8,
'num_leaves': 128,
'num_threads': 8,
'tree_learner': 'serial',
'objective': 'regression',
'verbosity': -1,
'random_state':123,
'max_bin': 8,
'early_stopping_round':100
}
lgbm = lgb.train(param,ltr,num_boost_round=10000,valid_sets=[(ltr)],verbose_eval=1000)
y_pred = lgbm.predict(X_val.drop(['SalePrice'],1))
sqrt(mean_squared_error(X_val.SalePrice,y_pred)) # 0.00494