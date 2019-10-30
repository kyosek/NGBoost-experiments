# XGBoost
params = {'max_depth': 4, 'eta': 0.01, 'objective':'reg:squarederror', 'eval_metric':['rmse'],'booster':'gbtree', 'verbosity':0,'sample_type':'weighted','max_delta_step':4, 'subsample':.5, 'min_child_weight':100,'early_stopping_round':50}
dtr, dte = xgb.DMatrix(X_tr.drop(['SalePrice'],1),label=X_tr.SalePrice), xgb.DMatrix(X_val.drop(['SalePrice'],1),label=X_val.SalePrice)
num_round = 5000
xgbst = xgb.train(params,dtr,num_round,verbose_eval=500)
y_pred = xgbst.predict(dte)
sqrt(mean_squared_error(X_val.SalePrice,y_pred)) # 0.00361