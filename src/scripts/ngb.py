# NGBoost
ngb = NGBoost(Base=default_tree_learner, Dist=Normal, Score=MLE(), natural_gradient=True,verbose=False)
ngboost = ngb.fit(np.asarray(X_tr.drop(['SalePrice'],1)), np.asarray(X_tr.SalePrice))
y_pred = ngb.predict(X_val.drop(['SalePrice'],1))
sqrt(mean_squared_error(X_val.SalePrice,y_pred)) # 0.003389


# see the probability distributions by visualising
Y_dists = ngb.pred_dist(X_val.drop(['SalePrice'],1))
y_range = np.linspace(min(X_val.SalePrice), max(X_val.SalePrice), 200)
dist_values = Y_dists.pdf(y_range).transpose()
# plot index 0 and 114
idx = 114
plt.plot(y_range,dist_values[idx])
plt.title(f"idx: {idx}")
plt.tight_layout()
plt.show()