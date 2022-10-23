#### Tuning mtry
cl <- makePSOCKcluster(5)
registerDoParallel(cl)

#use repeated cross-validation to select the optimal mtry value with the caret package
control <- trainControl(method = "repeatedcv", 
                        number = 5, 
                        repeats = 3, 
                        search = "grid")
tunegrid <- expand.grid(.mtry = c(10:20))

#search for optimal mtry value
set.seed(1)
rf_gridsearch <- train(Y~., 
                       data=train, 
                       method="rf", 
                       metric="RMSE", 
                       tuneGrid=tunegrid, 
                       trControl=control)
print(rf_gridsearch)
plot(rf_gridsearch)

stopCluster(cl)

# mtry = 20 produced the lowest RMSE in the grid search. Though the result changes slightly if we do not set the seed, 
# it seems that the optimal mtry would either = 19 or 20, 
# and the training RMSE value for these two mtry values do not differ much.


#----------------------------------------------------------------------------------------------------
#### Tuning ntrees
# Since we already know the best mtry and min.node size values that generate the lowest MSE, for tuning the number of trees, 
# we can start from the best parameters, 
# even though the number of trees might interact with mtry and min.node.size.

# #optimal tree number
tunegrid <- expand.grid(.mtry = 54
                        # .splitrule = "variance",
                        # .min.node.size = 44
                        )
models <- list()
control <- trainControl(method = "cv",
                        number = 10,
                        # repeats = 1,
                        search = "grid")

for (ntree in c(500, 1500, 2500)) {
	set.seed(142)
	fit <- train(Y~.,
	             data=train,
	             method="rf",
	             metric="RMSE",
	             tuneGrid=tunegrid,
	             trControl=control,
	             ntree=ntree)
	tree <- toString(ntree)
	models[[tree]] <- fit
}

results <- resamples(models)
summary(results)
dotplot(results)

# ntree = 500 generated the lowest RMSE, though only slightly smaller than when ntree = 2500 and 1500.

# Although tuning ntrees and mtry & min.node.size (tree depth) is not the best way to tune parameters 
# and tuning parameters do interact with each other, tuning the number of trees separately nevertheless 
# saved time on the tuning process as a whole (as the length of trees needed to match the length of mtry 
# and min.node.size within the tuning data frame). Moreover, mtry and tree depth are more important parameters 
# than ntree when it comes to tuning random forest models.


#----------------------------------------------------------------------------------------------------
# ggplot(all.pred) + 
#   geom_point(aes(x = all.pred$random_forest, y = all.pred$Y)) + 
#   geom_point(aes(x = all.pred$lasso, y = all.pred$Y)) +
#   scale_fill_manual(values = c("red","blue"))