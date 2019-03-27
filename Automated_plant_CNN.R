library(EBImage)
library(keras)

setwd("G:/PYTHON_AI_GOOGLE/PROJECT_PLANT")

pic1 <- c('A_1.jpg','A_2.jpg','A_3.jpg','A_4.jpg','A_5.jpg',
          'B_1.jpg','B_2.jpg','B_3.jpg','B_4.jpg','B_5.jpg',
          'C_1.jpg','C_2.jpg','C_3.jpeg','C_4.jpg','C_5.jpg')

train <- list()

for (i in 1:15) {train[[i]] <- readImage(pic1[i])}


pic2 <- c('A_6.jpg','B_6.jpg','C_6.jpg','D_6.jpg','D_7.jpg','D_8.jpg')
test <- list()

for (i in 1:6) {test[[i]] <- readImage(pic2[i])}
print(train[[12]])
summary(train[[12]])
plot(train[[6]])

par(mfrow= c(3,5))
for (i in 1:15) {plot(train[[i]])}
par(mfrow= c(1,1))


str(train)
for (i in 1:15){train[[i]] <- resize(train[[i]],100,100)}
for (i in 1:6){test[[i]] <- resize(test[[i]],100,100)}

train <- combine(train)
x <- tile(train,5)
plot(x)
display(x ,title='pictures')

test_disp <- test
test <- combine(test)
y <- tile(test,6)
plot(y)

train <- aperm(train, c(4,1,2,3))
test <- aperm(test, c(4,1,2,3))

trainy <- c(0,0,0,0,0,1,1,1,1,1,2,2,2,2,2)
testy <- c(0,1,2,0,1,2)


trainlabels <- to_categorical(trainy)
testlabels <- to_categorical(testy)

model <- keras_model_sequential()

model %>% layer_conv_2d(filters = 32, kernel_size = c(3,3),activation = 'relu',input_shape = c(100,100,3)) %>%
          layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu") %>%
          layer_max_pooling_2d(pool_size = c(2,2)) %>%
          layer_dropout(rate = 0.25) %>%
          layer_conv_2d(filters = 64, kernel_size = c(3,3),activation = 'relu') %>%
          layer_conv_2d(filters = 64, kernel_size = c(3,3),activation = 'relu') %>% 
          layer_max_pooling_2d(pool_size = c(2,2)) %>%
          layer_dropout(rate = 0.25) %>%
          layer_flatten() %>%
          layer_dense(units= 256, activation = 'relu') %>%
          layer_dropout(rate=0.25) %>%
          layer_dense(units= 500, activation = 'relu') %>%
          layer_dropout(rate=0.25) %>%
          layer_dense(units = 3, activation = 'softmax') 


          
model %>%  compile(loss='categorical_crossentropy',
                  optimizer = optimizer_sgd(lr= 0.01,
                                            decay = 1e-6,
                                            momentum = 0.9,
                                            nesterov = T), 
                                            metrics =c('accuracy'))

summary(model)

#Data agumentation of images
gen_images <- image_data_generator(featurewise_center = TRUE, featurewise_std_normalization = TRUE, rotation_range = 20, width_shift_range = 0.30, height_shift_range = 0.30, horizontal_flip = TRUE)

gen_images %>% fit_image_data_generator(train)

fit_generator(model,flow_images_from_data(train,trainlabels,gen_images,batch_size = 10, save_to_dir = "G:/PYTHON_AI_GOOGLE/PROJECT_PLANT/data_agumentation"), steps_per_epoch = 10,epochs = 10)


history <- model %>% fit(train, trainlabels, epochs = 60, batch_size = 30, validation_split = 0.2)

model %>% evaluate(train, trainlabels)
pred <- model %>% predict_classes(train)         
table(predicted =pred, actual = trainy)


prob <- model %>% predict_proba(train)
cbind(prob, Predicted_class = pred, Actual = trainy)


model %>% evaluate(test, testlabels)
pred <- model %>% predict_classes(test)         
table(predicted =pred, actual = testy)

#class probabilities
prob <- model %>% predict_proba(test)
x <- cbind(prob, Predicted_class = pred, Actual = testy)

spe_nam <- list()

for (i in pred) {
  if(i == 0) spe_nam <- c(spe_nam, "Eucalyptus")
  if(i == 1) spe_nam <- c(spe_nam, "Hazel")
  if(i == 2) spe_nam <- c(spe_nam, "Maple")
  
}
spe_nam

display(test_disp[[1]], title= spe_nam[1])

plot(test[[6]])
drawPoly(sp=FALSE,col='red',lwd=4)
text(19.74342, 78.13158,"Hazel 0.9987",col="blue", cex=0.7)
