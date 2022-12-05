#install.packages("ggplot2")
#install.packages("psych")
#install.packages("reshape2")

#library(ggplot2)
#library(psych)
#library(reshape2)

# Set working directory
fpath <- "~/LTTS/"
setwd(fpath)

ann <- c('reg','ann','relu', 'sig','lstm_seq', 'lstm', 'gmdhg', 'kgate', 'rbfg', 'cnnc')
n <- length(ann)

model <- c('nasdaq0704_', 'dj0704_', 'nikkei0704_', 'dax0704_')
m <- length(model)

data <- c('nasdaq_1_3_05-1_28_22.csv', 'dj_1_3_05-1_28_22.csv', 'nikkei_1_4_05_1_31_22.csv', 'dax_1_3_05_1_31_22.csv')
l <- length(data)

i <- 1
j <- 1
k <- 2

for (i in 1:n){
  for (j in 1:m){
    
    in_name1 <- paste('mape.1.', ann[i], '.', model[j], '.', data[j], '.txt', sep='')
    in_test1 <- read.delim(in_name1, sep = "", header = F, na.strings = " ", fill = T)

    for (k in 1:l){
      if(j != k){
        in_name2 <- paste('mape.1.', ann[i], '.', model[j], '.', data[k], '.txt', sep='')
        in_test2 <- read.delim(in_name2, sep = "", header = F, na.strings = " ", fill = T)

        x <- unlist(in_test1)
        y <- unlist(in_test2)
        
        testg <- wilcox.test(x[1:34], y[1:34], paired = TRUE, alternative = "greater", mu=-0.02)
        testl <- wilcox.test(x[1:34], y[1:34], paired = TRUE, alternative = "less", mu=-0.02)
        test <- wilcox.test(x[1:34], y[1:34], paired = TRUE, mu=-0.02)
        
        st <- sprintf( fmt="%s %s %s %f/%f/%f", ann[i], model[j], data[k], testg$p.value, testl$p.value, test$p.value)
        print(st)
        flush.console()
      }
    }
    
  }
}