install.packages(c("tm", "caret", "e1071", "wordcloud", "RColorBrewer"))
library(tm)
library(caret)
library(e1071)
library(wordcloud)
library(RColorBrewer)

data <- read.csv("spam_data.csv", stringsAsFactors = FALSE)
colnames(data) <- c("Label", "Message")
data$Label <- factor(data$Label, levels = c("ham", "spam"))

corpus <- VCorpus(VectorSource(data$Message))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, stopwords("en"))
corpus <- tm_map(corpus, stripWhitespace)
dtm <- DocumentTermMatrix(corpus)

set.seed(123)
train_index <- createDataPartition(data$Label, p = 0.8, list = FALSE)
train_dtm <- dtm[train_index, ]
test_dtm <- dtm[-train_index, ]
train_labels <- data$Label[train_index]
test_labels <- data$Label[-train_index]

train_matrix <- as.matrix(train_dtm)
test_matrix <- as.matrix(test_dtm)
nb_model <- naiveBayes(train_matrix, train_labels)
predictions <- predict(nb_model, test_matrix)

conf_matrix <- confusionMatrix(predictions, test_labels)
print(conf_matrix)

spam_messages <- subset(data, Label == "spam")
spam_corpus <- VCorpus(VectorSource(spam_messages$Message))
spam_corpus <- tm_map(spam_corpus, content_transformer(tolower))
spam_corpus <- tm_map(spam_corpus, removeNumbers)
spam_corpus <- tm_map(spam_corpus, removePunctuation)
spam_corpus <- tm_map(spam_corpus, removeWords, stopwords("en"))
spam_corpus <- tm_map(spam_corpus, stripWhitespace)

spam_dtm <- DocumentTermMatrix(spam_corpus)
freq <- colSums(as.matrix(spam_dtm))
wordcloud(names(freq), freq, max.words = 50, colors = brewer.pal(8, "Dark2"))

ham_messages <- subset(data, Label == "ham")
ham_corpus <- VCorpus(VectorSource(ham_messages$Message))
ham_corpus <- tm_map(ham_corpus, content_transformer(tolower))
ham_corpus <- tm_map(ham_corpus, removeNumbers)
ham_corpus <- tm_map(ham_corpus, removePunctuation)
ham_corpus <- tm_map(ham_corpus, removeWords, stopwords("en"))
ham_corpus <- tm_map(ham_corpus, stripWhitespace)

ham_dtm <- DocumentTermMatrix(ham_corpus)
freq_ham <- colSums(as.matrix(ham_dtm))
wordcloud(names(freq_ham), freq_ham, max.words = 50, colors = brewer.pal(8, "Blues"))
