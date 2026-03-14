# ==============================================================================
# PROJECT: Credit Default Risk Prediction (REAL DATASET)
# Models: Logistic Regression, LDA, Naive Bayes
# ==============================================================================

# ------------------------------------------------------------------------------
# PHASE 1: Setup
# ------------------------------------------------------------------------------
if(!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, caret, MASS, e1071, pROC, corrplot, car, MVN, rstatix)

set.seed(123)

# ------------------------------------------------------------------------------
# PHASE 2: Load Real Dataset
# ------------------------------------------------------------------------------
# Reading the Kaggle 'Give Me Some Credit' training file
raw_data <- read.csv("C:/Users/DELL/Downloads/cs-training.csv/cs-training.csv")

# Remove the first index column (Unnamed: 0)
raw_data <- raw_data[, -1] 

# ------------------------------------------------------------------------------
# PHASE 3: Exploratory Data Analysis (Optimized Visuals)
# ------------------------------------------------------------------------------

# 1. Reset Graphics Parameters (Clears margins)
dev.off() # Closes previous plot device to reset settings
par(mar = c(1, 1, 1, 1)) # Sets small, equal margins (bottom, left, top, right)

# 2. Select numeric data
num_cols <- raw_data %>% dplyr::select(where(is.numeric))

# 3. Calculate Correlation
cor_mat <- cor(num_cols, use = "complete.obs")

# 4. Generate Properly Sized & Centered Heatmap
corrplot(cor_mat, 
         method = "color",           # Full color squares for better visibility
         type = "upper",            # Shows only the top half to reduce clutter
         order = "hclust",          # Groups similar variables together
         tl.col = "black",          # Text label color
         tl.srt = 45,               # Rotates top labels 45 degrees for space
         tl.cex = 0.8,              # ADJUST THIS: Scale text size (0.5 to 1.5)
         addCoef.col = "black",     # Adds the actual correlation numbers
         number.cex = 0.7,          # Size of the numbers inside squares
         cl.ratio = 0.2,            # Controls the width of the color bar
         mar = c(0, 0, 1, 0),       # Internal corrplot margin adjustment
         title = "Credit Risk Correlation Matrix")

# ------------------------------------------------------------------------------
# PHASE 4: Data Cleaning
# ------------------------------------------------------------------------------
# 1. Median Imputation for MonthlyIncome and NumberOfDependents
raw_data$MonthlyIncome[is.na(raw_data$MonthlyIncome)] <- 
  median(raw_data$MonthlyIncome, na.rm = TRUE)

raw_data$NumberOfDependents[is.na(raw_data$NumberOfDependents)] <- 
  median(raw_data$NumberOfDependents, na.rm = TRUE)

# 2. Outlier Capping (Winsorization) at 99th percentile
cap_vars <- c("RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "MonthlyIncome")
for(v in cap_vars) {
  upper <- quantile(raw_data[[v]], 0.99, na.rm = TRUE)
  raw_data[[v]] <- ifelse(raw_data[[v]] > upper, upper, raw_data[[v]])
}

# ------------------------------------------------------------------------------
# PHASE 5: Feature Engineering (Normalization)
# ------------------------------------------------------------------------------
target <- raw_data$SeriousDlqin2yrs
features <- raw_data %>% dplyr::select(-SeriousDlqin2yrs)

# Standardization (Parametric models require Mean=0, SD=1)
scaler <- preProcess(features, method = c("center", "scale"))
scaled_features <- predict(scaler, features)

final_df <- data.frame(
  SeriousDlqin2yrs = factor(target, levels = c(0, 1), labels = c("No", "Yes")),
  scaled_features
)

# ------------------------------------------------------------------------------
# PHASE 6: Parametric Assumption Diagnostics
# ------------------------------------------------------------------------------
# VIF check for Multicollinearity
vif_model <- glm(SeriousDlqin2yrs ~ ., data = final_df, family = binomial)
print(vif(vif_model))

# ------------------------------------------------------------------------------
# PHASE 7: Train/Test Split
# ------------------------------------------------------------------------------
trainIndex <- createDataPartition(final_df$SeriousDlqin2yrs, p = 0.7, list = FALSE)
train_set <- final_df[trainIndex, ]
test_set  <- final_df[-trainIndex, ]

# ------------------------------------------------------------------------------
# PHASE 8: Balancing and Model Training
# ------------------------------------------------------------------------------
# IMPORTANT: Downsampling solves the "0 Recall" issue by balancing classes
train_set_balanced <- downSample(x = train_set %>% dplyr::select(-SeriousDlqin2yrs),
                                 y = train_set$SeriousDlqin2yrs,
                                 yname = "SeriousDlqin2yrs")

# Training on balanced data
fit_logit <- glm(SeriousDlqin2yrs ~ ., data = train_set_balanced, family = binomial)
fit_lda   <- lda(SeriousDlqin2yrs ~ ., data = train_set_balanced)
fit_nb    <- naiveBayes(SeriousDlqin2yrs ~ ., data = train_set_balanced)

# ------------------------------------------------------------------------------
# PHASE 9: Model Evaluation (Accuracy, AUC, Precision, Recall, F1)
# ------------------------------------------------------------------------------
p_logit <- predict(fit_logit, test_set, type = "response")
p_lda   <- predict(fit_lda, test_set)$posterior[, "Yes"]
p_nb    <- predict(fit_nb, test_set, type = "raw")[, "Yes"]

get_metrics <- function(probs, actual, label) {
  pred_class <- factor(ifelse(probs > 0.5, "Yes", "No"), levels = c("No", "Yes"))
  cm <- confusionMatrix(pred_class, actual, positive = "Yes", mode = "everything")
  roc_obj <- roc(actual, probs, quiet = TRUE)
  
  return(c(Model = label,
           Accuracy = cm$overall["Accuracy"],
           ROC_AUC = auc(roc_obj),
           Precision = cm$byClass["Precision"],
           Recall = cm$byClass["Recall"],
           F1_Score = cm$byClass["F1"]))
}

# Generate Results Table
comparison_df <- rbind(
  get_metrics(p_logit, test_set$SeriousDlqin2yrs, "Logistic Regression"),
  get_metrics(p_lda, test_set$SeriousDlqin2yrs, "LDA"),
  get_metrics(p_nb, test_set$SeriousDlqin2yrs, "Naive Bayes")
) %>% as.data.frame()

print(comparison_df)

# ------------------------------------------------------------------------------
# PHASE 10: ROC Plot
# ------------------------------------------------------------------------------
plot(roc(test_set$SeriousDlqin2yrs, p_logit), col="blue", lwd=2, main="Real Data: ROC Comparison")
plot(roc(test_set$SeriousDlqin2yrs, p_lda), col="red", lwd=2, add=TRUE)
plot(roc(test_set$SeriousDlqin2yrs, p_nb), col="green", lwd=2, add=TRUE)
legend("bottomright", legend=c("Logit", "LDA", "Naive Bayes"), col=c("blue", "red", "green"), lwd=2)
