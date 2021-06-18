from Preprocessing import Preprocessing
from WindowGenerator import WindowGenerator
from Models import Models

# Data Preprocessing
prep = Preprocessing()
train_df, val_df, test_df = prep.preprocess()

# Window Generator
window = WindowGenerator(train_df=train_df, val_df=val_df, test_df=test_df)

print(window)

# Define Model
models = Models(window=window)
model = models.seq2seq()

# Train and test model
history = models.train(model=model, save_nm='seq2seq')
eval_val, eval_test = models.evaluation(model=model)

# prediction
normalize_stats = (prep.train_mean, prep.train_std)
yhat = models.predict(model=model, data=prep.data_prep, normalize_stats=normalize_stats)

print("pred: ", yhat[0])
print("Finished")
