import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn import feature_extraction, model_selection, naive_bayes, metrics


# Load data
data = pd.read_csv("D:/ML_projects/Spam_detection_using_Naivve_Bayes/spam.csv", encoding='latin-1')

# Count the number of spam and not spam under v1
count_Class = pd.value_counts(data['v1'])
# Output ham     4825 spam     747


# Plot the spam and not spam in v1 in bar graph
count_Class.plot(kind= 'bar')
plt.title('Spam & Not Spam Count')
plt.show()


# pick and count the most common words in not spam
count1 = Counter(" ".join(data[data['v1']=='ham']['v2']).split()).most_common(20)

# Load it in a pandas dataframe
df1 = pd.DataFrame.from_dict(count1)

# rename column name
df1 = df1.rename(columns={0:'Words in Not Spam', 1:"Count"})

# pick and count the most common words in spam
count2 = Counter(" ".join(data[data['v1']=='spam']['v2']).split()).most_common(20)

# Load it in a pandas dataframe
df2 = pd.DataFrame.from_dict(count1)

# rename column name
df2 = df2.rename(columns={0:'Words in Spam', 1:"Count"})

# Plot it as a bar graph the common words and their count
df1.plot.bar()
y_pos = np.arange(len(df1["Words in Not Spam"]))
plt.xticks(y_pos, df1["Words in Not Spam"])
plt.show()


# Plot it as a bar graph the common words and their count
df2.plot.bar(color='red')
y_pos = np.arange(len(df2["Words in Spam"]))
plt.xticks(y_pos, df2["Words in Spam"])
plt.show()


# Assigns a number to the word
f = feature_extraction.text.CountVectorizer(stop_words='english')
X = f.fit_transform(data['v2'])

# Transform the data into binary variable
data['v1'] = data['v1'].map({'spam':1,'ham':0})

# Split the data into train and test
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, data['v1'], test_size=0.33, random_state=42)
print('X_train shape:', np.shape(X_train))
print('X_test shape:', np.shape(X_test))


# Training models using different regularization parameter alpha and calculate accuracy, precision and recall for the same
list_alpha = np.arange(1/100000, 20, 0.11)
score_train = np.zeros(len(list_alpha))
score_test = np.zeros(len(list_alpha))
recall_test = np.zeros(len(list_alpha))
precision_test = np.zeros(len(list_alpha))
count = 0
for alpha in list_alpha:
    bayes = naive_bayes.MultinomialNB(alpha=alpha)
    bayes.fit(X_train, y_train)
    score_train[count] = bayes.score(X_train, y_train)
    score_test[count] = bayes.score(X_test, y_test)
    recall_test[count] = metrics.recall_score(y_test, bayes.predict(X_test))
    precision_test[count] = metrics.precision_score(y_test, bayes.predict(X_test))
    count = count + 1

# Concatenate the values
matrix = np.matrix(np.c_[list_alpha, score_train, score_test, recall_test, precision_test])

# Load data in pandas dataframe and rename columns
models = pd.DataFrame(data=matrix, columns=['alpha', 'Train Acc', 'Test Accu', 'Test Recall', 'Test Precision'])

# Load models with precision 1
pre_1 = models[models['Test Precision']==1]
print('Models with Precision 1:')
print(pre_1)
'''
Models with Precision 1:
        alpha  Train Acc  Test Accu  Test Recall  Test Precision
143  15.73001   0.979641   0.969549     0.777778             1.0
144  15.84001   0.979641   0.969549     0.777778             1.0
145  15.95001   0.979641   0.969549     0.777778             1.0
146  16.06001   0.979373   0.969549     0.777778             1.0
147  16.17001   0.979373   0.969549     0.777778             1.0
148  16.28001   0.979105   0.967917     0.765873             1.0
149  16.39001   0.979105   0.967917     0.765873             1.0
150  16.50001   0.978837   0.967917     0.765873             1.0
151  16.61001   0.978570   0.967917     0.765873             1.0
152  16.72001   0.978570   0.967917     0.765873             1.0
153  16.83001   0.978570   0.967917     0.765873             1.0
154  16.94001   0.978570   0.967917     0.765873             1.0
155  17.05001   0.978570   0.967917     0.765873             1.0
156  17.16001   0.978570   0.967917     0.765873             1.0
157  17.27001   0.978570   0.967917     0.765873             1.0
158  17.38001   0.978570   0.967917     0.765873             1.0
159  17.49001   0.978302   0.967917     0.765873             1.0
160  17.60001   0.978302   0.967374     0.761905             1.0
161  17.71001   0.978302   0.967374     0.761905             1.0
162  17.82001   0.978034   0.967374     0.761905             1.0
163  17.93001   0.978034   0.967374     0.761905             1.0
164  18.04001   0.978034   0.967374     0.761905             1.0
165  18.15001   0.977766   0.967374     0.761905             1.0
166  18.26001   0.977766   0.967374     0.761905             1.0
167  18.37001   0.977498   0.967374     0.761905             1.0
168  18.48001   0.977498   0.966830     0.757937             1.0
169  18.59001   0.977498   0.966830     0.757937             1.0
170  18.70001   0.977498   0.966830     0.757937             1.0
171  18.81001   0.977498   0.966830     0.757937             1.0
172  18.92001   0.977498   0.966830     0.757937             1.0
173  19.03001   0.977498   0.966830     0.757937             1.0
174  19.14001   0.977498   0.966830     0.757937             1.0
175  19.25001   0.976962   0.966830     0.757937             1.0
176  19.36001   0.976694   0.966830     0.757937             1.0
177  19.47001   0.976426   0.966286     0.753968             1.0
178  19.58001   0.976159   0.966286     0.753968             1.0
179  19.69001   0.975891   0.966286     0.753968             1.0
180  19.80001   0.975355   0.966286     0.753968             1.0
181  19.91001   0.975087   0.966286     0.753968             1.0
'''


# Show data in the form confusion matrix
m_confusion_test = metrics.confusion_matrix(y_test, bayes.predict(X_test))
r = pd.DataFrame(data = m_confusion_test, columns=['Predicted 0', 'Predicted 1'], index=['Actual 0', 'Actual 1'])
print('Confusion matrix:')
print(r)
'''
Confusion matrix:
          Predicted 0  Predicted 1
Actual 0         1587            0
Actual 1           62          190
'''










