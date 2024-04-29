import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import init_notebook_mode
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics, svm, tree, neighbors
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LeakyReLU
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import plot_model
import pickle


init_notebook_mode(connected = True)

sns.set_style('darkgrid')
plt.style.use('dark_background')
pd.set_option('display.max_columns', None)
pd.options.plotting.backend = 'plotly'

df = pd.read_csv(r'D:\Vscode\googlesolution\portfolio\pythonautism\autism_screening.csv')

df.head()

# Rename mispelled column names
df = df.rename(columns = {'austim' : 'autism', 'jundice' : 'jaundice'})
# df=df.drop('coumtry_of_res', axis=1)
df.head()

# Checking for singular values in columns
df.nunique()

# Checking for outlier ages
print(f'Maximum age is data:', df['age'].max())
print(f'Minimum age is data:', df['age'].min())

df[df['age'] == df['age'].max()]

#removing this data as 383 is not possible
df.drop(index = 52, inplace = True)
df.reset_index(inplace = True)

# Checking for NaN values
pd.DataFrame(df.isnull().sum(), columns = ['Missing Values']).style.bar(color = '#0099C6')

df['age'] = df['age'].fillna(np.round(df['age'].mean(), 0))
pd.DataFrame(df.isnull().sum(), columns = ['Missing Values'])

# Check invalid values in each column
for col in df.select_dtypes('O').columns:
    print('==================================================================================')
    print(f'Column name: {col}\n')
    print(f'Unique values:\n{df[col].unique()}\n')
    
# Method to display 3 interactive plots together vertically
def three_subplots(plot1, plot2, plot3, title1, title2, title3, w, h) :
    figures = [plot1, plot2, plot3]
    fig = make_subplots(rows = len(figures), cols = 1, subplot_titles = (title1, title2, title3), vertical_spacing = 0.05)

    for i, figure in enumerate(figures) :
        for trace in range(len(figure['data'])) :
            fig.append_trace(figure['data'][trace], row = i + 1, col = 1)

    fig.update_layout(width = w, height = h)

    return fig

# Target Variable
data_df = df['Class/ASD']
# Select features for training
features = df[[ 'result',
               'A1_Score','A2_Score','A3_Score','A4_Score','A5_Score','A6_Score','A7_Score','A8_Score', 'A9_Score','A10_Score']]

# Make all the values of age and result a value between 0 and 1
scaler = MinMaxScaler()
# num = ['age', 'result']
features_transform = pd.DataFrame(data = features)
# features_transform[num] = scaler.fit_transform(features[num])

# display(features_transform.head())

# Convert categorical data into dummy variables
features = pd.get_dummies(features_transform)
features.head()

# Encode all classes data to numerical values
data_classes = data_df.apply(lambda x : 1 if x == 'YES' else 0)
data_classes.head()

# target=df['result']

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(features, data_classes, test_size = 0.2, random_state = 1)

# Fixes Tensorflow - ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type float)
x_train = np.asarray(x_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)
x_test = np.asarray(x_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)

print(f'Shape of x_train is : {x_train.shape}')
print(f'Shape of y_train is : {y_train.shape}\n')
print(f'Shape of x_test is : {x_test.shape}')
print(f'Shape of y_test is : {y_test.shape}')

def evaluate_model(model) :
    cm = confusion_matrix(y_test, y_pred)
    ax = sns.heatmap(cm, annot = True, cmap = 'Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    ax.set_xticklabels(['No Autism', 'Autism'])
    ax.set_yticklabels(['No Autism', 'Autism'], rotation = 90, va = 'center')
    plt.show()

    print(classification_report(y_test, y_pred, target_names = ['No Autism', 'Autism']))
    print('Classification Accuracy :','{0:.4}'.format(metrics.accuracy_score(y_test, y_pred)))
    cv_score = cross_val_score(model, features, data_classes, cv = 10)
    print('CV Score :', '{0:.4}'.format(cv_score.mean()))
    print('ROC AUC Score : ','{0:.2%}'.format(roc_auc_score(y_test, y_pred)))

dct = DecisionTreeClassifier()    # Create Decision Tree Classification Model 
dct.fit(x_train, y_train)         # Fit the training data 
y_pred = dct.predict(x_test)      # Make Predictions 

# Visualizes Decision Tree 
tree.plot_tree(dct.fit(x_train, y_train)) 

# Call the method to display the performance evaluation metrics of the Decision Tree Classification Model 
evaluate_model(dct)

with open('autism_model.pkl', 'wb') as f:
    pickle.dump(dct, f)