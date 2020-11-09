import streamlit as st
st.set_page_config(page_title="new_moon", page_icon=":smiley:")

import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from numpy import random
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.impute import KNNImputer

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)

def main():

        """Auro's ML shop"""
        st.title("Welcome to Auro's Machine learning shop - New Moon")

        st.write("Play with data and have fun :smiley:")
        st.write("This is a classification machine learning web-app; We are testing numerous tabular classification data to make it better and better :sunglasses:.")
        st.write("For testing purpose and bug fixing, this web app only supports tabular data in Excel Workbook (.xlsx) format. Later we will add other file extensions to upload. :bell:")
        st.write("This data set has been tested on the classic traditional ***Iris*** , ***Breast-cancer*** , ***Wine*** , ***Credit-card-fraud*** , ***Diabetes***, ***Titanic*** and other binary Injury classification dataset. :milky_way:")
        st.write("The final goal is to empower Data Science and Machine learning with automated data analysis and ML Model building. :blossom:")
        st.write("We will constantly update this web-app with more features e.g., better and complex visualizations, feature selection techniques, feature engineering etc. :new_moon:")
        st.write("We have imputed the missing values by ourselves with simple imputation techniques. But, later we will add advanced imputation techniques and users will have their own choice to fill :incoming_envelope: .")
        st.write("Credits: Streamlit, JCharisTech & J-Secur1ty for exceptional tutroials.")

        

        activities = ["About", "Dataset Basics", "Basic Data Visualization", "PCA", "t-SNE", "Model Building"]
        choice = st.sidebar.selectbox("Select Activity", activities)

        st.text("Upload and analyze any classification problem - Tabular data only")
        data = st.file_uploader("Upload dataset", type = [ "xlsx"])

        if data is not None:
                df = pd.read_excel(data)
                st.dataframe(df.head())

                if choice == "About":

                        st.write("Please visit to my website: https://kaii55.github.io  :wolf:")
                        st.write("My github profile: https://github.com/kaii55")
                        
                elif choice == "Dataset Basics":
                        st.subheader("Exploratory Data Analysis")

                        if st.checkbox("show shape"):
                                st.write(df.shape)
                        if st.checkbox("show columns"):
                                all_columns = df.columns.to_list()
                                st.write(all_columns)
                        if st.checkbox("show data types of the columns"):
                                st.write(df.dtypes)
                        if st.checkbox("show summary"):
                                st.write(df.describe())
                        if st.checkbox("show int and float columns"):
                                a = df.select_dtypes(include=[np.float64,np.int64])
                                st.write(a)
                        if st.checkbox("show category columns"):
                                a = df.select_dtypes(include=[object])
                                st.write(a) 
                        if st.checkbox("show missing values"):
                                st.write(df.isnull().sum())
                                     

                elif choice == "Basic Data Visualization":
                        st.subheader("Data Visualization") 

                        all_columns = df.columns.to_list()
                        selected_columns = st.multiselect("Select non-target columns", all_columns)
                        new_df = df[selected_columns]
                        st.dataframe(new_df) #same as st.write(new_df)

                        
                        selected_target_columns = st.multiselect("Select target column", all_columns)
                        target_df = df[selected_target_columns]
                        st.dataframe(target_df) #same as st.write(target_df)

                        labelencoder = LabelEncoder()
                        for col in target_df.columns:
                                target_df = pd.DataFrame(labelencoder.fit_transform(target_df[[col]]), columns = target_df.columns)        
                        
                        if st.checkbox("Target column visualization"):
                                for col in target_df.columns:
                                        st.write(target_df[col].value_counts())
                        
                                colors = ["#0101DF", "#DF0101"]
                                for col in target_df.columns:
                                        st.write(sns.countplot(col, data = target_df, palette = colors))
                                        st.pyplot()
                        

                        if st.checkbox("Co-relation among the non-target columns"):
                                st.text("Select multiple columns for creating a correlation heatmap")
                                fig, ax = plt.subplots(figsize=(40,40))
                                st.write(sns.heatmap(new_df.corr(), annot = True), linewidths= 0.1, annot_kws={"fontsize":12}, ax = ax)
                                st.pyplot()
                        
                        det = new_df.join(target_df)
                        if st.checkbox("Coorelation between the non-target and target columns"):
                                fig, ax = plt.subplots(figsize=(40,40))
                                for col in target_df.columns:
                                        st.write(det.corrwith(det[col]).plot.bar(figsize = (20, 10), grid = True))
                                        st.pyplot()

                
                elif choice == "Model Building":
                        st.subheader("Building ML model")

                        Imputation_techniques = ["No Imputation", "Mean", "Median"]
                        mv = st.sidebar.radio("Impute Missing Values", Imputation_techniques)

                        if mv == "No Imputation":

                                classifer = st.sidebar.selectbox("Select classifier", ("SVM", "KNN", "Log Reg", "RF", "DT"))

                                all_columns = df.columns.to_list()
                                selected_columns = st.multiselect("Select non-target columns", all_columns)
                                new_df = df[selected_columns]
                                st.dataframe(new_df) #same as st.write(new_df)

                                selected_target_columns = st.multiselect("Select target column - Select at-least one to build the model", all_columns)
                                target_df = df[selected_target_columns]
                                st.dataframe(target_df) #same as st.write(target_df)

                                labelencoder = LabelEncoder()
                                for col in target_df.columns:
                                        target_df = pd.DataFrame(labelencoder.fit_transform(target_df[[col]]), columns = target_df.columns)

                                X = new_df
                                y = target_df
                                

                                def add_parameter_ui(clf_name):

                                        params = dict()
                                        if clf_name == 'SVM':
                                                C = st.sidebar.slider('C', 0.01, 10.0)
                                                params['C'] = C
                                        elif clf_name == 'KNN':
                                                K = st.sidebar.slider('K', 1, 15)
                                                params['K'] = K
                                        elif clf_name == 'Log Reg':
                                                C = st.sidebar.slider('C', 1, 15)
                                                params['C'] = C
                                        elif clf_name == 'RF':
                                                max_depth = st.sidebar.slider('max_depth', 2, 15)
                                                params['max_depth'] = max_depth
                                                n_estimators = st.sidebar.slider('n_estimators', 1, 50)
                                                params['n_estimators'] = n_estimators
                                        else:
                                                max_depth = st.sidebar.slider('max_depth', 2, 15)
                                                params['max_depth'] = max_depth
                                                min_samples_split = st.sidebar.slider('min_samples_split', 2, 15)
                                                params['min_samples_split'] = min_samples_split

                                        return params
                                
                                params = add_parameter_ui(classifer)

                                def get_classifier(clf_name, params):

                                        clf = None

                                        if clf_name == 'SVM':
                                                clf = SVC(C=params['C'])
                                        elif clf_name == 'KNN':
                                                clf = KNeighborsClassifier(n_neighbors=params['K'])
                                        elif clf_name == 'Log Reg':
                                                clf = LogisticRegression(C=params['C'])
                                        elif clf_name == 'RF':
                                                clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
                                                max_depth=params['max_depth'], random_state=1234)
                                        else:
                                                clf = DecisionTreeClassifier(max_depth=params['max_depth'],
                                                min_samples_split=params['min_samples_split'], random_state=1234)
                                        return clf

                                clf = get_classifier(classifer, params)

                                if st.checkbox("Build ML model with important evaluation mertics"):

                                        scaling_techniques = ["No scaling", "Standard scaling", "Min-Max scaling", "Robust scaling"]
                                        sca_tech = st.sidebar.radio("Scaling the train data and transform the test data", scaling_techniques)

                                        if sca_tech == "No scaling":

                                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1234) 

                                                st.write('Shape of X_train:', X_train.shape)
                                                st.write('Shape of X_test:', X_test.shape) 

                                        if sca_tech == "Standard scaling":

                                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1234) 
                                                sc = StandardScaler()
                                                X_train = sc.fit_transform(X_train) 
                                                X_test = sc.transform(X_test) 

                                                st.write('Shape of X_train:', X_train.shape)
                                                st.write('Shape of X_test:', X_test.shape) 

                                        if sca_tech == "Min-Max scaling":

                                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1234) 
                                                mm= MinMaxScaler()
                                                X_train = mm.fit_transform(X_train) 
                                                X_test = mm.transform(X_test) 

                                                st.write('Shape of X_train:', X_train.shape)
                                                st.write('Shape of X_test:', X_test.shape) 

                                        if sca_tech == "Robust scaling":

                                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1234) 
                                                rb= RobustScaler()
                                                X_train = rb.fit_transform(X_train) 
                                                X_test = rb.transform(X_test) 

                                                st.write('Shape of X_train:', X_train.shape)
                                                st.write('Shape of X_test:', X_test.shape) 
                                        

                                        sampling = st.sidebar.selectbox("Select sampling technique", ("no sampling", "SMOTE", "RUS", "ROS"))

                                        if sampling == "no sampling":
                                                st.subheader("No Sampling")
                                                steps = [('m', clf)]
                                                pipe = Pipeline(steps=steps)

                                        elif sampling  == "SMOTE":
                                                st.subheader("Smote Oversampling technique")
                                                over = SMOTE(random_state = 1234)
                                                steps = [('os', over), ('m', clf)]
                                                pipe = Pipeline(steps=steps) 

                                        elif sampling == "RUS":
                                                st.subheader("Random Under sampling technique")
                                                rus = RandomUnderSampler(random_state = 1234)
                                                steps = [('u', rus), ('m', clf)]
                                                pipe = Pipeline(steps=steps) 

                                        elif sampling == "ROS":
                                                st.subheader("Random Over sampling technique")
                                                ros = RandomOverSampler(random_state = 1234)
                                                steps = [('o', ros), ('m', clf)]
                                                pipe = Pipeline(steps=steps) 

                                        pipe.fit(X_train, y_train)
                                        y_pred = pipe.predict(X_test)  

                                        acc = accuracy_score(y_test, y_pred)
                                        conf_mat = confusion_matrix(y_test, y_pred)
                                        #pre_score = precision_score(y_test, y_pred, average='weighted')
                                        #recall_score = recall_score(y_test, y_pred, average='weighted')
                                        f1 = f1_score(y_test, y_pred, average='weighted')
                                        #auc = roc_auc_score(y_test, y_pred)

                                        st.write(f'Classifier = {classifer}')
                                        st.write(f'Accuracy =', acc)
                                        st.write(f'Confusion Matrix =', conf_mat)
                                        #st.write(f'Precision_score =', pre_score)
                                        #st.write(f'Recall_score =', recall_score)
                                        st.write(f'f1_score =', f1)
                                        #st.write(f'auc_score =', auc)
                                
                        if mv == "Mean":
                                for col in df.columns:
                                        if df[col].dtypes == "int64":
                                                df[col] = df[col].replace(to_replace = np.nan, value = df[col].mean())
                                        elif df[col].dtypes == "float64":
                                                df[col] = df[col].replace(to_replace = np.nan, value = df[col].mean())
                                        elif df[col].dtypes == "object":
                                                df[col] = df[col].replace(to_replace = np.nan, value = random.choice(df[col].unique()))
                                        else:
                                                df[col] = df[col].dropna()

                                classifer = st.sidebar.selectbox("Select classifier", ("SVM", "KNN", "Log Reg", "RF", "DT"))

                                all_columns = df.columns.to_list()
                                selected_columns = st.multiselect("Select non-target columns", all_columns)
                                new_df = df[selected_columns]
                                st.dataframe(new_df) #same as st.write(new_df)

                                selected_target_columns = st.multiselect("Select target column - Select at-least one to build the model", all_columns)
                                target_df = df[selected_target_columns]
                                st.dataframe(target_df) #same as st.write(target_df)

                                labelencoder = LabelEncoder()
                                for col in target_df.columns:
                                        target_df = pd.DataFrame(labelencoder.fit_transform(target_df[[col]]), columns = target_df.columns)

                                X = new_df
                                y = target_df
                                

                                def add_parameter_ui(clf_name):

                                        params = dict()
                                        if clf_name == 'SVM':
                                                C = st.sidebar.slider('C', 0.01, 10.0)
                                                params['C'] = C
                                        elif clf_name == 'KNN':
                                                K = st.sidebar.slider('K', 1, 15)
                                                params['K'] = K
                                        elif clf_name == 'Log Reg':
                                                C = st.sidebar.slider('C', 1, 15)
                                                params['C'] = C
                                        elif clf_name == 'RF':
                                                max_depth = st.sidebar.slider('max_depth', 2, 15)
                                                params['max_depth'] = max_depth
                                                n_estimators = st.sidebar.slider('n_estimators', 1, 50)
                                                params['n_estimators'] = n_estimators
                                        else:
                                                max_depth = st.sidebar.slider('max_depth', 2, 15)
                                                params['max_depth'] = max_depth
                                                min_samples_split = st.sidebar.slider('min_samples_split', 2, 15)
                                                params['min_samples_split'] = min_samples_split

                                        return params
                                
                                params = add_parameter_ui(classifer)

                                def get_classifier(clf_name, params):

                                        clf = None

                                        if clf_name == 'SVM':
                                                clf = SVC(C=params['C'])
                                        elif clf_name == 'KNN':
                                                clf = KNeighborsClassifier(n_neighbors=params['K'])
                                        elif clf_name == 'Log Reg':
                                                clf = LogisticRegression(C=params['C'])
                                        elif clf_name == 'RF':
                                                clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
                                                max_depth=params['max_depth'], random_state=1234)
                                        else:
                                                clf = DecisionTreeClassifier(max_depth=params['max_depth'],
                                                min_samples_split=params['min_samples_split'], random_state=1234)
                                        return clf

                                clf = get_classifier(classifer, params)

                                if st.checkbox("Build ML model with important evaluation mertics"):

                                        scaling_techniques = ["No scaling", "Standard scaling", "Min-Max scaling", "Robust scaling"]
                                        sca_tech = st.sidebar.radio("Scaling the train data and transform the test data", scaling_techniques)

                                        if sca_tech == "No scaling":

                                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1234) 

                                                st.write('Shape of X_train:', X_train.shape)
                                                st.write('Shape of X_test:', X_test.shape) 

                                        if sca_tech == "Standard scaling":

                                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1234) 
                                                sc = StandardScaler()
                                                X_train = sc.fit_transform(X_train) 
                                                X_test = sc.transform(X_test) 

                                                st.write('Shape of X_train:', X_train.shape)
                                                st.write('Shape of X_test:', X_test.shape) 

                                        if sca_tech == "Min-Max scaling":

                                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1234) 
                                                mm= MinMaxScaler()
                                                X_train = mm.fit_transform(X_train) 
                                                X_test = mm.transform(X_test) 

                                                st.write('Shape of X_train:', X_train.shape)
                                                st.write('Shape of X_test:', X_test.shape) 

                                        if sca_tech == "Robust scaling":

                                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1234) 
                                                rb= RobustScaler()
                                                X_train = rb.fit_transform(X_train) 
                                                X_test = rb.transform(X_test) 

                                                st.write('Shape of X_train:', X_train.shape)
                                                st.write('Shape of X_test:', X_test.shape) 
                                        

                                        sampling = st.sidebar.selectbox("Select sampling technique", ("SMOTE", "RUS", "ROS", "no sampling"))

                                        if sampling  == "SMOTE":
                                                st.subheader("Smote Oversampling technique")
                                                over = SMOTE(random_state = 1234)
                                                steps = [('os', over), ('m', clf)]
                                                pipe = Pipeline(steps=steps) 
                                        elif sampling == "RUS":
                                                st.subheader("Random Under sampling technique")
                                                rus = RandomUnderSampler(random_state = 1234)
                                                steps = [('u', rus), ('m', clf)]
                                                pipe = Pipeline(steps=steps) 
                                        elif sampling == "ROS":
                                                st.subheader("Random Over sampling technique")
                                                ros = RandomOverSampler(random_state = 1234)
                                                steps = [('o', ros), ('m', clf)]
                                                pipe = Pipeline(steps=steps) 
                                        elif sampling == "no sampling":
                                                st.subheader("No Sampling")
                                                steps = [('m', clf)]
                                                pipe = Pipeline(steps=steps)


                                        pipe.fit(X_train, y_train)
                                        y_pred = pipe.predict(X_test)  

                                        acc = accuracy_score(y_test, y_pred)
                                        conf_mat = confusion_matrix(y_test, y_pred)
                                        #pre_score = precision_score(y_test, y_pred, average='weighted')
                                        #recall_score = recall_score(y_test, y_pred, average='weighted')
                                        f1 = f1_score(y_test, y_pred, average='weighted')
                                        #auc = roc_auc_score(y_test, y_pred)

                                        st.write(f'Classifier = {classifer}')
                                        st.write(f'Accuracy =', acc)
                                        st.write(f'Confusion Matrix =', conf_mat)
                                        #st.write(f'Precision_score =', pre_score)
                                        #st.write(f'Recall_score =', recall_score)
                                        st.write(f'f1_score =', f1)
                                        #st.write(f'auc_score =', auc)

                        if mv == "Median":

                                for col in df.columns:
                                        if df[col].dtypes == "int64":
                                                df[col] = df[col].replace(to_replace = np.nan, value = df[col].median())
                                        elif df[col].dtypes == "float64":
                                                df[col] = df[col].replace(to_replace = np.nan, value = df[col].median())
                                        elif df[col].dtypes == "object":
                                                df[col] = df[col].replace(to_replace = np.nan, value = random.choice(df[col].unique()))
                                        else:
                                                df[col] = df[col].dropna()

                                classifer = st.sidebar.selectbox("Select classifier", ("SVM", "KNN", "Log Reg", "RF", "DT"))

                                all_columns = df.columns.to_list()
                                selected_columns = st.multiselect("Select non-target columns", all_columns)
                                new_df = df[selected_columns]
                                st.dataframe(new_df) #same as st.write(new_df)

                                selected_target_columns = st.multiselect("Select target column - Select at-least one to build the model", all_columns)
                                target_df = df[selected_target_columns]
                                st.dataframe(target_df) #same as st.write(target_df)

                                labelencoder = LabelEncoder()
                                for col in target_df.columns:
                                        target_df = pd.DataFrame(labelencoder.fit_transform(target_df[[col]]), columns = target_df.columns)

                                X = new_df
                                y = target_df
                                

                                def add_parameter_ui(clf_name):

                                        params = dict()
                                        if clf_name == 'SVM':
                                                C = st.sidebar.slider('C', 0.01, 10.0)
                                                params['C'] = C
                                        elif clf_name == 'KNN':
                                                K = st.sidebar.slider('K', 1, 15)
                                                params['K'] = K
                                        elif clf_name == 'Log Reg':
                                                C = st.sidebar.slider('C', 1, 15)
                                                params['C'] = C
                                        elif clf_name == 'RF':
                                                max_depth = st.sidebar.slider('max_depth', 2, 15)
                                                params['max_depth'] = max_depth
                                                n_estimators = st.sidebar.slider('n_estimators', 1, 50)
                                                params['n_estimators'] = n_estimators
                                        else:
                                                max_depth = st.sidebar.slider('max_depth', 2, 15)
                                                params['max_depth'] = max_depth
                                                min_samples_split = st.sidebar.slider('min_samples_split', 2, 15)
                                                params['min_samples_split'] = min_samples_split

                                        return params
                                
                                params = add_parameter_ui(classifer)

                                def get_classifier(clf_name, params):

                                        clf = None

                                        if clf_name == 'SVM':
                                                clf = SVC(C=params['C'])
                                        elif clf_name == 'KNN':
                                                clf = KNeighborsClassifier(n_neighbors=params['K'])
                                        elif clf_name == 'Log Reg':
                                                clf = LogisticRegression(C=params['C'])
                                        elif clf_name == 'RF':
                                                clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
                                                max_depth=params['max_depth'], random_state=1234)
                                        else:
                                                clf = DecisionTreeClassifier(max_depth=params['max_depth'],
                                                min_samples_split=params['min_samples_split'], random_state=1234)
                                        return clf

                                clf = get_classifier(classifer, params)

                                if st.checkbox("Build ML model with important evaluation mertics"):

                                        scaling_techniques = ["No scaling", "Standard scaling", "Min-Max scaling", "Robust scaling"]
                                        sca_tech = st.sidebar.radio("Scaling the train data and transform the test data", scaling_techniques)

                                        if sca_tech == "No scaling":

                                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1234) 

                                                st.write('Shape of X_train:', X_train.shape)
                                                st.write('Shape of X_test:', X_test.shape) 

                                        if sca_tech == "Standard scaling":

                                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1234) 
                                                sc = StandardScaler()
                                                X_train = sc.fit_transform(X_train) 
                                                X_test = sc.transform(X_test) 

                                                st.write('Shape of X_train:', X_train.shape)
                                                st.write('Shape of X_test:', X_test.shape) 

                                        if sca_tech == "Min-Max scaling":

                                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1234) 
                                                mm= MinMaxScaler()
                                                X_train = mm.fit_transform(X_train) 
                                                X_test = mm.transform(X_test) 

                                                st.write('Shape of X_train:', X_train.shape)
                                                st.write('Shape of X_test:', X_test.shape) 

                                        if sca_tech == "Robust scaling":

                                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1234) 
                                                rb= RobustScaler()
                                                X_train = rb.fit_transform(X_train) 
                                                X_test = rb.transform(X_test) 

                                                st.write('Shape of X_train:', X_train.shape)
                                                st.write('Shape of X_test:', X_test.shape) 
                                        

                                        sampling = st.sidebar.selectbox("Select sampling technique", ("SMOTE", "RUS", "ROS", "no sampling"))

                                        if sampling  == "SMOTE":
                                                st.subheader("Smote Oversampling technique")
                                                over = SMOTE(random_state = 1234)
                                                steps = [('os', over), ('m', clf)]
                                                pipe = Pipeline(steps=steps) 
                                        elif sampling == "RUS":
                                                st.subheader("Random Under sampling technique")
                                                rus = RandomUnderSampler(random_state = 1234)
                                                steps = [('u', rus), ('m', clf)]
                                                pipe = Pipeline(steps=steps) 
                                        elif sampling == "ROS":
                                                st.subheader("Random Over sampling technique")
                                                ros = RandomOverSampler(random_state = 1234)
                                                steps = [('o', ros), ('m', clf)]
                                                pipe = Pipeline(steps=steps) 
                                        elif sampling == "no sampling":
                                                st.subheader("No Sampling")
                                                steps = [('m', clf)]
                                                pipe = Pipeline(steps=steps)


                                        pipe.fit(X_train, y_train)
                                        y_pred = pipe.predict(X_test)  

                                        acc = accuracy_score(y_test, y_pred)
                                        conf_mat = confusion_matrix(y_test, y_pred)
                                        #pre_score = precision_score(y_test, y_pred, average='weighted')
                                        #recall_score = recall_score(y_test, y_pred, average='weighted')
                                        f1 = f1_score(y_test, y_pred, average='weighted')
                                        #auc = roc_auc_score(y_test, y_pred)

                                        st.write(f'Classifier = {classifer}')
                                        st.write(f'Accuracy =', acc)
                                        st.write(f'Confusion Matrix =', conf_mat)
                                        #st.write(f'Precision_score =', pre_score)
                                        #st.write(f'Recall_score =', recall_score)
                                        st.write(f'f1_score =', f1)
                                        #st.write(f'auc_score =', auc)



                elif choice == "PCA":
                        st.subheader("PCA visulaization of data")

                        Imputation_techniques = ["No Imputation", "Mean", "Median"]
                        mv = st.sidebar.radio("Impute Missing Values", Imputation_techniques)

                        if mv == "No Imputation":

                                all_columns = df.columns.to_list()
                                selected_columns = st.multiselect("Select non-target columns", all_columns)
                                new_df = df[selected_columns]
                                st.dataframe(new_df) #same as st.write(new_df)

                                selected_target_columns = st.multiselect("Select target column - Select at-least one to build the model", all_columns)
                                target_df = df[selected_target_columns]
                                st.dataframe(target_df) #same as st.write(target_df)

                                labelencoder = LabelEncoder()
                                for col in target_df.columns:
                                        target_df = pd.DataFrame(labelencoder.fit_transform(target_df[[col]]), columns = target_df.columns)

                                if st.checkbox("PCA Visualization"):

                                        scaling_techniques = ["No scaling", "Standard scaling", "Min-Max scaling", "Robust scaling"]
                                        scale = st.sidebar.radio("Scale the data", scaling_techniques)

                                        if scale == "No scaling":
                                                X = new_df
                                        
                                                for col in target_df.columns:
                                                        y = target_df[[col]]
                                                        z = col

                                                pca = PCA(n_components=2)
                                                components = pca.fit_transform(X)

                                                xs = components[:,0]
                                                ys = components[:,1]

                                                det = new_df.join(target_df)

                                                st.write(plt.scatter(xs, ys, c = det[z]))
                                                plt.xlabel('First principal component')
                                                plt.ylabel('Second Principal Component')
                                                st.pyplot()

                                        if scale == "Standard scaling":
                                                X = new_df

                                                sc = StandardScaler()
                                                X = pd.DataFrame(sc.fit_transform(X), columns = X.columns)
                                                
                                                for col in target_df.columns:
                                                        y = target_df[[col]]
                                                        z = col

                                                pca = PCA(n_components=2)
                                                components = pca.fit_transform(X)

                                                xs = components[:,0]
                                                ys = components[:,1]

                                                det = new_df.join(target_df)

                                                st.write(plt.scatter(xs, ys, c = det[z]))
                                                plt.xlabel('First principal component')
                                                plt.ylabel('Second Principal Component')
                                                st.pyplot()

                                        if scale == "Min-Max scaling":
                                                X = new_df

                                                mm = MinMaxScaler()
                                                X = pd.DataFrame(mm.fit_transform(X), columns = X.columns)
                                                
                                                for col in target_df.columns:
                                                        y = target_df[[col]]
                                                        z = col

                                                pca = PCA(n_components=2)
                                                components = pca.fit_transform(X)

                                                xs = components[:,0]
                                                ys = components[:,1]

                                                det = new_df.join(target_df)

                                                st.write(plt.scatter(xs, ys, c = det[z]))
                                                plt.xlabel('First principal component')
                                                plt.ylabel('Second Principal Component')
                                                st.pyplot()

                                        if scale == "Robust scaling":
                                                X = new_df

                                                rb = RobustScaler()
                                                X = pd.DataFrame(rb.fit_transform(X), columns = X.columns)
                                                
                                                for col in target_df.columns:
                                                        y = target_df[[col]]
                                                        z = col

                                                pca = PCA(n_components=2)
                                                components = pca.fit_transform(X)

                                                xs = components[:,0]
                                                ys = components[:,1]

                                                det = new_df.join(target_df)

                                                st.write(plt.scatter(xs, ys, c = det[z]))
                                                plt.xlabel('First principal component')
                                                plt.ylabel('Second Principal Component')
                                                st.pyplot()

                        elif mv == "Mean":

                                for col in df.columns:
                                        if df[col].dtypes == "int64":
                                                df[col] = df[col].replace(to_replace = np.nan, value = df[col].mean())
                                        elif df[col].dtypes == "float64":
                                                df[col] = df[col].replace(to_replace = np.nan, value = df[col].mean())
                                        elif df[col].dtypes == "object":
                                                df[col] = df[col].replace(to_replace = np.nan, value = random.choice(df[col].unique()))
                                        else:
                                                df[col] = df[col].dropna()

                                all_columns = df.columns.to_list()
                                selected_columns = st.multiselect("Select non-target columns", all_columns)
                                new_df = df[selected_columns]
                                st.dataframe(new_df) #same as st.write(new_df)

                                selected_target_columns = st.multiselect("Select target column - Select at-least one to build the model", all_columns)
                                target_df = df[selected_target_columns]
                                st.dataframe(target_df) #same as st.write(target_df)

                                labelencoder = LabelEncoder()
                                for col in target_df.columns:
                                        target_df = pd.DataFrame(labelencoder.fit_transform(target_df[[col]]), columns = target_df.columns)

                                if st.checkbox("PCA Visualization"):

                                        scaling_techniques = ["No scaling", "Standard scaling", "Min-Max scaling", "Robust scaling"]
                                        scale = st.sidebar.radio("Scale the data", scaling_techniques)

                                        if scale == "No scaling":
                                                X = new_df
                                        
                                                for col in target_df.columns:
                                                        y = target_df[[col]]
                                                        z = col

                                                pca = PCA(n_components=2)
                                                components = pca.fit_transform(X)

                                                xs = components[:,0]
                                                ys = components[:,1]

                                                det = new_df.join(target_df)

                                                st.write(plt.scatter(xs, ys, c = det[z]))
                                                plt.xlabel('First principal component')
                                                plt.ylabel('Second Principal Component')
                                                st.pyplot()

                                        if scale == "Standard scaling":
                                                X = new_df

                                                sc = StandardScaler()
                                                X = pd.DataFrame(sc.fit_transform(X), columns = X.columns)
                                                
                                                for col in target_df.columns:
                                                        y = target_df[[col]]
                                                        z = col

                                                pca = PCA(n_components=2)
                                                components = pca.fit_transform(X)

                                                xs = components[:,0]
                                                ys = components[:,1]

                                                det = new_df.join(target_df)

                                                st.write(plt.scatter(xs, ys, c = det[z]))
                                                plt.xlabel('First principal component')
                                                plt.ylabel('Second Principal Component')
                                                st.pyplot()

                                        if scale == "Min-Max scaling":
                                                X = new_df

                                                mm = MinMaxScaler()
                                                X = pd.DataFrame(mm.fit_transform(X), columns = X.columns)
                                                
                                                for col in target_df.columns:
                                                        y = target_df[[col]]
                                                        z = col

                                                pca = PCA(n_components=2)
                                                components = pca.fit_transform(X)

                                                xs = components[:,0]
                                                ys = components[:,1]

                                                det = new_df.join(target_df)

                                                st.write(plt.scatter(xs, ys, c = det[z]))
                                                plt.xlabel('First principal component')
                                                plt.ylabel('Second Principal Component')
                                                st.pyplot()

                                        if scale == "Robust scaling":
                                                X = new_df

                                                rb = RobustScaler()
                                                X = pd.DataFrame(rb.fit_transform(X), columns = X.columns)
                                                
                                                for col in target_df.columns:
                                                        y = target_df[[col]]
                                                        z = col

                                                pca = PCA(n_components=2)
                                                components = pca.fit_transform(X)

                                                xs = components[:,0]
                                                ys = components[:,1]

                                                det = new_df.join(target_df)

                                                st.write(plt.scatter(xs, ys, c = det[z]))
                                                plt.xlabel('First principal component')
                                                plt.ylabel('Second Principal Component')
                                                st.pyplot()

                        elif mv == "Median":

                                for col in df.columns:
                                        if df[col].dtypes == "int64":
                                                df[col] = df[col].replace(to_replace = np.nan, value = df[col].median())
                                        elif df[col].dtypes == "float64":
                                                df[col] = df[col].replace(to_replace = np.nan, value = df[col].median())
                                        elif df[col].dtypes == "object":
                                                df[col] = df[col].replace(to_replace = np.nan, value = random.choice(df[col].unique()))
                                        else:
                                                df[col] = df[col].dropna()

                                all_columns = df.columns.to_list()
                                selected_columns = st.multiselect("Select non-target columns", all_columns)
                                new_df = df[selected_columns]
                                st.dataframe(new_df) #same as st.write(new_df)

                                selected_target_columns = st.multiselect("Select target column - Select at-least one to build the model", all_columns)
                                target_df = df[selected_target_columns]
                                st.dataframe(target_df) #same as st.write(target_df)

                                labelencoder = LabelEncoder()
                                for col in target_df.columns:
                                        target_df = pd.DataFrame(labelencoder.fit_transform(target_df[[col]]), columns = target_df.columns)

                                if st.checkbox("PCA Visualization"):

                                        scaling_techniques = ["No scaling", "Standard scaling", "Min-Max scaling", "Robust scaling"]
                                        scale = st.sidebar.radio("Scale the data", scaling_techniques)

                                        if scale == "No scaling":
                                                X = new_df
                                        
                                                for col in target_df.columns:
                                                        y = target_df[[col]]
                                                        z = col

                                                pca = PCA(n_components=2)
                                                components = pca.fit_transform(X)

                                                xs = components[:,0]
                                                ys = components[:,1]

                                                det = new_df.join(target_df)

                                                st.write(plt.scatter(xs, ys, c = det[z]))
                                                plt.xlabel('First principal component')
                                                plt.ylabel('Second Principal Component')
                                                st.pyplot()

                                        if scale == "Standard scaling":
                                                X = new_df

                                                sc = StandardScaler()
                                                X = pd.DataFrame(sc.fit_transform(X), columns = X.columns)
                                                
                                                for col in target_df.columns:
                                                        y = target_df[[col]]
                                                        z = col

                                                pca = PCA(n_components=2)
                                                components = pca.fit_transform(X)

                                                xs = components[:,0]
                                                ys = components[:,1]

                                                det = new_df.join(target_df)

                                                st.write(plt.scatter(xs, ys, c = det[z]))
                                                plt.xlabel('First principal component')
                                                plt.ylabel('Second Principal Component')
                                                st.pyplot()

                                        if scale == "Min-Max scaling":
                                                X = new_df

                                                mm = MinMaxScaler()
                                                X = pd.DataFrame(mm.fit_transform(X), columns = X.columns)
                                                
                                                for col in target_df.columns:
                                                        y = target_df[[col]]
                                                        z = col

                                                pca = PCA(n_components=2)
                                                components = pca.fit_transform(X)

                                                xs = components[:,0]
                                                ys = components[:,1]

                                                det = new_df.join(target_df)

                                                st.write(plt.scatter(xs, ys, c = det[z]))
                                                plt.xlabel('First principal component')
                                                plt.ylabel('Second Principal Component')
                                                st.pyplot()

                                        if scale == "Robust scaling":
                                                X = new_df

                                                rb = RobustScaler()
                                                X = pd.DataFrame(rb.fit_transform(X), columns = X.columns)
                                                
                                                for col in target_df.columns:
                                                        y = target_df[[col]]
                                                        z = col

                                                pca = PCA(n_components=2)
                                                components = pca.fit_transform(X)

                                                xs = components[:,0]
                                                ys = components[:,1]

                                                det = new_df.join(target_df)

                                                st.write(plt.scatter(xs, ys, c = det[z]))
                                                plt.xlabel('First principal component')
                                                plt.ylabel('Second Principal Component')
                                                st.pyplot()
              
                elif choice == "t-SNE":
                        st.subheader("t-Distributed Stochastic Neighbor Embedding")

                        Imputation_techniques = ["No Imputation", "Mean", "Median"]
                        mv = st.sidebar.radio("Impute Missing Values", Imputation_techniques)

                        if mv == "No Imputation":

                                all_columns = df.columns.to_list()
                                selected_columns = st.multiselect("Select non-target columns", all_columns)
                                new_df = df[selected_columns]
                                st.dataframe(new_df) #same as st.write(new_df)

                                selected_target_columns = st.multiselect("Select target column - Select at-least one to build the model", all_columns)
                                target_df = df[selected_target_columns]
                                st.dataframe(target_df) #same as st.write(target_df)

                                labelencoder = LabelEncoder()
                                for col in target_df.columns:
                                                target_df = pd.DataFrame(labelencoder.fit_transform(target_df[[col]]), columns = target_df.columns)
                                
                                df = df.round(1)

                                if st.checkbox("t-SNE Visualization"):

                                        scaling_techniques = ["No scaling", "Standard scaling", "Min-Max scaling", "Robust scaling"]
                                        scale = st.sidebar.radio("Scale the data", scaling_techniques)

                                        if scale == "No scaling":
                                                X = new_df
                                                
                                                for col in target_df.columns:
                                                        y = target_df[[col]]
                                                        z = col
                                                
                                                det = new_df.join(target_df)
                                                model = TSNE(learning_rate = 200)
                                                tsne_features = model.fit_transform(X)

                                                det["S"] = tsne_features[:,0]
                                                det["P"] = tsne_features[:,1]

                                                sns.scatterplot(x = "S", y = "P", hue = z, data = det)
                                                st.pyplot()

                                        if scale == "Standard scaling":
                                                X = new_df

                                                sc = StandardScaler()
                                                new_df = pd.DataFrame(sc.fit_transform(new_df), columns = new_df.columns)
                                                
                                                for col in target_df.columns:
                                                        y = target_df[[col]]
                                                        z = col
                                                
                                                det = new_df.join(target_df)
                                                model = TSNE(learning_rate = 200)
                                                tsne_features = model.fit_transform(X)

                                                det["S"] = tsne_features[:,0]
                                                det["P"] = tsne_features[:,1]

                                                sns.scatterplot(x = "S", y = "P", hue = z, data = det)
                                                st.pyplot()

                                        if scale == "Min-Max scaling":
                                                X = new_df

                                                mm = MinMaxScaler()
                                                new_df = pd.DataFrame(mm.fit_transform(new_df), columns = new_df.columns)
                                                
                                                for col in target_df.columns:
                                                        y = target_df[[col]]
                                                        z = col
                                                
                                                det = new_df.join(target_df)
                                                model = TSNE(learning_rate = 200)
                                                tsne_features = model.fit_transform(X)

                                                det["S"] = tsne_features[:,0]
                                                det["P"] = tsne_features[:,1]

                                                sns.scatterplot(x = "S", y = "P", hue = z, data = det)
                                                st.pyplot()

                                        if scale == "Robust scaling":
                                                X = new_df

                                                rb = RobustScaler()
                                                new_df = pd.DataFrame(rb.fit_transform(new_df), columns = new_df.columns)
                                                
                                                for col in target_df.columns:
                                                        y = target_df[[col]]
                                                        z = col
                                                
                                                det = new_df.join(target_df)
                                                model = TSNE(learning_rate = 200)
                                                tsne_features = model.fit_transform(X)

                                                det["S"] = tsne_features[:,0]
                                                det["P"] = tsne_features[:,1]

                                                sns.scatterplot(x = "S", y = "P", hue = z, data = det)
                                                st.pyplot()

                        elif mv == "Mean":

                                for col in df.columns:
                                        if df[col].dtypes == "int64":
                                                df[col] = df[col].replace(to_replace = np.nan, value = df[col].mean())
                                        elif df[col].dtypes == "float64":
                                                df[col] = df[col].replace(to_replace = np.nan, value = df[col].mean())
                                        elif df[col].dtypes == "object":
                                                df[col] = df[col].replace(to_replace = np.nan, value = random.choice(df[col].unique()))
                                        else:
                                                df[col] = df[col].dropna()

                                df = df.round(1)

                                all_columns = df.columns.to_list()
                                selected_columns = st.multiselect("Select non-target columns", all_columns)
                                new_df = df[selected_columns]
                                st.dataframe(new_df) #same as st.write(new_df)

                                selected_target_columns = st.multiselect("Select target column - Select at-least one to build the model", all_columns)
                                target_df = df[selected_target_columns]
                                st.dataframe(target_df) #same as st.write(target_df)

                                labelencoder = LabelEncoder()
                                for col in target_df.columns:
                                                target_df = pd.DataFrame(labelencoder.fit_transform(target_df[[col]]), columns = target_df.columns)
                                
                                if st.checkbox("t-SNE Visualization"):

                                        scaling_techniques = ["No scaling", "Standard scaling", "Min-Max scaling", "Robust scaling"]
                                        scale = st.sidebar.radio("Scale the data", scaling_techniques)

                                        if scale == "No scaling":
                                                X = new_df
                                                
                                                for col in target_df.columns:
                                                        y = target_df[[col]]
                                                        z = col
                                                
                                                det = new_df.join(target_df)
                                                model = TSNE(learning_rate = 200)
                                                tsne_features = model.fit_transform(X)

                                                det["S"] = tsne_features[:,0]
                                                det["P"] = tsne_features[:,1]

                                                sns.scatterplot(x = "S", y = "P", hue = z, data = det)
                                                st.pyplot()

                                        if scale == "Standard scaling":
                                                X = new_df

                                                sc = StandardScaler()
                                                new_df = pd.DataFrame(sc.fit_transform(new_df), columns = new_df.columns)
                                                
                                                for col in target_df.columns:
                                                        y = target_df[[col]]
                                                        z = col
                                                
                                                det = new_df.join(target_df)
                                                model = TSNE(learning_rate = 200)
                                                tsne_features = model.fit_transform(X)

                                                det["S"] = tsne_features[:,0]
                                                det["P"] = tsne_features[:,1]

                                                sns.scatterplot(x = "S", y = "P", hue = z, data = det)
                                                st.pyplot()

                                        if scale == "Min-Max scaling":
                                                X = new_df

                                                mm = MinMaxScaler()
                                                new_df = pd.DataFrame(mm.fit_transform(new_df), columns = new_df.columns)
                                                
                                                for col in target_df.columns:
                                                        y = target_df[[col]]
                                                        z = col
                                                
                                                det = new_df.join(target_df)
                                                model = TSNE(learning_rate = 200)
                                                tsne_features = model.fit_transform(X)

                                                det["S"] = tsne_features[:,0]
                                                det["P"] = tsne_features[:,1]

                                                sns.scatterplot(x = "S", y = "P", hue = z, data = det)
                                                st.pyplot()

                                        if scale == "Robust scaling":
                                                X = new_df

                                                rb = RobustScaler()
                                                new_df = pd.DataFrame(rb.fit_transform(new_df), columns = new_df.columns)
                                                
                                                for col in target_df.columns:
                                                        y = target_df[[col]]
                                                        z = col
                                                
                                                det = new_df.join(target_df)
                                                model = TSNE(learning_rate = 200)
                                                tsne_features = model.fit_transform(X)

                                                det["S"] = tsne_features[:,0]
                                                det["P"] = tsne_features[:,1]

                                                sns.scatterplot(x = "S", y = "P", hue = z, data = det)
                                                st.pyplot()


                        elif mv == "Median":

                                for col in df.columns:
                                        if df[col].dtypes == "int64":
                                                df[col] = df[col].replace(to_replace = np.nan, value = df[col].median())
                                        elif df[col].dtypes == "float64":
                                                df[col] = df[col].replace(to_replace = np.nan, value = df[col].median())
                                        elif df[col].dtypes == "object":
                                                df[col] = df[col].replace(to_replace = np.nan, value = random.choice(df[col].unique()))
                                        else:
                                                df[col] = df[col].dropna()

                                df = df.round(1)

                                all_columns = df.columns.to_list()
                                selected_columns = st.multiselect("Select non-target columns", all_columns)
                                new_df = df[selected_columns]
                                st.dataframe(new_df) #same as st.write(new_df)

                                selected_target_columns = st.multiselect("Select target column - Select at-least one to build the model", all_columns)
                                target_df = df[selected_target_columns]
                                st.dataframe(target_df) #same as st.write(target_df)

                                labelencoder = LabelEncoder()
                                for col in target_df.columns:
                                                target_df = pd.DataFrame(labelencoder.fit_transform(target_df[[col]]), columns = target_df.columns)

                                if st.checkbox("t-SNE Visualization"):

                                        scaling_techniques = ["No scaling", "Standard scaling", "Min-Max scaling", "Robust scaling"]
                                        scale = st.sidebar.radio("Scale the data", scaling_techniques)

                                        if scale == "No scaling":
                                                X = new_df
                                                
                                                for col in target_df.columns:
                                                        y = target_df[[col]]
                                                        z = col
                                                
                                                det = new_df.join(target_df)
                                                model = TSNE(learning_rate = 200)
                                                tsne_features = model.fit_transform(X)

                                                det["S"] = tsne_features[:,0]
                                                det["P"] = tsne_features[:,1]

                                                sns.scatterplot(x = "S", y = "P", hue = z, data = det)
                                                st.pyplot()

                                        if scale == "Standard scaling":
                                                X = new_df

                                                sc = StandardScaler()
                                                new_df = pd.DataFrame(sc.fit_transform(new_df), columns = new_df.columns)
                                                
                                                for col in target_df.columns:
                                                        y = target_df[[col]]
                                                        z = col
                                                
                                                det = new_df.join(target_df)
                                                model = TSNE(learning_rate = 200)
                                                tsne_features = model.fit_transform(X)

                                                det["S"] = tsne_features[:,0]
                                                det["P"] = tsne_features[:,1]

                                                sns.scatterplot(x = "S", y = "P", hue = z, data = det)
                                                st.pyplot()

                                        if scale == "Min-Max scaling":
                                                X = new_df

                                                mm = MinMaxScaler()
                                                new_df = pd.DataFrame(mm.fit_transform(new_df), columns = new_df.columns)
                                                
                                                for col in target_df.columns:
                                                        y = target_df[[col]]
                                                        z = col
                                                
                                                det = new_df.join(target_df)
                                                model = TSNE(learning_rate = 200)
                                                tsne_features = model.fit_transform(X)

                                                det["S"] = tsne_features[:,0]
                                                det["P"] = tsne_features[:,1]

                                                sns.scatterplot(x = "S", y = "P", hue = z, data = det)
                                                st.pyplot()

                                        if scale == "Robust scaling":
                                                X = new_df

                                                rb = RobustScaler()
                                                new_df = pd.DataFrame(rb.fit_transform(new_df), columns = new_df.columns)
                                                
                                                for col in target_df.columns:
                                                        y = target_df[[col]]
                                                        z = col
                                                
                                                det = new_df.join(target_df)
                                                model = TSNE(learning_rate = 200)
                                                tsne_features = model.fit_transform(X)

                                                det["S"] = tsne_features[:,0]
                                                det["P"] = tsne_features[:,1]

                                                sns.scatterplot(x = "S", y = "P", hue = z, data = det)
                                                st.pyplot()
                                   
                
if __name__ == '__main__':
    main()

