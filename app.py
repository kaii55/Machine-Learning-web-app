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


from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
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
        st.title("Welcome to Auro's Machine learning shop")
        

        st.write("This is a classification machine learning web-app; We are testing numerous tabular classification data to make it better and better :sunglasses:.")
        st.write("For testing purpose and bug fixing, this web app only supports tabular data in Excel Workbook (.xlsx) format. Later we will add other file extensions to upload. :bell:")
        st.write("This data set has been tested on the classic traditional ***Iris*** , ***Breast-cancer*** , ***Wine*** , ***Credit-card-fraud*** and other binary Injury classification dataset. :milky_way:")
        st.write("The final goal is to empower Data Science and Machine learning with automated data analysis and ML Model building. :blossom:")
        st.write("We will constantly update this web-app with more features e.g., better and complex visualizations, feature selection techniques, feature engineering etc. :new_moon:")
        st.write("We have imputed the missing values by ourselves with simple imputation techniques. But, later we will add advanced imputation techniques and users will have their own choice to fill :incoming_envelope: .")
        st.write("Credits: Streamlit, JCharisTech & J-Secur1ty for exceptional tutroials.")
        st.write("Please visit to my website: https://kaii55.github.io  :wolf:")
        st.write("My github profile: https://github.com/kaii55")

        st.text("Upload and analyze any classification problem - Tabular data only")

        missing_values = ["Before Imputation", "After Imputation"]
        mv = st.sidebar.radio("Show Missing Values", missing_values)

        activities = ["About", "EDA", "Plot", "Model Building"]
        choice = st.sidebar.selectbox("Select Activity", activities)

        
        data = st.file_uploader("Upload dataset", type = [ "xlsx"])

       


        if data is not None:
                df = pd.read_excel(data)
                st.dataframe(df.head())

                if mv == "Before Imputation":
                        st.write(df.isnull().sum())
                elif mv == "After Imputation":
                        for col in df.columns:
                                if df[col].dtypes == "int64":
                                        df[col] = df[col].replace(to_replace = np.nan, value = df[col].mean())
                                elif df[col].dtypes == "float64":
                                        df[col] = df[col].replace(to_replace = np.nan, value = df[col].mean())
                                elif df[col].dtypes == "object":
                                        df[col] = df[col].replace(to_replace = np.nan, value = random.choice(df[col].unique()))
                                else:
                                        df[col] = df[col].dropna()

                 
                        st.write(df.isnull().sum())
                
                df = df.round(1)

                if choice == "About":
                        st.write("Play with data and have fun :smiley:")
                        
                elif choice == "EDA":
                        st.subheader("Exploratory Data Analysis")

                        if st.checkbox("show shape"):
                                st.write(df.shape)
                        if st.checkbox("show columns"):
                                all_columns = df.columns.to_list()
                                st.write(all_columns)
                        if st.checkbox("show summary"):
                                st.write(df.describe())
                        if st.checkbox("show int and float columns"):
                                a = df.select_dtypes(include=[np.float64,np.int64])
                                st.write(a)
                        if st.checkbox("show category columns"):
                                a = df.select_dtypes(include=[object])
                                st.write(a) 
                        
                

                elif choice == "Plot":
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
                        

                        if st.checkbox("Coorelation among the non-target columns"):
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
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1234)
                                sc = StandardScaler()
                                X_train = sc.fit_transform(X_train) 
                                X_test = sc.transform(X_test) 

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


                
if __name__ == '__main__':
    main()

