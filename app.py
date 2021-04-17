import os
import base64
import streamlit as st
 
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')
import seaborn as sns 
import streamlit as st 
from sklearn.ensemble import  AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,plot_confusion_matrix,plot_roc_curve,precision_score,recall_score,precision_recall_curve,roc_auc_score,auc
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
import base64
import streamlit as st
import numpy as np
import streamlit as st
import pandas as pd
# Security
#passlib,hashlib,bcrypt,scrypt
import hashlib
import sqlite3 
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()
 
def check_hashes(password,hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False
# DB Management
 
conn = sqlite3.connect('data.db')
c = conn.cursor()
# DB  Functions
def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')
 
 
def add_userdata(username,password):
    c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
    conn.commit()
 
def login_user(username,password):
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
    data = c.fetchall()
    return data
 
 
def view_all_users():
    c.execute('SELECT * FROM userstable')
    data = c.fetchall()
    return data
 
 
def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href
try:
 
    from enum import Enum
    from io import BytesIO, StringIO
    from typing import Union
 
    import pandas as pd
    import streamlit as st
except Exception as e:
    print(e)
import streamlit.components.v1 as stc
def myApp():
  
  html_temp = """
  <div style="background-color:#000080;border-radius:5px;"><p style="color:white;font-weight:10px;font-size:28px;padding:10px">Multi-Domain machine learning platform for the novice</p></div>
  """
  st.markdown(html_temp,unsafe_allow_html=True)
  st.set_option('deprecation.showfileUploaderEncoding', False)
  st.subheader("Dataset")
  datasetchoice = st.radio("Do you what to use your own dataset?", ("Yes", "No"))
  if datasetchoice=='No':
    def file_selector(folder_path='./datasets'):
      filenames = os.listdir(folder_path)
      selected_filename = st.selectbox("Select A file",filenames)
      return os.path.join(folder_path,selected_filename)
  
    filename = file_selector()
    st.info("You Selected {}".format(filename))
    
    def writetofile(text,file_name):
      with open(os.path.join('./datasets',file_name),'w') as f:
        f.write(text)
      return file_name
    def make_downloadable(filename):
      readfile = open(os.path.join("./datasets",filename)).read()
      b64 = base64.b64encode(readfile.encode()).decode()
      href = 'Download File File (right-click and save as <some_name>.txt)'.format(b64)
      return href 
    # Read Data
    df = pd.read_csv(filename)
    # Show Dataset
    st.subheader("Data Explonatory Analysis")
    st.info("This part refers to the various ways to explore your choosen data because When you have a raw data set, it won't provide any insight until you start to organize it. for more info check this link: https://fluvid.com/videos/detail/EDRPXuo-2aS5Ak4PM")
    if st.checkbox("Show Dataset"):
      st.dataframe(df)
  
    # Show Columns
    if st.button("Column Names"):
      st.success("This is the name of your featuresin your dataset")
      st.write(df.columns)
  
    # Show Shape
    if st.checkbox("Shape of Dataset"):
      st.success("Here you will see number of Rows and Columns and shape of your entire dataset")
      
      data_dim = st.radio("Show Dimensions By ",("Rows","Columns"))
      if data_dim == 'Rows':
        st.text("Number of Rows")
        st.write(df.shape[0])
      elif data_dim == 'Columns':
        st.text("Number of Columns")
        st.write(df.shape[1])
      else:
        st.write(df.shape)
  
    # Select Columns
    st.info("If you want to visualize the column you want only for better understanding your dataset?")
    if st.checkbox("Select Columns To Show"):
      all_columns = df.columns.tolist()
      selected_columns = st.multiselect("Select",all_columns)
      new_df = df[selected_columns]
      st.dataframe(new_df)
  
    # Show Values
    if st.button("Value Counts"):
      st.info("This part shows the value count of target in your dataset?")
      st.text("Value Counts By Target/Class")
      st.write(df.iloc[:,-1].value_counts())
  
  
    # Show Datatypes
    if st.button("Data Types"):
      st.info("This part specifies the type of data your attributes in your Dataset have?")
      st.write(df.dtypes)
  
  
    # Show Summary
    st.info("Now let 's visualize Statistical Analysis of the chosen dataset,min,max,etc")
    if st.checkbox("Summary"):
      st.write(df.describe().T)
  
    ## and Visualization
  
    st.subheader("Data Visualization")
    # Correlation
    # Seaborn Plot
    #measures the relationship between two variables, that is, how they are linked to each other
    st.info("Now you can perform the graphical representation of information and data. By using visual elements like charts, graphs. Data visualization tools will provide an accessible way to see and understand trends, outliers, and patterns in datasets")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    if st.checkbox("Correlation Plot[Seaborn]"):
      st.success("Correlation measures the relationship between two variables,how they are linked to each other")
      st.write(sns.heatmap(df.corr(),annot=True))
      st.pyplot()
    
  
    # Pie Chart
    if st.checkbox("Pie Plot"):
      st.set_option('deprecation.showPyplotGlobalUse', False)
      all_columns_names = df.columns.tolist()
      if st.button("Generate Pie Plot"):
        st.success("Generating A Pie Plot")
        st.write(df.iloc[:,-1].value_counts().plot.pie(autopct="%1.1f%%"))
        st.pyplot()
  
    # Count
    if st.checkbox("Plot of Value Counts"):
      st.text("Value Counts By Target")
      st.set_option('deprecation.showPyplotGlobalUse', False)
      all_columns_names = df.columns.tolist()
      primary_col = st.selectbox("Primary Columm to GroupBy",all_columns_names)
      selected_columns_names = st.multiselect("Select Columns",all_columns_names)
      if st.button("Plot"):
        st.success(" this part select the columns you want to plot")
        st.text("Generate Plot")
        if selected_columns_names:
          vc_plot = df.groupby(primary_col)[selected_columns_names].count()
        else:
          vc_plot = df.iloc[:,-1].value_counts()
        st.write(vc_plot.plot(kind="bar"))
        st.pyplot()
  
  
    # Customizable Plot
  
    st.subheader("Customizable Plot")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    all_columns_names = df.columns.tolist()
    type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
    selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)
  
    if st.button("Generate Plot"):
      st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))
  
      # Plot By Streamlit
      if type_of_plot == 'area':
        st.set_option('deprecation.showPyplotGlobalUse', False)
        cust_data = df[selected_columns_names]
        st.area_chart(cust_data)
  
      elif type_of_plot == 'bar':
        st.set_option('deprecation.showPyplotGlobalUse', False)
        cust_data = df[selected_columns_names]
        st.bar_chart(cust_data)
  
      elif type_of_plot == 'line':
        st.set_option('deprecation.showPyplotGlobalUse', False)
        cust_data = df[selected_columns_names]
        st.line_chart(cust_data)
  
      # Custom Plot 
      elif type_of_plot:
        st.set_option('deprecation.showPyplotGlobalUse', False)
        cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
        st.write(cust_plot)
        st.pyplot()
  
      if st.button("End of Data Exploration"):
        st.balloons()
    st.subheader("Data Cleaning")
    st.info("Preparing dataset for analysis by removing or modifying data that is incorrect, incomplete, irrelevant, duplicated, or improperly formatted.")
    if st.checkbox("Visualize null value"):
      st.success("Generating features which is having null values in your dataset")
      st.dataframe(df.isnull().sum())
    if st.checkbox("Visualize categorical features"):
      st.success("Generating non numeric features in your dataset")
      categorical_feature_columns = list(set(df.columns) - set(df._get_numeric_data().columns))
      dt=df[categorical_feature_columns]
      st.dataframe(dt)
    if st.checkbox("Encoding features"):
      st.success("Converting non numeric features into numerical feature in your dataset")
      categorical_feature_columns = list(set(df.columns) - set(df._get_numeric_data().columns))
      label= LabelEncoder()
      for col in df[categorical_feature_columns]:
        df[col]=label.fit_transform(df[col])
      st.dataframe(df)
    Y = df.target
    X = df.drop(columns=['target'])
    
    
    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.33, random_state=8) 
    from sklearn.preprocessing import StandardScaler
    sl=StandardScaler()
    X_trained= sl.fit_transform(X_train)
    X_tested= sl.fit_transform(X_test)
    if st.checkbox("Scaling your dataset"):
      st.dataframe(X_trained)
      
      
      
    st.subheader("Feature Engineering")
    st.info("Now extract features from your dataset to improve the performance of machine learning algorithms")
    try:
  
      if st.checkbox("Select Columns for creation of model"):
        all_columns = df.columns.tolist()
        select_columns = st.multiselect("Select",all_columns,key='engenering')
        new_df = df[select_columns]
        df=new_df
        categorical_feature_columns = list(set(df.columns) - set(df._get_numeric_data().columns))
        label= LabelEncoder()
        for col in df[categorical_feature_columns]:
          df[col]=label.fit_transform(df[col])
        st.dataframe(df)
    except Exception as e:
      st.write("please choose target attribute")
  
  
    st.subheader('Data Preparation')
    st.button('Now that we have done selecting the data set let see the summary for what we have done so far')
    st.write("Wrangle data and prepare it for training,Clean that which may require it (remove duplicates, correct errors, deal with missing values, normalization, data type conversion,Randomize data, which erases the effects of the particular order in which we collected and/or otherwise prepared our data,Visualize data to help detect relevant relationships between variables or class imbalances (bias alert!), or perform other exploratory analysis,Split into training and evaluation sets")
    if st.checkbox(" Click here to see next steps"):
      st.write(" 1 step : Choose a Model: Different algorithms are  provides for different tasks; choose the right one")
      st.write(" 2 step : Train the Model: The goal of training is to answer a question or make a prediction correctly as often as possible")
      st.write(" 3 step : Evaluate the Model: Uses some metric or combination of metrics to objective performance of model example accuracy score,confusion metrics,precision call,etc..")
      st.write(" 4 step : Parameter Tuning: This step refers to hyperparameter tuning, which is an artform as opposed to a science,Tune model parameters for improved performance,Simple model hyperparameters may include: number of training steps, learning rate, initialization values and distribution, etc.")
      st.write(" 5 step : Using further (test set) data which have, until this point, been withheld from the model (and for which class labels are known), are used to test the model; a better approximation of how the model will perform in the real world")
  
      
    st.sidebar.subheader('Choose Classifer')
    classifier_name = st.sidebar.selectbox(
        'Choose classifier',
        ('KNN', 'SVM', 'Random Forest','Logistic Regression','GradientBoosting','ADABoost','Unsupervised Learning(K-MEANS)','Deep Learning')
    )
    
    
    Y = df.target
    X = df.drop(columns=['target'])
    
    
    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.33, random_state=8)
    
    from sklearn.preprocessing import StandardScaler
    sl=StandardScaler()
    X_trained= sl.fit_transform(X_train)
    X_tested= sl.fit_transform(X_test)
    
    class_name=['yes','no']  

    
      
            
  
        
    if classifier_name == 'Unsupervised Learning(K-MEANS)':
        st.sidebar.subheader('Model Hyperparmeter')
        n_clusters= st.sidebar.number_input("number of clusters",2,10,step=1,key='clusters')
        save_option= st.sidebar.radio("Save to file" ,("Yes","No"))
        if st.sidebar.button("Classify",key='unspervised'):  
            sc = StandardScaler()
            X_transformed = sc.fit_transform(df)
            pca = PCA(n_components=2).fit_transform(X_transformed) # calculation Cov matrix is embeded in PCA
            kmeans = KMeans(n_clusters)
            kmeans.fit(pca)
            filename = 'kmeans_model.sav'
            pickle.dump(kmeans, open(filename, 'wb'))
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.scatter(pca[:,0],pca[:,1], c=kmeans.labels_, cmap='rainbow')
            plt.title('Clustering Projection');
            st.pyplot()
          
            if save_option == 'Yes':
                st.markdown(get_binary_file_downloader_html('kmeans_model.sav', 'Model Download'), unsafe_allow_html=True)
                st.success("model successfully saved")
                #file_to_download = writetofile(kmeans,file)
                #st.info("Saved Result As :: {}".format(file))
                # d_link= make_downloadable(file_to_download)
                # st.markdown(d_link,unsafe_allow_html=True)
                  
            # else:
                #st.subheader("Downloads List")
                #files = os.listdir(os.path.join('./datasets'))
                #file_to_download = st.selectbox("Select File To Download",files)
                # st.info("File Name: {}".format(file_to_download))
                #d_link = make_downloadable(file_to_download)
                #st.markdown(d_link,unsafe_allow_html=True)
                    
    if classifier_name == 'Deep Learning':
        st.sidebar.subheader('Model Hyperparmeter')
        epochs= st.sidebar.slider("number of Epoch",1,30,key='epoch')
        units= st.sidebar.number_input("Dense layers",3,30,key='units')
        #rate= st.sidebar.slider("Learning Rate",0,5,step=1,key='rates')
        activation= st.sidebar.radio("Activation Function",("softmax","sigmoid"),key='activations')
        optimizer= st.sidebar.radio("Optimizer",("rmsprop","Adam"),key='opts')
        
        if st.sidebar.button("classify",key='deeps'):
            X_train = X_train / 256.
            model = Sequential()
            model.add(Flatten())
            model.add(Dense(units=units,activation='relu'))
            model.add(Dense(units=units,activation=activation))
            model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
            model.fit(X_train.values, y_train.values, epochs=epochs)
            test_loss, test_acc =model.evaluate(X_test.values,  y_test.values, verbose=2)
            st.write('Deep Learning Model accuracy: ',test_acc)
        
    if classifier_name == 'SVM':
        st.sidebar.subheader('Model Hyperparmeter')
        c= st.sidebar.number_input("c(Reguralization)",0.01,10.0,step=0.01,key='c')
        kernel= st.sidebar.radio("kernel",("linear","rbf"),key='kernel')
        gamma= st.sidebar.radio("gamma(kernel coefficiency",("scale","auto"),key='gamma')
        save_option= st.sidebar.radio("Save to file" ,("Yes","No"))
        metrics= st.sidebar.multiselect("What is the metrics to plot?",('confusion matrix','roc_curve','precision_recall_curve'))
      
              
        
        if st.sidebar.button("classify",key='classify'):
            st.subheader("SVM result")
            svcclassifier= SVC(C=c,kernel=kernel,gamma=gamma)
            svcclassifier.fit(X_trained,y_train)
            y_pred= svcclassifier.predict(X_tested)
            acc= accuracy_score(y_test,y_pred)
            st.write("Accuracy:",acc.round(2))
    #     st.write("precision_score:",precision_score(y_test,y_pred,average='micro').round(2))
            st.write("recall_score:",recall_score(y_test,y_pred,average='micro').round(2))
            if save_option == 'Yes':
                with open('mysaved_md_pickle', 'wb') as file:
                    pickle.dump(svcclassifier,file)
                st.success("model successfully saved")
            if 'confusion matrix' in metrics:
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.subheader('confusion matrix')
                plot_confusion_matrix(svcclassifier,X_tested,y_test)
                st.pyplot()
            if 'roc_curve' in metrics:
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.subheader('plot_roc_curve')
                plot_roc_curve(svcclassifier,X_tested,y_test,normalize=False)
                st.pyplot()
            if 'precision_recall_curve' in metrics:
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.subheader('precision_recall_curve')
                plot_roc_curve(svcclassifier,X_tested,y_test,normalize=False)
                st.pyplot()
            
    
    
    if classifier_name == 'Logistic Regression':
        st.sidebar.subheader('Model Hyperparmeter')
        c= st.sidebar.number_input("c(Reguralization)",0.01,10.0,step=0.01,key='Logistic')
        max_iter= st.sidebar.slider("maximum number of iteration",100,500,key='max_item')
      
    
        metrics= st.sidebar.multiselect("What is the metrics to plot?",('confusion matrix','roc_curve','precision_recall_curve'))
    
        if st.sidebar.button("classify",key='classify'):
            st.subheader("Logistic Regression result")
            Regression= LogisticRegression(C=c,max_iter=max_iter)
            Regression.fit(X_trained,y_train)
            y_prediction= Regression.predict(X_tested)
            acc= accuracy_score(y_test,y_prediction)
            st.write("Accuracy:",acc.round(2))
            st.write("precision_score:",precision_score(y_test,y_prediction,average='micro').round(2))
            st.write("recall_score:",recall_score(y_test,y_prediction,average='micro').round(2))
            if 'confusion matrix' in metrics:
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.subheader('confusion matrix')
                plot_confusion_matrix(Regression,X_tested,y_test)
                st.pyplot()
            if 'roc_curve' in metrics:
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.subheader('plot_roc_curve')
                plot_roc_curve(Regression,X_tested,y_test)
                st.pyplot()
            if 'precision_recall_curve' in metrics:
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.subheader('precision_recall_curve')
                plot_roc_curve(Regression,X_tested,y_test)
                st.pyplot()
            
                
    
    if classifier_name == 'Random Forest':
        st.sidebar.subheader('Model Hyperparmeter')
        n_estimators= st.sidebar.number_input("Number of trees in the forest",100,5000,step=10,key='estimators')
        max_depth= st.sidebar.number_input("maximum depth of tree",1,20,step=1,key='max_depth')
        bootstrap= st.sidebar.radio("Boostrap sample when building trees",("True","False"),key='boostrap')
    
    
        metrics= st.sidebar.multiselect("What is the metrics to plot?",('confusion matrix','roc_curve','precision_recall_curve'))
    
        if st.sidebar.button("classify",key='classify'):
            st.subheader("Random Forest result")
            model= RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,bootstrap=bootstrap)
            model.fit(X_trained,y_train)
            y_prediction= model.predict(X_tested)
            acc= accuracy_score(y_test,y_prediction)
            st.write("Accuracy:",acc.round(2))
            st.write("precision_score:",precision_score(y_test,y_prediction,average='micro').round(2))
            st.write("recall_score:",recall_score(y_test,y_prediction,average='micro').round(2))
            if 'confusion matrix' in metrics:
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.subheader('confusion matrix')
                plot_confusion_matrix(model,X_tested,y_test)
                st.pyplot()
            if 'roc_curve' in metrics:
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.subheader('plot_roc_curve')
                plot_roc_curve(model,X_tested,y_test)
                st.pyplot()
            if 'precision_recall_curve' in metrics:
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.subheader('precision_recall_curve')
                plot_roc_curve(model,X_tested,y_test)
                st.pyplot() 
    
    
    if classifier_name == 'KNN':
        st.sidebar.subheader('Model Hyperparmeter')
        n_neighbors= st.sidebar.number_input("Number of n_neighbors",5,30,step=1,key='neighbors')
        leaf_size= st.sidebar.slider("leaf size",30,200,key='leaf')
        weights= st.sidebar.radio("weight function used in prediction",("uniform","distance"),key='weight')
    
    
        metrics= st.sidebar.multiselect("What is the metrics to plot?",('confusion matrix','roc_curve','precision_recall_curve'))
    
        if st.sidebar.button("classify",key='classify'):
            st.subheader("KNN result")
            model= KNeighborsClassifier(n_neighbors=n_neighbors,leaf_size=leaf_size,weights=weights)
            model.fit(X_trained,y_train)
            y_prediction= model.predict(X_tested)
            acc= accuracy_score(y_test,y_prediction)
            st.write("Accuracy:",acc.round(2))
            st.write("precision_score:",precision_score(y_test,y_prediction,average='micro').round(2))
            st.write("recall_score:",recall_score(y_test,y_prediction,average='micro').round(2))
            if 'confusion matrix' in metrics:
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.subheader('confusion matrix')
                plot_confusion_matrix(model,X_tested,y_test)
                st.pyplot()
            if 'roc_curve' in metrics:
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.subheader('plot_roc_curve')
                plot_roc_curve(model,X_tested,y_test)
                st.pyplot()
            if 'precision_recall_curve' in metrics:
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.subheader('precision_recall_curve')
                plot_roc_curve(model,X_tested,y_test)
                st.pyplot() 
    
    if classifier_name == 'ADABoost':
        st.sidebar.subheader('Model Hyperparmeter')
        n_estimators= st.sidebar.number_input("Number of trees in the forest",100,5000,step=10,key='XGBestimators')
        seed= st.sidebar.number_input("learning rate",1,150,step=1,key='seed')
        metrics= st.sidebar.multiselect("What is the metrics to plot?",('confusion matrix','roc_curve','precision_recall_curve'))
    
        if st.sidebar.button("classify",key='classify'):
            st.subheader("ADABoost result")
      
            model=AdaBoostClassifier(n_estimators=n_estimators,learning_rate=seed)
            model.fit(X_trained,y_train)
            y_prediction= model.predict(X_tested)
            acc= accuracy_score(y_test,y_prediction)
            st.write("Accuracy:",acc.round(2))
            st.write("precision_score:",precision_score(y_test,y_prediction,average='micro').round(2))
            st.write("recall_score:",recall_score(y_test,y_prediction,average='micro').round(2))
            
    
          
    
            if 'confusion matrix' in metrics:
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.subheader('confusion matrix')
                plot_confusion_matrix(model,X_tested,y_test)
                st.pyplot()
            if 'roc_curve' in metrics:
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.subheader('plot_roc_curve')
                plot_roc_curve(model,X_tested,y_test)
                st.pyplot()
            if 'precision_recall_curve' in metrics:
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.subheader('precision_recall_curve')
                plot_roc_curve(model,X_tested,y_test)
                st.pyplot() 
    st.sidebar.subheader('Model Optimization ')
    model_optimizer = st.sidebar.selectbox(
        'Choose Optimizer',
        ('Cross Validation', 'Voting'))
    if model_optimizer == 'Cross Validation':
        cv= st.sidebar.radio("cv",("Kfold","LeaveOneOut"),key='cv')
        algorithim_name = st.sidebar.selectbox(
        'Choose algorithm',
        ('KNN', 'SVM', 'Random Forest','Logistic Regression')
    )
        n_splits= st.sidebar.slider("maximum number of splits",1,30,key='n_splits')
        if st.sidebar.button("optimize",key='opt'):
            if cv=='Kfold':
                kfold= KFold(n_splits=n_splits)
                if algorithim_name =='KNN':
                    score =  cross_val_score(KNeighborsClassifier(n_neighbors=4),X,Y,cv=kfold)
                    st.write("KNN Accuracy:",score.mean()) 
                if algorithim_name =='SVM':
                    score =  cross_val_score(SVC(),X,Y,cv=kfold)
                    st.write("SVM Accuracy:",score.mean())
                if algorithim_name =='Random Forest':
                    score =  cross_val_score(RandomForestClassifier(),X,Y,cv=kfold)
                    st.write("Random Forest Accuracy:",score.mean())
                if algorithim_name =='Logistic Regression':
                    score =  cross_val_score(LogisticRegression(),X,Y,cv=kfold)
                    st.write("Logistic Regression Accuracy:",score.mean())   
  
          
            if cv=='LeaveOneOut':
                loo = LeaveOneOut()
                score =  cross_val_score(SVC(),X,Y,cv=loo)
                st.write("Accuracy:",score.mean())
  
    if model_optimizer == 'Voting':
        voting= st.sidebar.multiselect("What is the algorithms you want to use?",('LogisticRegression','DecisionTreeClassifier','SVC','KNeighborsClassifier','GaussianNB','LinearDiscriminantAnalysis','AdaBoostClassifier','GradientBoostingClassifier','ExtraTreesClassifier'))
        estimator=[]
        if 'LogisticRegression' in voting:
            model1=LogisticRegression()
            estimator.append(model1)
        if 'DecisionTreeClassifier' in voting:
            model2=DecisionTreeClassifier()
            estimator.append(model2)
        if 'SVC' in voting:
            model3=SVC()
            estimator.append(model3)   
        if 'KNeighborsClassifier' in voting:
            model4=KNeighborsClassifier()
            estimator.append(model4)
        if st.sidebar.button("optimize",key='opt'):
            ensemble = VotingClassifier(estimator)
            results = cross_val_score(ensemble, X, Y)
            st.write(results.mean())   
    
    if st.sidebar.checkbox('Prediction Part'):
        st.subheader('Please fill out this form')
        dt= set(X.columns)
        user_input=[]
        
        for i in dt:
            firstname = st.text_input(i,"Type here...")
            user_input.append(firstname)
        if st.button("Prediction",key='algorithm'):
            my_array= np.array([user_input])
            model=AdaBoostClassifier(n_estimators=12)
            model.fit(X_train,y_train)
            y_user_prediction= model.predict(my_array)
            for i in df.target.unique():
                if i == y_user_prediction:
                  st.success('This Data located in this class {}'.format(y_user_prediction))                    
        
    if classifier_name == 'GradientBoosting':
        st.sidebar.subheader('Model Hyperparmeter')
        n_estimators= st.sidebar.number_input("Number of trees in the forest",100,5000,step=10,key='XGBestimators')
        seed= st.sidebar.number_input("learning rate",1,150,step=1,key='seed')
        metrics= st.sidebar.multiselect("What is the metrics to plot?",('confusion matrix','roc_curve','precision_recall_curve'))
    
        if st.sidebar.button("classify",key='classify'):
            st.subheader("gradientBoosting result")
      
            model=GradientBoostingClassifier(n_estimators=n_estimators,learning_rate=seed)
            model.fit(X_trained,y_train)
            y_prediction= model.predict(X_tested)
            acc= accuracy_score(y_test,y_prediction)
            st.write("Accuracy:",acc.round(2))
            st.write("precision_score:",precision_score(y_test,y_prediction,average='micro').round(2))
            st.write("recall_score:",recall_score(y_test,y_prediction,average='micro').round(2))
            
    
          
    
            if 'confusion matrix' in metrics:
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.subheader('confusion matrix')
                plot_confusion_matrix(model,X_tested,y_test)
                st.pyplot()
            if 'roc_curve' in metrics:
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.subheader('plot_roc_curve')
                plot_roc_curve(model,X_tested,y_test)
                st.pyplot()
            if 'precision_recall_curve' in metrics:
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.subheader('precision_recall_curve')
                plot_roc_curve(model,X_tested,y_test)
                st.pyplot() 
                
                
  elif datasetchoice == 'Yes': 
    data_file = st.file_uploader("Upload CSV",type=['csv'])
    st.warning("Note:if you want to do classification make sure your target attributes in your Dataset labeled <target>")
          
    def file_selector(dataset):
      if dataset is not None:
        dataset.seek(0)
        file_details = {"Filename":dataset.name,"FileType":dataset.type,"FileSize":dataset.size}
        st.write(file_details)
        df = pd.read_csv(dataset)
        return df 
    df = file_selector(data_file) 
    st.dataframe(df)
      
      
    st.subheader("Data Explonatory Analysis")
    st.info("This part refers to the various ways to explore your choosen data because When you have a raw data set, it won't provide any insight until you start to organize it") 
    if st.checkbox("Show Dataset"):
      st.dataframe(df)
  
    # Show Columns
    if st.button("Column Names"): 
      st.success("This is the name of your featuresin your dataset")  
      st.write(df.columns)
  
    # Show Shape
    if st.checkbox("Shape of Dataset"):
      st.success("Here you will see number of Rows and Columns and shape of your entire dataset")     
      data_dim = st.radio("Show Dimensions By ",("Rows","Columns"))
      if data_dim == 'Rows':
        st.text("Number of Rows")
        st.write(df.shape[0])
      elif data_dim == 'Columns':
        st.text("Number of Columns")
        st.write(df.shape[1])
      else:
        st.write(df.shape)
  
    # Select Columns
    st.info("If you want to visualize the column you want only for better understanding your dataset?")   
    if st.checkbox("Select Columns To Show"):
      all_columns = df.columns.tolist()
      selected_columns = st.multiselect("Select",all_columns)
      new_df = df[selected_columns]
      st.dataframe(new_df)
  
    # Show Values
    
    if st.button("Value Counts"):
      st.info("This part shows the value count of target in your dataset?")   
      st.text("Value Counts By Target/Class")
      st.write(df.iloc[:,-1].value_counts())
  
  
    # Show Datatypes
    if st.button("Data Types"):
      st.info("This part specifies the type of data your attributes in your Dataset have?")       
      st.write(df.dtypes)
  
  
    # Show Summary
    st.info("Now let 's visualize Statistical Analysis of the chosen dataset,min,max,etc")
    if st.checkbox("Summary"):        
      st.write(df.describe().T)
  
    ## Plot and Visualization
  
    st.subheader("Data Visualization")
    # Correlation
    # Seaborn Plot
    st.info("Now you can perform the graphical representation of information and data. By using visual elements like charts, graphs. Data visualization tools will provide an accessible way to see and understand trends, outliers, and patterns in datasets")
    if st.checkbox("Correlation Plot[Seaborn]"):
      st.set_option('deprecation.showPyplotGlobalUse', False)
      st.write(sns.heatmap(df.corr(),annot=True))
      st.pyplot()
  
  
    if st.checkbox("Pie Plot"):
      all_columns_names = df.columns.tolist()
      if st.button("Generate Pie Plot"):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.success("Generating A Pie Plot")
        st.write(df.iloc[:,-1].value_counts().plot.pie(autopct="%1.1f%%"))
        st.pyplot()
  
    # Count Plot
    if st.checkbox("Plot of Value Counts"):
      st.text("Value Counts By Target")
      all_columns_names = df.columns.tolist()
      primary_col = st.selectbox("Primary Columm to GroupBy",all_columns_names)
      selected_columns_names = st.multiselect("Select Columns",all_columns_names)
      if st.button("Plot"):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.text("Generate Plot")
        if selected_columns_names:
          vc_plot = df.groupby(primary_col)[selected_columns_names].count()
        else:
          vc_plot = df.iloc[:,-1].value_counts()
        st.write(vc_plot.plot(kind="bar"))
        st.pyplot()
  
  
    # Customizable Plot
  
    st.subheader("Customizable Plot")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    try: 
        all_columns_names = df.columns.tolist()
        type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
        selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)
    
        if st.button("Generate Plot"):
          st.set_option('deprecation.showPyplotGlobalUse', False)
          st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))
      
          # Plot By Streamlit
          if type_of_plot == 'area':
            cust_data = df[selected_columns_names]
            st.area_chart(cust_data)
      
          elif type_of_plot == 'bar':
            cust_data = df[selected_columns_names]
            st.bar_chart(cust_data)
      
          elif type_of_plot == 'line':
            cust_data = df[selected_columns_names]
            st.line_chart(cust_data)
      
          # Custom Plot 
          elif type_of_plot:
            cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
            st.write(cust_plot)
            st.pyplot()
      
          if st.button("End of Data Exploration"):
            st.balloons()
        st.subheader("Data Cleaning")
        st.info("Preparing dataset for analysis by removing or modifying data that is incorrect, incomplete, irrelevant, duplicated, or improperly formatted.")
        if st.checkbox("Visualize null value"):
          st.dataframe(df.isnull().sum())
        if st.checkbox("Visualize categorical features"):
  #   st.success("Generating non numeric features in your dataset")
          categorical_feature_columns = list(set(df.columns) - set(df._get_numeric_data().columns))
          dt=df[categorical_feature_columns]
          st.dataframe(dt)
        if st.checkbox("Encoding features"):
  #   st.success("Converting non numeric features into numerical feature in your dataset")
          categorical_feature_columns = list(set(df.columns) - set(df._get_numeric_data().columns))
          label= LabelEncoder()
          for col in df[categorical_feature_columns]:
            df[col]=label.fit_transform(df[col])
          st.dataframe(df)
        
        Y = df.target
        X = df.drop(columns=['target'])
        X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.33, random_state=8) 
        from sklearn.preprocessing import StandardScaler
        sl=StandardScaler()
        X_trained= sl.fit_transform(X_train)
        X_tested= sl.fit_transform(X_test)
        if st.checkbox("Scaling your dataset"):
          st.dataframe(X_trained)
        st.subheader("Feature Engineering")    
        if st.checkbox("Select Column for creation of model"):
  #   st.info("Now extract features from your dataset to improve the performance of machine learning algorithms") 
          all_columns = df.columns.tolist()
          selected_column = st.multiselect("Sele",all_columns)
          new_df = df[selected_column]
      #     st.dataframe(new_df)
          df=new_df  
          categorical_feature_columns = list(set(df.columns) - set(df._get_numeric_data().columns))
          label= LabelEncoder()
          for col in df[categorical_feature_columns]:
            df[col]=label.fit_transform(df[col])
          st.dataframe(df) 
        st.subheader('Data Preparation')
        st.button('Now that we have done selecting the data set let see the summary for what we have done so far')
        st.write("Wrangle data and prepare it for training,Clean that which may require it (remove duplicates, correct errors, deal with missing values, normalization, data type conversion,Randomize data, which erases the effects of the particular order in which we collected and/or otherwise prepared our data,Visualize data to help detect relevant relationships between variables or class imbalances (bias alert!), or perform other exploratory analysis,Split into training and evaluation sets")
        if st.checkbox(" Click here to see next steps"):
          st.write(" 1 step : Choose a Model: Different algorithms are  provides for different tasks; choose the right one")
          st.write(" 2 step : Train the Model: The goal of training is to answer a question or make a prediction correctly as often as possible")
          st.write(" 3 step : Evaluate the Model: Uses some metric or combination of metrics to objective performance of model example accuracy score,confusion metrics,precision call,etc..")
          st.write(" 4 step : Parameter Tuning: This step refers to hyperparameter tuning, which is an artform as opposed to a science,Tune model parameters for improved performance,Simple model hyperparameters may include: number of training steps, learning rate, initialization values and distribution, etc.")
          st.write(" 5 step : Using further (test set) data which have, until this point, been withheld from the model (and for which class labels are known), are used to test the model; a better approximation of how the model will perform in the real world")
      
        st.sidebar.subheader('Choose Classifer')
        classifier_name = st.sidebar.selectbox(
            'Choose classifier',
            ('KNN', 'SVM', 'Random Forest','Logistic Regression','GradientBoosting','ADABoost','Unsupervised Learning(K-MEANS)','Deep Learning')
        )
        label= LabelEncoder()
        for col in df.columns:
          df[col]=label.fit_transform(df[col])
      
      
      
        if classifier_name == 'Unsupervised Learning(K-MEANS)':
          st.sidebar.subheader('Model Hyperparmeter')
          n_clusters= st.sidebar.number_input("number of clusters",2,10,step=1,key='clusters')
          save_option= st.sidebar.radio("Save to file" ,("Yes","No"))
          if st.sidebar.button("classify",key='classify'):    
              sc = StandardScaler()
              X_transformed = sc.fit_transform(df)
              pca = PCA(n_components=2).fit_transform(X_transformed) # calculation Cov matrix is embeded in PCA
              kmeans = KMeans(n_clusters)
              kmeans.fit(pca)
              filename = 'kmeans_model.sav'
              pickle.dump(kmeans, open(filename, 'wb'))
              st.set_option('deprecation.showPyplotGlobalUse', False)
          # plt.figure(figsize=(12,10))
              plt.scatter(pca[:,0],pca[:,1], c=kmeans.labels_, cmap='rainbow')
              plt.title('CLustering Projection');
              st.pyplot()
              if save_option == 'Yes':
                st.markdown(get_binary_file_downloader_html('kmeans_model.sav', 'Model Download'), unsafe_allow_html=True)
                st.success("model successfully saved")
        
        Y = df.target
        X = df.drop(columns=['target'])
        
        
        X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.33, random_state=8)
        
        from sklearn.preprocessing import StandardScaler
        sl=StandardScaler()
        X_trained= sl.fit_transform(X_train)
        X_tested= sl.fit_transform(X_test)
        
        class_name=['yes','no']
        if classifier_name == 'Deep Learning':
            st.sidebar.subheader('Model Hyperparmeter')
            epochs= st.sidebar.slider("number of Epoch",1,30,key='epoch')
            units= st.sidebar.number_input("Dense layers",3,30,step=1,key='units')
            #rate= st.sidebar.slider("Learning Rate",0,5,key='rate')
            activation= st.sidebar.radio("Activation Function",("softmax","sigmoid"),key='activation')
            optimizer= st.sidebar.radio("Optimizer",("rmsprop","Adam"),key='opt')
            
            if st.sidebar.button("classify",key='deep'):
                X_train = X_train / 256.
                model = Sequential()
                model.add(Flatten())
                model.add(Dense(units=units,activation='relu'))
                model.add(Dense(units=units,activation=activation))
                model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
                model.fit(X_train.values, y_train.values, epochs=epochs)
                test_loss, test_acc =model.evaluate(X_test.values,  y_test.values, verbose=2)
                st.write('Deep Learning Model accuracy: ',test_acc)
                
        if classifier_name == 'SVM':
            st.sidebar.subheader('Model Hyperparmeter')
            c= st.sidebar.number_input("c(Reguralization)",0.01,10.0,step=0.01,key='c')
            kernel= st.sidebar.radio("kernel",("linear","rbf"),key='kernel')
            gamma= st.sidebar.radio("gamma(kernel coefficiency",("scale","auto"),key='gamma')
        
            metrics= st.sidebar.multiselect("What is the metrics to plot?",('confusion matrix','roc_curve','precision_recall_curve'))
            save_option= st.sidebar.radio("Save to file" ,("Yes","No"))
            if st.sidebar.button("classify",key='classify'):
                st.subheader("SVM result")
                svcclassifier= SVC(C=c,kernel=kernel,gamma=gamma)
                svcclassifier.fit(X_trained,y_train)
                filename = 'svm_model.sav'
                pickle.dump(svcclassifier, open(filename, 'wb'))
                y_pred= svcclassifier.predict(X_tested)
                acc= accuracy_score(y_test,y_pred)
                st.write("Accuracy:",acc.round(2))
        #     st.write("precision_score:",precision_score(y_test,y_pred,average='micro').round(2))
                st.write("recall_score:",recall_score(y_test,y_pred,average='micro').round(2))
                if save_option == 'Yes':
                    st.markdown(get_binary_file_downloader_html('svm_model.sav', 'Model Download'), unsafe_allow_html=True)
                    st.success("model successfully saved")
                if 'confusion matrix' in metrics:
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.subheader('confusion matrix')
                    plot_confusion_matrix(svcclassifier,X_tested,y_test)
                    st.pyplot()
                if 'roc_curve' in metrics:
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.subheader('plot_roc_curve')
                    plot_roc_curve(svcclassifier,X_tested,y_test,normalize=False)
                    st.pyplot()
                if 'precision_recall_curve' in metrics:
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.subheader('precision_recall_curve')
                    plot_roc_curve(svcclassifier,X_tested,y_test,normalize=False)
                    st.pyplot()
                
                
        
        
        if classifier_name == 'Logistic Regression':
            st.sidebar.subheader('Model Hyperparmeter')
            c= st.sidebar.number_input("c(Reguralization)",0.01,10.0,step=0.01,key='Logistic')
            max_iter= st.sidebar.slider("maximum number of iteration",100,500,key='max_item')
          
        
            metrics= st.sidebar.multiselect("Wht is the metrics to plot?",('confusion matrix','roc_curve','precision_recall_curve'))
            save_option= st.sidebar.radio("Save to file" ,("Yes","No"))
            if st.sidebar.button("classify",key='classify'):
                st.subheader("Logistic Regression result")
                Regression= LogisticRegression(C=c,max_iter=max_iter)
                Regression.fit(X_trained,y_train)
                filename = 'logistic_model.sav'
                pickle.dump( Regression, open(filename, 'wb'))
                y_prediction= Regression.predict(X_tested)
                acc= accuracy_score(y_test,y_prediction)
                st.write("Accuracy:",acc.round(2))
                st.write("precision_score:",precision_score(y_test,y_prediction,average='micro').round(2))
                st.write("recall_score:",recall_score(y_test,y_prediction,average='micro').round(2))
                if save_option == 'Yes':
                  st.markdown(get_binary_file_downloader_html('logistic_model.sav', 'Model Download'), unsafe_allow_html=True)
                st.success("model successfully saved")
                if 'confusion matrix' in metrics:
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.subheader('confusion matrix')
                    plot_confusion_matrix(Regression,X_tested,y_test)
                    st.pyplot()
                if 'roc_curve' in metrics:
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.subheader('plot_roc_curve')
                    plot_roc_curve(Regression,X_tested,y_test)
                    st.pyplot()
                if 'precision_recall_curve' in metrics:
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.subheader('precision_recall_curve')
                    plot_roc_curve(Regression,X_tested,y_test)
                    st.pyplot()
                
                    
        
        if classifier_name == 'Random Forest':
            st.sidebar.subheader('Model Hyperparmeter')
            n_estimators= st.sidebar.number_input("Number of trees in the forest",100,5000,step=10,key='estimators')
            max_depth= st.sidebar.number_input("maximum depth of tree",1,20,step=1,key='max_depth')
            bootstrap= st.sidebar.radio("Boostrap sample when building trees",("True","False"),key='boostrap')
        
        
            metrics= st.sidebar.multiselect("What is the metrics to plot?",('confusion matrix','roc_curve','precision_recall_curve'))
            save_option= st.sidebar.radio("Save to file" ,("Yes","No"))
            if st.sidebar.button("classify",key='classify'):
                st.subheader("Random Forest result")
                model= RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,bootstrap=bootstrap)
                model.fit(X_trained,y_train)
                filename = 'randomforest_model.sav'
                pickle.dump(model, open(filename, 'wb'))
                y_prediction= model.predict(X_tested)
                acc= accuracy_score(y_test,y_prediction)
                st.write("Accuracy:",acc.round(2))
                st.write("precision_score:",precision_score(y_test,y_prediction,average='micro').round(2))
                st.write("recall_score:",recall_score(y_test,y_prediction,average='micro').round(2))
                if save_option == 'Yes':
                  st.markdown(get_binary_file_downloader_html('randomforest_model.sav', 'Model Download'), unsafe_allow_html=True)
                st.success("model successfully")
                if 'confusion matrix' in metrics:
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.subheader('confusion matrix')
                    plot_confusion_matrix(model,X_tested,y_test)
                    st.pyplot()
                if 'roc_curve' in metrics:
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.subheader('plot_roc_curve')
                    plot_roc_curve(model,X_tested,y_test)
                    st.pyplot()
                if 'precision_recall_curve' in metrics:
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.subheader('precision_recall_curve')
                    plot_roc_curve(model,X_tested,y_test)
                    st.pyplot() 
        
        
        if classifier_name == 'KNN':
            st.sidebar.subheader('Model Hyperparmeter')
            n_neighbors= st.sidebar.number_input("Number of n_neighbors",5,30,step=1,key='neighbors')
            leaf_size= st.sidebar.slider("leaf size",30,200,key='leaf')
            weights= st.sidebar.radio("weight function used in prediction",("uniform","distance"),key='weight')
        
        
            metrics= st.sidebar.multiselect("What is the metrics to plot?",('confusion matrix','roc_curve','precision_recall_curve'))
            save_option= st.sidebar.radio("Save to file" ,("Yes","No"))
            if st.sidebar.button("classify",key='classify'):
                st.subheader("KNN result")
                model= KNeighborsClassifier(n_neighbors=n_neighbors,leaf_size=leaf_size,weights=weights)
                model.fit(X_trained,y_train)
                filename = 'knn_model.sav'
                pickle.dump(model, open(filename, 'wb'))
                y_prediction= model.predict(X_tested)
                acc= accuracy_score(y_test,y_prediction)
                st.write("Accuracy:",acc.round(2))
                st.write("precision_score:",precision_score(y_test,y_prediction,average='micro').round(2))
                st.write("recall_score:",recall_score(y_test,y_prediction,average='micro').round(2))
                if save_option == 'Yes':
                  st.markdown(get_binary_file_downloader_html('knn_model.sav', 'Model Download'), unsafe_allow_html=True)
                st.success("model successfully")
                if 'confusion matrix' in metrics:
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.subheader('confusion matrix')
                    plot_confusion_matrix(model,X_tested,y_test)
                    st.pyplot()
                if 'roc_curve' in metrics:
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.subheader('plot_roc_curve')
                    plot_roc_curve(model,X_tested,y_test)
                    st.pyplot()
                if 'precision_recall_curve' in metrics:
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.subheader('precision_recall_curve')
                    plot_roc_curve(model,X_tested,y_test)
                    st.pyplot() 
        if st.sidebar.checkbox('Prediction Part'):
            st.subheader('Please fill out this form')
            dt= set(X.columns)
            user_input=[]
            
            for i in dt:
                firstname = st.text_input(i,"Type here...")
                user_input.append(firstname)
            if st.button("Prediction",key='algorithm'):
                my_array= np.array([user_input])
                model=AdaBoostClassifier(n_estimators=12)
                model.fit(X_train,y_train)
                y_user_prediction= model.predict(my_array)
                for i in df.target.unique():
                    if i == y_user_prediction:
                      st.success('This Data located in this class {}'.format(y_user_prediction))
                      
        if classifier_name == 'ADABoost':
            st.sidebar.subheader('Model Hyperparmeter')
            n_estimators= st.sidebar.number_input("Number of trees in the forest",100,5000,step=10,key='XGBestimators')
            seed= st.sidebar.number_input("learning rate",1,150,step=1,key='seed')
            metrics= st.sidebar.multiselect("What is the metrics to plot?",('confusion matrix','roc_curve','precision_recall_curve'))
        
            if st.sidebar.button("classify",key='classify'):
                st.subheader("ADABoost result")
          
                model=AdaBoostClassifier(n_estimators=n_estimators,learning_rate=seed)
                model.fit(X_trained,y_train)
                y_prediction= model.predict(X_tested)
                acc= accuracy_score(y_test,y_prediction)
                st.write("Accuracy:",acc.round(2))
                st.write("precision_score:",precision_score(y_test,y_prediction,average='micro').round(2))
                st.write("recall_score:",recall_score(y_test,y_prediction,average='micro').round(2))
                #    prediction part    
              
        
              
        
                if 'confusion matrix' in metrics:
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.subheader('confusion matrix')
                    plot_confusion_matrix(model,X_tested,y_test)
                    st.pyplot()
                if 'roc_curve' in metrics:
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.subheader('plot_roc_curve')
                    plot_roc_curve(model,X_tested,y_test)
                    st.pyplot()
                if 'precision_recall_curve' in metrics:
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.subheader('precision_recall_curve')
                    plot_roc_curve(model,X_tested,y_test)
                    st.pyplot() 
                    
        if classifier_name == 'GradientBoosting':
            st.sidebar.subheader('Model Hyperparmeter')
            n_estimators= st.sidebar.number_input("Number of trees in the forest",100,5000,step=10,key='XGBestimators')
            seed= st.sidebar.number_input("learning rate",1,150,step=1,key='seed')
            metrics= st.sidebar.multiselect("What is the metrics to plot?",('confusion matrix','roc_curve','precision_recall_curve'))
        
            if st.sidebar.button("classify",key='classify'):
                st.subheader("gradientBoosting result")
          
                model=GradientBoostingClassifier(n_estimators=n_estimators,learning_rate=seed)
                model.fit(X_trained,y_train)
                y_prediction= model.predict(X_tested)
                acc= accuracy_score(y_test,y_prediction)
                st.write("Accuracy:",acc.round(2))
                st.write("precision_score:",precision_score(y_test,y_prediction,average='micro').round(2))
                st.write("recall_score:",recall_score(y_test,y_prediction,average='micro').round(2))
                
        
              
        
                if 'confusion matrix' in metrics:
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.subheader('confusion matrix')
                    plot_confusion_matrix(model,X_tested,y_test)
                    st.pyplot()
                if 'roc_curve' in metrics:
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.subheader('plot_roc_curve')
                    plot_roc_curve(model,X_tested,y_test)
                    st.pyplot()
                if 'precision_recall_curve' in metrics:
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.subheader('precision_recall_curve')
                    plot_roc_curve(model,X_tested,y_test)
                    st.pyplot() 
                    
        st.sidebar.subheader('Model Optimization ')
        model_optimizer = st.sidebar.selectbox(
            'Choose Optimizer',
            ('Cross Validation', 'Voting'))
        if model_optimizer == 'Cross Validation':
            cv= st.sidebar.radio("cv",("Kfold","LeaveOneOut"),key='cv')
            algorithim_name = st.sidebar.selectbox(
            'Choose algorithm',
            ('KNN', 'SVM', 'Random Forest','Logistic Regression')
        )
            n_splits= st.sidebar.slider("maximum number of splits",1,30,key='n_splits')
            if st.sidebar.button("optimize",key='opt'):
                if cv=='Kfold':
                    kfold= KFold(n_splits=n_splits)
                    if algorithim_name =='KNN':
                        score =  cross_val_score(KNeighborsClassifier(n_neighbors=4),X,Y,cv=kfold)
                        st.write("KNN Accuracy:",score.mean()) 
                    if algorithim_name =='SVM':
                        score =  cross_val_score(SVC(),X,Y,cv=kfold)
                        st.write("SVM Accuracy:",score.mean())
                    if algorithim_name =='Random Forest':
                        score =  cross_val_score(RandomForestClassifier(),X,Y,cv=kfold)
                        st.write("Random Forest Accuracy:",score.mean())
                    if algorithim_name =='Logistic Regression':
                        score =  cross_val_score(LogisticRegression(),X,Y,cv=kfold)
                        st.write("Logistic Regression Accuracy:",score.mean())
              
                if cv=='LeaveOneOut':
                    loo = LeaveOneOut()
                    score =  cross_val_score(SVC(),X,Y,cv=loo)
                    st.write("Accuracy:",score.mean())
      
        if model_optimizer == 'Voting':
            voting= st.sidebar.multiselect("What is the algorithm you want to use?",('LogisticRegression','DecisionTreeClassifier','SVC','KNeighborsClassifier','GaussianNB','LinearDiscriminantAnalysis','AdaBoostClassifier','GradientBoostingClassifier','ExtraTreesClassifier'))
            estimator=[]
            if 'LogisticRegression' in voting:
                model1=LogisticRegression()
                estimator.append(model1)
            if 'DecisionTreeClassifier' in voting:
                model2=DecisionTreeClassifier()
                estimator.append(model2)
            if 'SVC' in voting:
                model3=SVC()
                estimator.append(model3)   
            if 'KNeighborsClassifier' in voting:
                model4=KNeighborsClassifier()
                estimator.append(model4)
            if st.sidebar.button("optimize",key='opt'):
                ensemble = VotingClassifier(estimator)
                results = cross_val_score(ensemble, X, Y)
                st.write(results.mean())       
                
        
        if classifier_name == 'Deep Learning':
            if st.sidebar.button("classify",key='classify'):
                model = Sequential()
                model.add(Flatten())
                model.add(Dense(units=25,activation='relu'))
                model.add(Dense(units=15,activation='softmax'))
                model.compile(loss='sparse_categorical_crossentropy',optimizer='rmsprop', metrics=['accuracy'])
                model.fit(X_train, y_train, epochs=10)
                test_loss, test_acc =model.evaluate(X_test,  y_test, verbose=2)
                st.write('Model accuracy: ',test_acc*100)
    except AttributeError:
            st.write('Please upload dataset')

 
st.title("Multi-Domain machine learning platform for the novice")
 
 
 
menu = ["Home","Login","SignUp"]
choice = st.sidebar.selectbox("Menu",menu)

    
if choice == "Home":
    st.subheader("Home")
    st.image("https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.einfochips.com%2Fblog%2Fwp-content%2Fuploads%2F2018%2F11%2Fhow-to-develop-machine-learning-applications-for-business-featured.jpg&f=1&nofb=1",use_column_width=True)
    st.subheader("About")
    st.markdown("This web platform provides Machine Learning as a service and the most critical existing classiﬁcation algorithms, Natural language processing, Deep Learning and Computer Vision. As a result, you will try many different algorithms and pick and evaluate each performance and select the winner. The algorithms must be appropriate for the problem. This app make machine learning more accessible to the novice.  The platform developed in this project will be open-source")

    st.subheader('Machine Learning Classification –Algorithms')
    st.image('https://data-flair.training/blogs/wp-content/uploads/sites/2/2019/10/Machine-Learning-Classification-Algorithms.jpg')
    st.markdown('Classification is one of the most important aspects of supervised learning. This app, we will support the various classification algorithms like logistic regression, naive bayes, decision trees, random forests and many more. ')
    st.subheader('Natural Language Processing')
    st.image('https://d1m75rqqgidzqn.cloudfront.net/wp-data/2021/01/18170655/an-introduction-to-natural-language-processing-with-python-for-seos-5f3519eeb8368.png')
    st.markdown('Not so long ago, it didn’t really matter if you spoke English, Russian, Hebrew, Italian, Portuguese, etc. —  computers were able to understand only those who had command of the language made up of millions of zeros and ones, and this was a relatively small number of people. But things have changed, and today, with a developing branch of artificial intelligence — natural language processing —  the communication between humans and machines is greatly facilitated. Considering this, NLP can be defined as:')
    st.image('https://res.cloudinary.com/emerline/image/upload/v1600419508/hg5ghmm7qlmbdu0fedjy.jpg')
    st.markdown('Each of the tasks NLP algorithms complete can be quite useful when applied to deal with certain business challenges. To provide you with a clearer picture of what NLP has on offer, we’ve collected the list of key advantages of the technology. ')
    st.image('https://res.cloudinary.com/emerline/image/upload/v1600419645/dfi67pikepaqmsxpnqe9.jpg')
    st.subheader('Computer Vision')
    st.markdown('In 1966, Seymour Papert and Marvin Minsky, two pioneers of artificial intelligence, launched the Summer Vision Project, a two-month, 10-man effort to create a computer system that could identify objects in images.')
    st.image('https://i.pcmag.com/imagery/articles/061CyMCZV6G2sXUmreKHvXS-1..1581020108.jpg')
    st.markdown('To accomplish the task, a computer program had to be able to determine which pixels belonged to which object. This is a problem that the human vision system, powered by our vast knowledge of the world and billions of years of evolution, solves easily. But for computers, whose world consists only of numbers, it is a challenging task.')
    htm_temps ='''<h3>Video Description on how the system works</h3>
    
    
    https://youtu.be/Kdw4g3QCgt8
    
    '''
    st.markdown(htm_temps,unsafe_allow_html=True)
    

elif choice == "Login":
    st.subheader("Login Section")
    st.image("data:image/png;base64,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",use_column_width=True)
    username = st.sidebar.text_input("User Name")
    password = st.sidebar.text_input("Password",type='password')
    if st.sidebar.checkbox("Login"):
        # if password == '12345':
        create_usertable()
        hashed_pswd = make_hashes(password)
 
        result = login_user(username,check_hashes(password,hashed_pswd))
        if result:
 
            st.success("Logged In as {}".format(username))
 
            task = st.selectbox("Task",["Dashboard","Developer","Profiles"])
            if task == "Dashboard":
                st.subheader("Dashboard")
                myApp()
 
            elif task == "Developer":
                st.subheader("About Developer")
                st.write('Name: Tuyizere Diane')
                st.write('Email:dtuyizere17@alustudent.com')
                st.markdown('Data Scientist enthusiast with experience in delivering valuable insights via  Data Analytics and Advanced data-driven methods proficient in building a statistical model with python.')
            elif task == "Profiles":
                st.subheader("User Profiles")
                user_result = view_all_users()
                clean_db = pd.DataFrame(user_result,columns=["Username","Password"])
                st.dataframe(clean_db)
        else:
            st.warning("Incorrect Username/Password")
 
 
 
 
 
elif choice == "SignUp":
    st.subheader("Create New Account")
    new_user = st.text_input("Username")
    new_password = st.text_input("Password",type='password')
 
    if st.button("Signup"):
        create_usertable()
        add_userdata(new_user,make_hashes(new_password))
        st.success("You have successfully created a valid Account")
        st.info("Go to Login Menu to login")
 
