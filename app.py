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
 
""" Common ML Dataset Explorer """
st.title("Machine Learning Tutorial App")
st.subheader("Explorer with Streamlit")
 
html_temp = """
<div style="background-color:#000080;"><p style="color:white;font-size:50px;padding:10px">ML is Awesome</p></div>
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
      rate= st.sidebar.slider("Learning Rate",0,5,step=1,key='rates')
      activation= st.sidebar.radio("Activation Function",("softmax","sigmoid"),key='activations')
      optimizer= st.sidebar.radio("Optimizer",("rmsprop","Adam"),key='opts')
      
      if st.sidebar.button("classify",key='deeps'):
          X_train = X_train / 256.
          model = Sequential()
          model.add(Flatten())
          model.add(Dense(units=units,activation='relu'))
          model.add(Dense(units=units,activation=activation))
          model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizer,learning_rate=rate,metrics=['accuracy'])
          model.fit(X_train.values, y_train.values, epochs=epochs)
          test_loss, test_acc =model.evaluate(X_test.values,  y_test.values, verbose=2)
          st.write('Deep Learning Model accuracy: ',test_acc.round(2))
      
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
          rate= st.sidebar.slider("Learning Rate",0,5,key='rate')
          activation= st.sidebar.radio("Activation Function",("softmax","sigmoid"),key='activation')
          optimizer= st.sidebar.radio("Optimizer",("rmsprop","Adam"),key='opt')
          
          if st.sidebar.button("classify",key='deep'):
              X_train = X_train / 256.
              model = Sequential()
              model.add(Flatten())
              model.add(Dense(units=units,activation='relu'))
              model.add(Dense(units=units,activation=activation))
              model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizer,learning_rate=rate,metrics=['accuracy'])
              model.fit(X_train.values, y_train.values, epochs=epochs)
              test_loss, test_acc =model.evaluate(X_test.values,  y_test.values, verbose=2)
              st.write('Deep Learning Model accuracy: ',test_acc.round(2))
              
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
