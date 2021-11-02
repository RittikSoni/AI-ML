import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns




df = pd.read_csv('dementia3.csv')

# HEADINGS
st.title('Dementia Prediction')
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())
st.bar_chart(df)

# X AND Y DATA
x = df.drop(['Outcome'], axis = 1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)


# FUNCTION
def user_report():
  Age = st.sidebar.slider('Age', 77,100, 80 )
  Sex = st.sidebar.slider('Sex (0=male,female=1)', 0,1, 1 )
  Education = st.sidebar.slider('Education', 6,21, 19 )
  TBI = st.sidebar.slider('Age at 1st TBI', 0,89, 56 )
  Reagan = st.sidebar.slider('Nia Reagan', 0,3, 1 )
  Cerad = st.sidebar.slider('Cerad Score', 0,3, 1 )
  TBILoc = st.sidebar.slider('Num TBI Loc', 0,3, 1 )
  Braak = st.sidebar.slider('Braak Score', 0,6, 3 )

  user_report_data = {
      'Age':Age,
      'Sex':Sex,
      'Education':Education,
      'TBI':TBI,
      'Reagan':Reagan,
      'Cerad':Cerad,
      'TBILoc':TBILoc,
      'Braak':Braak
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data




# PATIENT DATA
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)




# MODEL
rf  = RandomForestClassifier()
rf.fit(x_train, y_train)
user_result = rf.predict(user_data)



# VISUALISATIONS
st.title('Visualised Patient Report')



# COLOR FUNCTION
if user_result[0]==0:
  color = 'blue'
else:
  color = 'red'


# Braak vs Age
st.header('Braak vs Age count Graph (Others vs Yours)')
fig_preg = plt.figure()
ax1 = sns.scatterplot(x = 'Age', y = 'Braak', data = df, hue = 'Outcome', palette = 'Greens')
ax2 = sns.scatterplot(x = user_data['Age'], y = user_data['Braak'], s = 150, color = color)
plt.xticks(np.arange(70,100,3))
plt.yticks(np.arange(0,20,2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_preg)



# Braak vs Sex
st.header('Braak vs Sex Value Graph (Others vs Yours)')
fig_Sex = plt.figure()
ax3 = sns.scatterplot(x = 'Braak', y = 'Sex', data = df, hue = 'Outcome' , palette='magma')
ax4 = sns.scatterplot(x = user_data['Braak'], y = user_data['Sex'], s = 150, color = color)
plt.xticks(np.arange(0,10,1))
plt.yticks(np.arange(0,2,1))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_Sex)



# Braak vs Education
st.header('Braak vs Education Value Graph (Others vs Yours)')
fig_Education = plt.figure()
ax5 = sns.scatterplot(x = 'Braak', y = 'Education', data = df, hue = 'Outcome', palette='Reds')
ax6 = sns.scatterplot(x = user_data['Braak'], y = user_data['Education'], s = 150, color = color)
plt.xticks(np.arange(0,10,1))
plt.yticks(np.arange(0,50,5))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_Education)


# Braak vs TBI
st.header('Braak vs TBI Value Graph (Others vs Yours)')
fig_st = plt.figure()
ax7 = sns.scatterplot(x = 'Braak', y = 'TBI', data = df, hue = 'Outcome', palette='Blues')
ax8 = sns.scatterplot(x = user_data['Braak'], y = user_data['TBI'], s = 150, color = color)
plt.xticks(np.arange(0,10,1))
plt.yticks(np.arange(0,110,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_st)


# Braak vs Reagan
st.header('Reagan Value Graph (Others vs Yours)')
fig_i = plt.figure()
ax9 = sns.scatterplot(x = 'Braak', y = 'Reagan', data = df, hue = 'Outcome', palette='rocket')
ax10 = sns.scatterplot(x = user_data['Braak'], y = user_data['Reagan'], s = 150, color = color)
plt.xticks(np.arange(0,10,1))
plt.yticks(np.arange(0,7,1))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_i)


# Braak vs Cerad
st.header('Braak vs Cerad Value Graph (Others vs Yours)')
fig_Cerad = plt.figure()
ax11 = sns.scatterplot(x = 'Braak', y = 'Cerad', data = df, hue = 'Outcome', palette='rainbow')
ax12 = sns.scatterplot(x = user_data['Braak'], y = user_data['Cerad'], s = 150, color = color)
plt.xticks(np.arange(0,10,1))
plt.yticks(np.arange(0,7,1))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_Cerad)


# Braak vs TBILoc
st.header('TBILoc Value Graph (Others vs Yours)')
fig_TBILoc = plt.figure()
ax13 = sns.scatterplot(x = 'Braak', y = 'TBILoc', data = df, hue = 'Outcome', palette='YlOrBr')
ax14 = sns.scatterplot(x = user_data['Braak'], y = user_data['TBILoc'], s = 150, color = color)
plt.xticks(np.arange(0,10,1))
plt.yticks(np.arange(0,7,1))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_TBILoc)



# OUTPUT
st.subheader('Your Report: ')
output=''
if user_result[0]==0:
  output = 'You are healthy (You do not have any Dementia)'
else:
  output = 'Your symptoms matches with Dementia, Please consult to a Doctor'
st.title(output)
st.subheader('Accuracy: ')
st.write(str(accuracy_score(y_test, rf.predict(x_test))*100)+'%')
