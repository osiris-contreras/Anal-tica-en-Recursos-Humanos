import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer ### para imputación
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
import joblib
from sklearn.preprocessing import StandardScaler ## escalar variables 

####Este archivo contienen funciones utiles a utilizar en diferentes momentos del proyecto

###########Esta función permite ejecutar un archivo  con extensión .sql que contenga varias consultas

def ejecutar_sql (nombre_archivo, cur):
  sql_file=open(nombre_archivo)
  sql_as_string=sql_file.read()
  sql_file.close
  cur.executescript(sql_as_string)
  
 

def sel_variables(modelos,X,y,threshold):
    
    var_names_ac=np.array([])
    for modelo in modelos:
        #modelo=modelos[i]
        modelo.fit(X,y)
        sel = SelectFromModel(modelo, prefit=True,threshold=threshold)
        var_names= modelo.feature_names_in_[sel.get_support()]
        var_names_ac=np.append(var_names_ac, var_names)
        var_names_ac=np.unique(var_names_ac)
    
    return var_names_ac


def medir_modelos(modelos,scoring,X,y,cv):

    metric_modelos=pd.DataFrame()
    for modelo in modelos:
        scores=cross_val_score(modelo,X,y, scoring=scoring, cv=cv )
        pdscores=pd.DataFrame(scores)
        metric_modelos=pd.concat([metric_modelos,pdscores],axis=1)
    
    metric_modelos.columns=["reg_lineal","decision_tree","random_forest","gradient_boosting"]
    return metric_modelos



def preparar_datos(df):     

    #######Cargar y procesar nuevos datos ######       
    #### Cargar modelo y listas
    
    list_dummies=joblib.load("salidas\\list_dummies.pkl")
    var_names=joblib.load("salidas\\var_names.pkl")
    scaler=joblib.load( "salidas\\scaler.pkl") 

    ####Ejecutar funciones de transformaciones
   #Eliminación de variables InfoDate, retirementDate, DateSurvey y SurveyDate ya que no son relevantes dentro del dataframe (informacion 2015-retiros 2016)
    df=df.drop(columns=['InfoDate','retirementDate','DateSurvey','SurveyDate'])

    #Cambiar variables float a integer
    columnas_float=df.select_dtypes(include=['float']).columns
    df[columnas_float]=df[columnas_float].astype(int)
    
    df_dummies=pd.get_dummies(df,columns=list_dummies)
    
    #Eliminacion de  resignationReason ya que se presentan una alta correlación con la variable objetivo Attrition lo cual sesga el desempeño de los modelos.
    df_dummies=df_dummies.drop(columns=['resignationReason_Fired', 'resignationReason_NoRetirement',
       'resignationReason_Others', 'resignationReason_Salary',
       'resignationReason_Stress', 'retirementType_Fired',
       'retirementType_NoRetirement', 'retirementType_Resignation' ])    
    df_dummies = df_dummies.loc[:,~df_dummies.columns.isin(['Attrition','EmployeeID'])]    
    X2 = scaler.transform(df_dummies)
    X = pd.DataFrame(X2,columns=df_dummies.columns)
    X = X[var_names]

    return X