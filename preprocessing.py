import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder , OrdinalEncoder
from sklearn.compose import ColumnTransformer

numerical = ['CreditScore' , 'Age' ,'Tenure' ,'Balance' , 'NumOfProducts' ,'EstimatedSalary' , 'Satisfaction Score' ,'Point Earned'
    
]

labelcat = ['Gender']

onehotcat =['Geography' , 'Card Type']

def preprocess():
    return ColumnTransformer(
        transformers = [(
            'encoder' ,
            OneHotEncoder(handle_unknown = 'ignore'),
            onehotcat ) ,
                        
            ('encoder2', OrdinalEncoder(), labelcat),
            
            ('scaler' , StandardScaler(), numerical)
        ] , remainder = 'passthrough'
    )