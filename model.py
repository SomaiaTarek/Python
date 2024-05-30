import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

def predict_mobile_price_range(user_input):
    # Load data
    df = pd.read_csv('train.csv')
    
    # Cleaning
    x = df.drop(columns=['price_range','m_dep','fc','talk_time'],axis=1)
    y = df['price_range']
    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Pipeline
    num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median"))])
    x_prepared = num_pipeline.fit_transform(x_train)
    
    # Model
    sv = SVC(kernel='rbf', random_state=23, degree=5, max_iter=-1)
    sv.fit(x_prepared, y_train)
    test_prepared = num_pipeline.transform(user_input)
    y_pred = sv.predict(test_prepared)
    
    # Result
    if y_pred == 0:
        return "Mobile Price Range is 50 : 300 $"
    elif y_pred == 1:
        return "Mobile Price Range is 300 : 600 $"
    elif y_pred == 2:
        return "Mobile Price Range is 600 : 900 $"
    else: 
        return "Mobile Price Range is 900 : 1300 $"