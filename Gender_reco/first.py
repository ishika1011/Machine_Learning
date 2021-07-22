from flask import Flask,render_template,url_for,request
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

@app.route('/')

def index():
    return render_template('index.html')

#label encoder,one_hot_encoder
@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv("names_dataset.csv")
    df_x = df.name
    #df_y = df.sex
    
    corpus = df_x
    cv= CountVectorizer()
    x = cv.fit_transform(corpus)    
    print(x)
    clf=joblib.load(open("models/naivebayesgendermodel.pkl","rb"))
    
    if request.method == 'POST':
        pname = request.form['pname']
        data=[pname]
        vect = cv.transform(data).toarray()
        pred = clf.predict(vect)
    return render_template('results.html',prediction=pred,name = pname.upper())

if __name__ =='__main__':
    app.run(debug=True)
    
###if __name__ =='__main__':
 ###   return render_template('index.html')


