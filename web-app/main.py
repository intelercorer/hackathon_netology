import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, send_file
import joblib
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)

model = joblib.load('../predictor/SVMScaled.joblib')
scaler = joblib.load('../predictor/scaler.joblib')

prediction_df = None

def generate_recommendations(predictions):
    recommendations = []
    for pred in predictions:
        if pred == 1:
            recommendations.append("Клиент находится в зоне риска.")
        else:
            recommendations.append("Клиент не в зоне риска.")
    return recommendations
#придумать больше рекомендаций в зависимости от вероятности*

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    global prediction_df
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join('uploads', file.filename)
            file.save(filepath)

            df = pd.read_csv(filepath)
            customer_ids = df['CustomerId']

            for column in ['Surname', 'Geography', 'Gender']:
                try:
                    df[column] = df[column].astype('category').cat.codes
                except:
                    pass

            try:
                df['ProdRiskCat'] = np.where(df['NumOfProducts'] <= 2, 1, 0)
            except:
                pass

            features = df.drop(columns=['RowNumber', 'CustomerId', 'Tenure', 'EstimatedSalary', 'Surname', 'NumOfProducts', 'HasCrCard', 'Exited'])  # Modify as per your dataset
            features = scaler.transform(features)

            predictions = model.predict(features)
            probabilities = model.predict_proba(features)[:, 1]

            prediction_df = pd.DataFrame({
                'CustomerId': df['CustomerId'],
                'Probability': probabilities.round(2),
                'Recommendation': generate_recommendations(predictions)
            })

            prediction_df = prediction_df.sort_values(by=['Probability', 'CustomerId'], ascending=False)

            prediction_table = prediction_df.to_dict(orient='records')

            plt.figure()
            plt.pie(np.unique(predictions, return_counts= True)[1], labels=['Нет риска', 'Группа риска'], shadow=True, autopct='%1.1f%%', startangle=180)
            plt.title('Диаграмма предсказаний')
            plot_path = os.path.join('static', 'plot.png')
            plt.savefig(plot_path)

            return render_template('index.html', predictions=prediction_table, visualization=True)

    return redirect(url_for('index'))


@app.route('/download', methods=['GET'])
def download_file():
    if prediction_df is not None:
        csv_path = os.path.join('uploads', 'predictions.csv')
        prediction_df.to_csv(csv_path, index=False)
        return send_file(csv_path, as_attachment=True)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=False)
