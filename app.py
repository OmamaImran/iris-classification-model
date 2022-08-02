
from flask import Flask, render_template, request
import numpy as np
import pickle
app=Flask(__name__)

with open("iris.pkl","rb") as f:
    mp= pickle.load(f)

@app.route("/")
def hello():
    return render_template("iris.html")


@app.route("/classify",methods=["POST","GET"])
def classify():
    float_features= [float(x) for x in request.form.values()]
    final= [np.array(float_features)]
    prediction= mp.predict(final)

    if prediction== np.array([[0]]):
        return render_template("iris.html", pred="This is SETOSA specie")

    if prediction== np.array([[1]]):
        return render_template("iris.html", pred="This is VERSICOLOR specie")

    if prediction== np.array([[2]]):
        return render_template("iris.html", pred="This is VIRGINICA specie")

if __name__ == "__main__" :
    app.run(debug=True, threaded=True)