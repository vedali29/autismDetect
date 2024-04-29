from flask import Flask, request
from flask_cors import CORS

app=Flask('__name__')
app = Flask(__name__)
CORS(app,origins='*') 


@app.route('/', methods=['post'])
def index():
    print('this is index')         
    data = request.json
    # pred=l
    # y_pred = dct.predict(x_test)
    
    return data


if __name__ == '__main__':
    app.run(debug=True)
    
    