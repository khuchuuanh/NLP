from flask import Flask, request, jsonify,render_template
from werkzeug.middleware.proxy_fix import ProxyFix
from config import *
from lib import *
from data_process import *
from predict import *
import numpy as np
from config import *
from data_process import *
from flask_cors import CORS



app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
model = load_model("Bi_GRU.h5")
@app.route('/')
def home():
	return render_template('index.html')

@app.route('/generate_text', methods=['POST'])
def generate_new_text():
    input_text = request.form['input_text']
    num_words = int(request.form['num_words'])
    output_text = generate_text(input_text, num_words, model)
    return render_template('result.html', input_text=input_text, output_text=output_text)

if __name__ == '__main__':
    app.run(debug=True)

    