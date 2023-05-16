from flask import Flask, render_template, request, jsonify
from requete import *
app = Flask(__name__)
@app.route('/')
def index():
    if request.method == 'GET':
        data = search_init()
    return render_template('main.html',data=data)

@app.route('/filtrage_lieu', methods=['GET', 'POST'])
def filtrage_lieu():
    if request.method == 'GET':
        lieu = request.args.get('lieu')
        data = search_lieu(lieu)
        return render_template('search_lieu.html', data=data)
    
@app.route('/filtrage_object', methods=['GET', 'POST'])
def filtrage_object():
    if request.method == 'GET':
        object = request.args.get('object')
        data = search_object(object)
        return render_template('search_object.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)