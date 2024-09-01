from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from cnn.predecir import predict


app = Flask(__name__)

CORS(app)

@app.route("/prediccion", methods = ['POST'])
def prediccion():

    print(request.files)

    if 'image' != request.files:
          return jsonify({'error': 'No se ha enviado ninguna imagen'}), 400
    
    img = request.files['image']

    if img.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400
    
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}

    if '.' not in img.filename or img.filename.split('.')[-1].lower() not in allowed_extensions:
        return jsonify({'error': 'Formato de imagen no válido'}), 400
    
    img_bytes = img.read()

    img_io = BytesIO(img_bytes)

    prediccion = predict(img_io)

    return jsonify ({'clasificacion':prediccion}), 200


if __name__ == "__main__":
    app.run(debug=True, port=8000)