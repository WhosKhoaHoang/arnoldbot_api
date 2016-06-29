from flask import Flask, jsonify, request
from arnoldbot import ArnoldBot

app = Flask(__name__)
ab = ArnoldBot()
#Instantiate ArnoldBot outside of response() so that the training text
#isn't loaded everytime response is called()

@app.route('/talk')
def response():
    data = {}
    msg = request.args.get("msg")
    data["response"] = ab.speak(msg)
    return jsonify(data)

if __name__ == '__main__':
  app.run(debug=False)
