from flask import Flask, jsonify, request
from arnoldbot import ArnoldBot

app = Flask(__name__)
ab = ArnoldBot()

@app.route('/talk')
def response():
    data = {}
    msg = request.args.get("msg")
    data["response"] = ab.speak(msg)
    return jsonify(data)

if __name__ == '__main__':
  app.run(debug="True")
