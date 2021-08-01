from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


# @app.route("/get-status")
# def get_status():
#     try:
#         with open("log.txt", newline="\n") as log:
#             for row in log:
#                 return {}
#     except FileNotFoundError:
#         return {"timestamp": "--", "total": 0, "up": 0, "down": 0}


if __name__ == "__main__":
    app.run(debug=True)
