from flask import Flask, render_template
import csv

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get-status")
def get_status():
    try:
        with open("../results.csv", newline="\n") as csvfile:
            for row in reversed(list(csv.reader(csvfile))):
                return {
                    "timestamp": row[0],
                    "total": int(row[1]) + int(row[2]),
                    "up": row[1],
                    "down": row[2],
                }
    except FileNotFoundError:
        return {"timestamp": "--", "total": 0, "up": 0, "down": 0}


if __name__ == "__main__":
    app.run(debug=True)
