from flask import Flask, request, render_template
from recommend import recommend_learning_path

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    recommendation = ""
    if request.method == "POST":
        input_data = [
            int(request.form["videos"]),
            int(request.form["quiz_attempts"]),
            int(request.form["quiz_score"]),
            int(request.form["forum"]),
            int(request.form["assignments"]),
            int(request.form["exam"]),
            int(request.form["feedback"])
        ]
        recommendation = recommend_learning_path(input_data)

    return render_template("index.html", recommendation=recommendation)

if __name__ == "__main__":
    app.run(debug=True)
