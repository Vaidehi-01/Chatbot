from flask import Flask, render_template, jsonify, request
from recommendations import (
    find_similar_books_by_author,
    find_similar_books_by_publisher,
    get_id_from_partial_name,
    recommend_books_by_average_rating,
    get_index_from_name,
)

app = Flask(__name__)


@app.route("/")
def hello():
    return render_template("index.html")


@app.route("/recommendations/<case>", methods=["POST"])
def recommendations(case):
    if request.method == "POST":
        query = request.json.get("query")
        case = str(case)
        case = case.strip()
        if case == "1":
            result = find_similar_books_by_author(query)
            return jsonify(result)
        elif case == "2":
            result = find_similar_books_by_publisher(query)
            return jsonify(result)
        elif case == "3":
            result = get_id_from_partial_name(query)
            return jsonify(result)
        elif case == "4":
            result = recommend_books_by_average_rating(num_recommendations=int(query))
            return jsonify(result)
        elif case == "5":
            result = get_index_from_name(query)
            return jsonify(result)
        else:
            return jsonify("Invalid Input or invalid case number")
