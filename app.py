from flask import Flask, render_template
from recommendations import 
    get_index_from_name,
    get_id_from_partial_name,
    find_similar_books_by_author,
    find_similar_books_by_publisher,
    print_similar_books,
    recommend_books_by_average_rating,

get
app = Flask(__name__)


@app.route("/")
def hello():
    return render_template("index.html")



