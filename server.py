from flask import Flask, render_template_string, send_file
from llm import get_results 

app = Flask(__name__)

images = []
descriptions = []

def compute_results():
    global images, descriptions
    images, descriptions = get_results()

@app.route("/")
def index():
    global images, descriptions
    pairs = list(zip(images, descriptions))  # pair up in Python
    html = """
    <html>
    <head>
        <title>Results</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f8f9fa;
                margin: 0;
                padding: 20px;
            }
            h1 {
                text-align: center;
                margin-bottom: 40px;
            }
            .card {
                display: flex;
                align-items: flex-start;
                background: white;
                border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                margin: 20px auto;
                max-width: 800px;
                overflow: hidden;
            }
            .card img {
                width: 300px;
                height: auto;
                border-right: 1px solid #ddd;
            }
            .card-content {
                padding: 20px;
                flex: 1;
            }
            .card-content p {
                margin: 0;
                font-size: 16px;
                line-height: 1.4;
                color: #333;
            }
        </style>
    </head>
    <body>
        <h1>Real Estate Recommendations</h1>
        {% for img, desc in pairs %}
            <div class="card">
                <img src="/image/{{ loop.index0 }}" alt="Recommendation {{ loop.index }}">
                <div class="card-content">
                    <p>{{ desc }}</p>
                </div>
            </div>
        {% endfor %}
    </body>
    </html>
    """
    return render_template_string(html, pairs=pairs)

@app.route("/image/<int:num>")
def image(num):
    global images
    if 0 <= num < len(images):
        return send_file(images[num], mimetype="image/png")
    else:
        return "No such image", 404

if __name__ == "__main__":
    compute_results()
    app.run(debug=True)
