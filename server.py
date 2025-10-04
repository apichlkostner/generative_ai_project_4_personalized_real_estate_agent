from flask import Flask, render_template_string, send_file
from llm import get_results 

app = Flask(__name__)

def compute_results(size, priorities, amenities, transport, urban, style):
    answers = [size, priorities, amenities, transport, urban, style]
    chosen_images, descriptions = get_results(answers)

    return chosen_images, descriptions


@app.route("/", methods=["GET"])
def form():
    html = """
    <html>
    <head>
        <title>Find Your Home</title>
        <style>
            body { font-family: Arial, sans-serif; background: #f8f9fa; padding: 20px; }
            h1 { text-align: center; }
            form { max-width: 600px; margin: auto; background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
            label { display: block; margin-top: 15px; font-weight: bold; }
            input, textarea { width: 100%; padding: 8px; margin-top: 5px; border: 1px solid #ccc; border-radius: 6px; }
            button { margin-top: 20px; padding: 10px 20px; background: #007BFF; color: white; border: none; border-radius: 6px; cursor: pointer; }
            button:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <h1>Tell Us About Your Dream Home</h1>
        <form action="/results" method="post">
            <label>How big do you want your house to be?</label>
            <input type="text" name="size" required>

            <label>What are 3 most important things for you in choosing this property?</label>
            <textarea name="priorities" required></textarea>

            <label>Which amenities would you like?</label>
            <textarea name="amenities" required></textarea>

            <label>Which transportation options are important to you?</label>
            <textarea name="transport" required></textarea>

            <label>How urban do you want your neighborhood to be?</label>
            <input type="text" name="urban" required>

            <label>How should your house look like?</label>
            <textarea name="style" required></textarea>

            <button type="submit">Find Recommendations</button>
        </form>
    </body>
    </html>
    """
    return render_template_string(html)


from flask import request

@app.route("/results", methods=["POST"])
def results():
    global images
    # Collect answers
    size = request.form["size"]
    priorities = request.form["priorities"]
    amenities = request.form["amenities"]
    transport = request.form["transport"]
    urban = request.form["urban"]
    style = request.form["style"]

    # Run computation
    images, descriptions = compute_results(size, priorities, amenities, transport, urban, style)
    pairs = list(zip(images, descriptions))

    # Show results
    html = """
    <html>
    <head>
        <title>Recommendations</title>
        <style>
            body { font-family: Arial, sans-serif; background: #f8f9fa; padding: 20px; }
            h1 { text-align: center; margin-bottom: 40px; }
            .card { display: flex; align-items: flex-start; background: white; border-radius: 12px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 20px auto; max-width: 800px; overflow: hidden; }
            .card img { width: 300px; border-right: 1px solid #ddd; }
            .card-content { padding: 20px; flex: 1; }
        </style>
    </head>
    <body>
        <h1>Your Personalized Real Estate Recommendations</h1>
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
    app.run(debug=True)
