from flask import Flask, render_template, request, redirect, url_for, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os, random
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
model = load_model('model/muffin_or_chihuahua.h5')

DIFFICULTY_SETTINGS = {
    'cupcake_cadet': 20,
    'whisker_warrior': 12,
    'baker_boss': 6,
    'muffin_master': 3
}

TEST_IMAGE_PATH = 'static/test'
ALL_IMAGES = {
    'muffin': os.listdir(os.path.join(TEST_IMAGE_PATH, 'muffin')),
    'chihuahua': os.listdir(os.path.join(TEST_IMAGE_PATH, 'chihuahua'))
}

@app.route('/', methods=['GET', 'POST'])
def index():
    session['question_count'] = 0
    return render_template('index.html', difficulties=DIFFICULTY_SETTINGS)

@app.route('/play', methods=['POST'])
def play():
    difficulty = request.form['difficulty']
    session['difficulty'] = difficulty
    session['question_count'] = 0
    return redirect(url_for('next_question'))

@app.route('/next_question', methods=['GET', 'POST'])
def next_question():
    session['question_count'] += 1

    if session['question_count'] > 10:
        return redirect(url_for('end_game'))

    difficulty = session.get('difficulty', 'cupcake_cadet')
    label = random.choice(['muffin', 'chihuahua'])
    filename = random.choice(ALL_IMAGES[label])

    relative_image_path = f"test/{label}/{filename}"
    full_image_path = os.path.join(TEST_IMAGE_PATH, label, filename)

    img = load_img(full_image_path, target_size=(150, 150))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0][0]
    ai_guess = 'chihuahua' if prediction > 0.5 else 'muffin'

    return render_template(
        'game.html',
        image_path=relative_image_path,
        true_label=label,
        ai_guess=ai_guess,
        difficulty=difficulty,
        time_limit=DIFFICULTY_SETTINGS[difficulty],
    )

@app.route('/game', methods=['POST'])
def game():
    user_guess = request.form['user_guess']
    ai_guess = request.form['ai_guess']
    true_label = request.form['true_label']
    image_path = request.form['image_path']

    user_correct = user_guess == true_label
    ai_correct = ai_guess == true_label

    return redirect(url_for('next_question'))

@app.route('/end_game')
def end_game():
    return render_template('end_game.html')

if __name__ == '__main__':
    app.run(debug=True)
