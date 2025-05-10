from flask import Flask, render_template, request, redirect, url_for, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from dotenv import load_dotenv
import os, random
import numpy as np
import imghdr

app = Flask(__name__)
load_dotenv()
app.secret_key = os.getenv('SECRET_KEY')
model = load_model('model/muffin_or_chihuahua.h5')

DIFFICULTY_SETTINGS = {
    'cupcake_cadet': 20,
    'whisker_warrior': 12,
    'baker_boss': 6,
    'muffin_master': 3
}

TEST_IMAGE_PATH = 'static/test'

def get_valid_images(folder_path):
    return [f for f in os.listdir(folder_path)
            if imghdr.what(os.path.join(folder_path, f)) in ['jpeg', 'png']]

ALL_IMAGES = {
    'muffin': get_valid_images(os.path.join(TEST_IMAGE_PATH, 'muffin')),
    'chihuahua': get_valid_images(os.path.join(TEST_IMAGE_PATH, 'chihuahua'))
}

@app.route('/', methods=['GET', 'POST'])
def index():
    session.clear()
    return render_template('index.html', difficulties=DIFFICULTY_SETTINGS)

@app.route('/play', methods=['POST'])
def play():
    difficulty = request.form['difficulty']
    session['difficulty'] = difficulty
    session['question_count'] = 0
    session['user_score'] = 0
    session['ai_score'] = 0
    return redirect(url_for('next_question'))

@app.route('/next_question')
def next_question():
    session['question_count'] += 1
    if session['question_count'] > 10:
        return redirect(url_for('end_game'))

    difficulty = session.get('difficulty', 'cupcake_cadet')
    label = random.choice(['muffin', 'chihuahua'])
    filename = random.choice(ALL_IMAGES[label])

    relative_image_path = f"test/{label}/{filename}"
    full_image_path = os.path.join(TEST_IMAGE_PATH, label, filename)

    try:
        img = load_img(full_image_path, target_size=(150, 150))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)
    except Exception as e:
        print(f"[ERROR] Could not load image: {full_image_path} — {e}")
        return redirect(url_for('next_question'))

    prediction = model.predict(img)
    print(f"Raw prediction: {prediction}")

    true_label = label

    if random.random() < 0.70:
        if prediction.shape[-1] == 1:
            pred_value = prediction[0][0]
            ai_guess = 'muffin' if pred_value > 0.9 else 'chihuahua'
        else:
            pred_index = np.argmax(prediction[0])
            ai_guess = 'muffin' if pred_index == 1 else 'chihuahua'
    else:
        ai_guess = 'chihuahua' if true_label == 'muffin' else 'muffin'


    session['true_label'] = label
    session['ai_guess'] = ai_guess
    session['image_path'] = relative_image_path

    return render_template(
        'game.html',
        image_path=relative_image_path,
        true_label=label,
        ai_guess=ai_guess,
        difficulty=difficulty,
        time_limit=DIFFICULTY_SETTINGS[difficulty],
        user_score=session['user_score'],
        ai_score=session['ai_score']
    )

@app.route('/game', methods=['POST'])
def game():
    user_guess = request.form['user_guess']
    true_label = session.get('true_label')
    ai_guess = session.get('ai_guess')

    user_correct = user_guess == true_label if user_guess != 'timeout' else False
    ai_correct = ai_guess == true_label

    if user_correct:
        session['user_score'] += 1
    if ai_correct:
        session['ai_score'] += 1

    return redirect(url_for('next_question'))

@app.route('/end_game')
def end_game():
    user = session.get('user_score', 0)
    ai = session.get('ai_score', 0)
    return render_template('end_game.html', user_score=user, ai_score=ai)

if __name__ == '__main__':
    app.run(debug=True)
