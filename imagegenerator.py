from flask import Flask, request, render_template, send_file, session
from diffusers import AutoPipelineForImage2Image
from PIL import Image
import torch
import io
import uuid
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
app.secret_key = "GGS"  # For session management
executor = ThreadPoolExecutor(max_workers=4)

# Initialize the AI pipeline for image generation
pipeline = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipeline.enable_sequential_cpu_offload()
pipeline.enable_attention_slicing()

# Image storage per session to maintain state across requests
image_storage = {}

# Define color change function
def change_color(image, target_color, replacement_color, tolerance=40):
    img_np = np.array(image)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    lower_bound = np.array([max(0, c - tolerance) for c in target_color], dtype=np.uint8)
    upper_bound = np.array([min(255, c + tolerance) for c in target_color], dtype=np.uint8)
    mask = cv2.inRange(img_cv, lower_bound, upper_bound)
    img_cv[mask != 0] = replacement_color
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

# Generate image function
def generate_image(prompt, init_image, strength, guidance_scale):
    output = pipeline(prompt=prompt, image=init_image, strength=strength, guidance_scale=guidance_scale, num_inference_steps=20)
    return output.images[0]

# Convert Image to Bytes
def img_to_bytes(img):
    img_io = io.BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    return img_io

@app.route('/')
def home():
    session['session_id'] = str(uuid.uuid4())  # Generate unique session_id
    return render_template('index.html', session_id=session['session_id'])

@app.route('/generate', methods=['POST'])
def generate():
    session_id = session.get('session_id')
    if not session_id:
        return "Session ID is missing.", 400

    image_file = request.files.get('image')
    prompt = request.form.get('prompt')
    strength = float(request.form.get('strength', 0.4))
    guidance_scale = float(request.form.get('guidance_scale', 6.0))

    init_image = Image.open(image_file).convert("RGB") if image_file else Image.new('RGB', (512, 512), (255, 255, 255))
    init_image = init_image.resize((512, 512))

    future = executor.submit(generate_image, prompt, init_image, strength, guidance_scale)
    generated_image = future.result()

    # Store image for session
    image_storage[session_id] = {'base_image': img_to_bytes(generated_image), 'history': [], 'current_step': 0}

    return send_file(image_storage[session_id]['base_image'], mimetype='image/png')

@app.route('/change_color', methods=['POST'])
def color_change():
    session_id = session.get('session_id')
    if session_id not in image_storage:
        return "Session not found.", 400

    target_color = tuple(map(int, request.form.get('target_color', '255,0,0').split(',')))
    replacement_color = tuple(map(int, request.form.get('replacement_color', '0,255,0').split(',')))

    base_image = Image.open(image_storage[session_id]['base_image'])
    updated_image = change_color(base_image, target_color, replacement_color)

    # Update session history
    image_storage[session_id]['history'].append(img_to_bytes(updated_image))
    image_storage[session_id]['current_step'] += 1

    return send_file(image_storage[session_id]['history'][-1], mimetype='image/png')

@app.route('/undo', methods=['POST'])
def undo():
    session_id = session.get('session_id')
    if session_id not in image_storage or image_storage[session_id]['current_step'] == 0:
        return "No more steps to undo.", 400

    image_storage[session_id]['current_step'] -= 1
    img_io = image_storage[session_id]['history'][image_storage[session_id]['current_step']]
    return send_file(img_io, mimetype='image/png')

@app.route('/redo', methods=['POST'])
def redo():
    session_id = session.get('session_id')
    if session_id not in image_storage or image_storage[session_id]['current_step'] >= len(image_storage[session_id]['history']) - 1:
        return "No more steps to redo.", 400

    image_storage[session_id]['current_step'] += 1
    img_io = image_storage[session_id]['history'][image_storage[session_id]['current_step']]
    return send_file(img_io, mimetype='image/png')
    
if __name__ == '__main__':
    app.run(debug=True)
