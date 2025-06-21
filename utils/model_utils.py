import numpy as np
import librosa

# mengikuti konfigurasi file testing
SAMPLE_RATE = 16000
DURATION = 2.0
SAMPLES = int(SAMPLE_RATE * DURATION)
MFCC_N = 13

def relu(x): return np.maximum(0, x)
def softmax(x): return np.exp(x - np.max(x, axis=1, keepdims=True)) / np.sum(np.exp(x - np.max(x, axis=1, keepdims=True)), axis=1, keepdims=True)

def conv2d(x, f):
    h, w = x.shape
    fh, fw = f.shape
    out = np.zeros((h - fh + 1, w - fw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(x[i:i+fh, j:j+fw] * f)
    return out

def extract_features_from_audio_file(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    y = librosa.util.fix_length(data=y, size=SAMPLES)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_N).T

    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    extras = np.array([rms, zcr, centroid, rolloff, bandwidth])

    # debug
    print("\n🟢 Feature Extraction:")
    print(f"- MFCC shape: {mfcc.shape}")
    print(f"- Raw extras: {extras}")

    extras = (extras - np.mean(extras)) / (np.std(extras) + 1e-8)

    # debug
    print(f"- Normalized extras: {extras}")

    return mfcc, extras

def predict(mfcc, extras, model):
    # debug
    print("\n🔵 Prediction Phase:")

    max_len = model['max_len']
    if mfcc.shape[0] < max_len:
        mfcc = np.pad(mfcc, ((0, max_len - mfcc.shape[0]), (0, 0)))

        # debug
        print(f"- MFCC padded to: {mfcc.shape}")

    filters = model['filters']

    # debug
    print(f"- Number of CNN filters: {len(filters)}")

    conv_outputs = [relu(conv2d(mfcc, f)) for f in filters]
    conv_stack = np.stack(conv_outputs, axis=0)
    flat = conv_stack.reshape(1, -1)

    # debug
    print(f"- Conv output shape: {conv_stack.shape}")
    print(f"- Flattened conv shape: {flat.shape}")

    mean = model['extra_mean']
    std = model['extra_std']
    extras = (extras - mean) / (std + 1e-8)

    # debug
    print(f"- Final normalized extras: {extras}")

    W_mlp = model['W_mlp']
    b_mlp = model['b_mlp']
    h_mlp = relu(np.dot(extras.reshape(1, -1), W_mlp) + b_mlp)

    # debug
    print(f"- Hidden MLP output: {h_mlp}")

    z_concat = np.concatenate([flat, h_mlp], axis=1)
    z_concat = z_concat / (np.linalg.norm(z_concat) + 1e-8)

    # debug
    print(f"- z_concat shape: {z_concat.shape}")

    W_out = model['W_out']
    b_out = model['b_out']
    logits = np.dot(z_concat, W_out) + b_out

    temperature = 2.0

    y_out = softmax(logits/temperature)

    # debug
    print(f"- Logits: {logits}")
    print(f"- Softmax output: {y_out}")

    pred_idx = np.argmax(y_out)
    label_encoder = model['label_encoder']
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    confidence = np.max(y_out)

    # debug
    print("\nClass probabilities:", {
        label_encoder.inverse_transform([0])[0]: y_out[0][0],
        label_encoder.inverse_transform([1])[0]: y_out[0][1]
    })
    print(f"Prediction: {pred_label} ({confidence * 100:.2f}% confidence)")

    return pred_label, confidence
