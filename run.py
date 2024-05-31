import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.io.wavfile import write, read
from scipy.signal import spectrogram, istft
from skimage.transform import resize

def create_tone(frequency, duration, samplerate):
    t = np.linspace(0., duration, int(samplerate * duration))
    return 0.5 * np.sin(2 * np.pi * frequency * t)

def embed_message(audio, message, samplerate, modulation_strength=20, gain_db=20):
    audio = audio.astype(np.float32)

    audio *= 10**(gain_db / 20)

    fig, ax = plt.subplots()
    ax.axis('off')
    plt.text(0.5, 0.5, message, fontsize=30, ha='center', transform=ax.transAxes, color='black')
    fig.canvas.draw()

    buf = fig.canvas.buffer_rgba()
    image = np.frombuffer(buf, dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)

    image_gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

    freqs, times, Sxx = spectrogram(audio, samplerate)
    image_resized = resize(image_gray, (Sxx.shape[0], Sxx.shape[1]), anti_aliasing=True)

    image_min, image_max = np.min(image_resized), np.max(image_resized)
    if image_max - image_min > 0:
        image_resized = (image_resized - image_min) / (image_max - image_min)
    else:
        image_resized = np.zeros_like(image_resized)

    image_resized = np.flipud(image_resized)

    Sxx_mod = Sxx * (1 + image_resized * modulation_strength)

    _, audio_mod = istft(Sxx_mod, samplerate)

    audio_mod *= 10**(gain_db / 20)

    return audio_mod, Sxx_mod, freqs, times

def main():
    parser = argparse.ArgumentParser(description="Embed a message into an audio file's spectrogram")
    parser.add_argument("input_file", type=str, help="Path to the input audio file (WAV format)")
    parser.add_argument("message", type=str, help="Message to embed into the audio spectrogram")
    parser.add_argument("output_file", type=str, help="Path to save the output audio file (WAV format)")
    parser.add_argument("--modulation_strength", type=float, default=20, help="Strength of the modulation for embedding the message")
    parser.add_argument("--gain_db", type=float, default=20, help="Gain in dB to apply to the audio signal")

    args = parser.parse_args()

    samplerate, audio = read(args.input_file)

    audio_with_message, Sxx_mod, freqs, times = embed_message(audio, args.message, samplerate, args.modulation_strength, args.gain_db)

    write(args.output_file, samplerate, audio_with_message.astype(np.float32))

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(times, freqs, 10 * np.log10(Sxx_mod), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.title('Modified Spectrogram with Embedded Message')
    plt.colorbar(label='Intensity [dB]')
    plt.show()

if __name__ == "__main__":
    main()
