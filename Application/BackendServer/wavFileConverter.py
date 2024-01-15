import wave

import pyaudio as pyaudio


def convert(input_stream, output_file_path):
    # Set audio parameters
    sample_width = 2  # 16-bit audio
    channels = 2      # Stereo
    frame_rate = 44100

    # Create a PyAudio stream from the input stream
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(sample_width),
                    channels=channels,
                    rate=frame_rate,
                    output=True)
    stream.write(input_stream.read())

    # Save the stream to a WAV file
    stream.stop_stream()
    stream.close()
    p.terminate()

    input_stream.seek(0)  # Reset the stream position
    wave_data = input_stream.read()
    with wave.open(output_file_path, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(frame_rate)
        wf.writeframes(wave_data)
    return wf;