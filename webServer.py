from flask import Flask, request
import logging
import os
import socket

app = Flask(__name__)


@app.route('/audios', methods=['POST'])
def upload_audio():
    try:
        audio_file = request.files['audio']
        if audio_file and audio_file.filename.endswith('.wav'):
            # Save the uploaded file to a temporary location
            temp_filename = 'temp_uploaded_audio.wav'
            audio_file.save(temp_filename)

            # Forward the file to the socket server
            forward_to_socket(temp_filename)

            # Delete the temporary file
            os.remove(temp_filename)

            return 'File uploaded successfully and forwarded to socket server!\n'
        else:
            return 'Invalid file format. Please upload a .wav file.\n', 400
    except Exception as e:
        return f'Error: {str(e)}\n', 500

def forward_to_socket(filename):
    # Create a socket connection
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Set the server address and port number
    server_address = ("127.0.0.1", 38438)

    try:
        # connect to the server
        client_socket.connect(server_address)
        logging.basicConfig(level=logging.INFO)
        logging.info('Successfully connect to the server!')

        # The character name sent by the server
        character_name = client_socket.recv(1024).decode()
        logging.info('Receive the character name: %s', character_name)

        # Get the audio file path entered by the user
        audio_file_path = filename

        # Send audio file
        try:
            with open(audio_file_path, 'rb') as file:
                while True:
                   # Reading a fixed -size data block
                    audio_data = file.read(1024)

                    if not audio_data:
                        break # File reads complete

                    client_socket.sendall(audio_data)
                    # Waiting for the confirmation signal of the server
                    ack = client_socket.recv(2)
                    if ack != b'sb':
                        break
                        

                # Send the end of the end
                client_socket.sendall(b'?!')
                logging.info('The audio file is complete!')

        except FileNotFoundError:
            logging.error('The file is not found, please check whether the path is correct.')
            

        # Returned by the server
        response = b''
        while True:
            data = client_socket.recv(1024)
            if data == b'stream_finished':
                logging.info("Receive the notification of the stream dialogue.")
                break
            response += data
        logging.info('Receive the complete voice')

    except ConnectionRefusedError:
        logging.info('Cant connect to the server!')

    finally:
        logging.info('Closing socket connection')
        client_socket.close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True, port=8000)