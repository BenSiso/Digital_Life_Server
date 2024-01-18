#!/bin/bash

# Script and configuration parameters
SCRIPT_NAME="SocketServer.py"
PROXY="http://127.0.0.1:7890"
STREAM="False"
CHARACTER="paimon"
MODEL="gpt-3.5-turbo"
OPENAI_API_KEY=""

# Running the script with specified parameters
python %SCRIPT_NAME% --APIKey %OPENAI_API_KEY%  --proxy %PROXY% --stream %STREAM% --model %MODEL% --character %CHARACTER%
