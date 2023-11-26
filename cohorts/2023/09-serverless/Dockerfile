FROM agrigorev/zoomcamp-bees-wasps:v2

# Install the specific TensorFlow Lite wheel and Pillow
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl \
    && pip install Pillow

# Copy the lambda function code into the container
COPY lambda_function.py /lambda_function.py

# Set the command to run when the container starts
CMD ["python", "/lambda_function.py"]
