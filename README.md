# Weather station with TFLite for Microcontrollers
## Description
This project is used for weather forecasting based on TinyML.
## Hardware
- Raspberry pi pico.
- DHT22.

![Screenshot 2024-04-06 155615](https://github.com/VThuong99/Weather_station_with_TFLite_for_Microcontrollers/assets/146172312/f3e2861c-7006-47f4-bbce-05248ccd2bcf)

## Python part
This has been designed using Tensorflow and Keras with several layers. This network has been trained on [Instanbul weather dataset](https://www.kaggle.com/datasets/vonline9/weather-istanbul-data-20092019). Then this model has been exported into a none quantized tflite model. After that I have used [xxd tool](https://cygwin.com/packages/summary/xxd.html) to create a .h file containing an hexadecimal array with the neural network's parameters.
## Arduino part
I have used Arduino IDE to deploy the model into the Raspberry pi pico. I deploy my model based on the guide of [TensorFlow Lite for microcontroller]( https://www.tensorflow.org/lite/microcontrollers/get_started_low_level?hl=vi).
Here is the link to the library I used: https://downloads.arduino.cc/libraries/github.com/bcmi-labs/Arduino_TensorFlowLite-2.4.0-ALPHA.zip.

Here is the result of the inferences:

![Screenshot 2024-04-06 151903](https://github.com/VThuong99/Weather_station_with_TFLite_for_Microcontrollers/assets/146172312/be2bd286-0f9c-4f04-954a-c92ec38ab8c7)
