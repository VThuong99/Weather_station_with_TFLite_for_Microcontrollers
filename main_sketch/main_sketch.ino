
//Note: Set to 1 if you want to check whether the model can forecast the snow
#define DEBUG_RAIN 1

#include "_model.h"

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>


// TensorFlow Lite for Microcontroller global variables
const tflite::Model* tflu_model            = nullptr;
tflite::MicroInterpreter* tflu_interpreter = nullptr;
TfLiteTensor* tflu_i_tensor                = nullptr;
TfLiteTensor* tflu_o_tensor                = nullptr;
tflite::MicroErrorReporter tflu_error;

constexpr int tensor_arena_size = 4 * 1024;
byte tensor_arena[tensor_arena_size] __attribute__((aligned(16)));

void tflu_initialization()
{
  Serial.println("TFLu initialization - start");

  // Load the TFLITE model
  tflu_model = tflite::GetModel(rain_forecast_model_tflite);
  if (tflu_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print(tflu_model->version());
    Serial.println("");
    Serial.print(TFLITE_SCHEMA_VERSION);
    Serial.println("");
    while(1);
  }

  tflite::AllOpsResolver tflu_ops_resolver;

  // Initialize the TFLu interpreter
  tflu_interpreter = new tflite::MicroInterpreter(tflu_model, tflu_ops_resolver, tensor_arena, tensor_arena_size, &tflu_error);

  // Allocate TFLu internal memory
  tflu_interpreter->AllocateTensors();

  // Get the pointers for the input and output tensors
  tflu_i_tensor = tflu_interpreter->input(0);
  tflu_o_tensor = tflu_interpreter->output(0);

  Serial.println("TFLu initialization - completed");
}

#include <DHT.h>

const int gpio_pin_dht_pin = 28;

DHT dht(gpio_pin_dht_pin, DHT22);

#define READ_TEMPERATURE() dht.readTemperature()
#define READ_HUMIDITY()    dht.readHumidity()

void setup() {
  Serial.begin(9600);
  while (!Serial);

  // Initialize the DHT sensor
  dht.begin();

  // Waiting for the peripheral for being ready
  delay(2000);

  tflu_initialization();
}

void loop() {
  float t = 0.0f;
  float h = 0.0f;

#if DEBUG_RAIN == 1
  t = 22.0f;
  h = 57.0f;
#else
  for(int i = 0; i < 5; ++i) {
    t += READ_TEMPERATURE();
    h += READ_HUMIDITY();

    delay(2000);
  }

  // Take the average
  t /= 5.0;
  h /= 5.0;
  // t = READ_TEMPERATURE();
  // h = READ_HUMIDITY();
#endif

  Serial.print("Temperature = ");
  Serial.print(t, 2);
  Serial.println(" °C");
  Serial.print("Humidity = ");
  Serial.print(h, 2);
  Serial.println(" %");

  // Initialize the input tensor
  tflu_i_tensor->data.f[0] = t;
  tflu_i_tensor->data.f[1] = h;

  // Run inference
  TfLiteStatus invoke_status = tflu_interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Error invoking the TFLu interpreter");
    return;
  }

  float out[5];
  out[0] = tflu_o_tensor->data.f[0];
  out[1] = tflu_o_tensor->data.f[1];
  out[2] = tflu_o_tensor->data.f[2];
  out[3] = tflu_o_tensor->data.f[3];
  out[4] = tflu_o_tensor->data.f[4];

  Serial.print("Softmax output array: ");
  Serial.print(out[0]);
  Serial.print(" ");
  Serial.print(out[1]);
  Serial.print(" ");
  Serial.print(out[2]);
  Serial.print(" ");
  Serial.print(out[3]);
  Serial.print(" ");
  Serial.println(out[4]);

  float max = out[0];
  int index = 0;
  for(int i = 0; i<5; i++){
    if(out[i] > max){
      index = i;
    }
  }
  Serial.print("The sky is: ");
  if (index == 0) {
    Serial.println("Clear");
  } else if (index== 1) {
    Serial.println("Partly cloudy");
  } else if (index== 2) {
    Serial.println("Mostly cloudy");
  } else if (index== 3) {
    Serial.println("Overcast");
  } else {
    Serial.println("Foggy");
  }

  // We should have a delay of 1 hour but, for practical reasons, we have reduced it to 2 seconds
  delay(2000);
}