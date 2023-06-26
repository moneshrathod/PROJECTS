// Include the IMU Library
#include "Arduino_BMI270_BMM150.h"
UART mySerial(digitalPinToPinName(3), digitalPinToPinName(2), NC, NC); 
//Include the BLE Library
//#include <ArduinoBLE.h>

//Include the Tensorflow Lite Library
#include "TensorFlowLite.h"

#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Our model
#include "fyp2.h"

// Figure out what's going on in our model
#define DEBUG 1

const float accelerationThreshold = 2.5; // threshold of significant in G's
const int numSamples = 119; //no of data sample 

int samplesRead = numSamples;
//int samplesRead=0;



// TFLite globals, used for compatibility with Arduino-style sketches
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;
  TfLiteTensor* model_output = nullptr;


  // Create an area of memory to use for input, output, and other TensorFlow
  // arrays. You'll need to adjust this by combiling, running, and looking
  // for errors.
  constexpr int kTensorArenaSize = 40 * 1024;//40kb datasize occupy in nano
  uint8_t tensor_arena[kTensorArenaSize];
}

void setup() {

#if DEBUG
  while(!Serial); //gsm srtated
#endif

Serial.begin(9600);
mySerial.begin(9600);
//Serial.println("Started");

//Start IMU
if(!IMU.begin()){
  Serial.println("Failed to Initialize IMU!"); //if not detect
  while(1);
}

//    IMU.setContinuousMode();
  


  // Set up logging (will report to Serial, even within TFLite functions)
  static tflite::MicroErrorReporter micro_error_reporter; //to report error
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure
  model = tflite::GetModel(fyp2); //we have call the model fyp
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }



  /*static tflite::MicroMutableOpResolver<2> micro_op_resolver;  // NOLINT
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddSoftmax();*/
  static tflite::AllOpsResolver micro_op_resolver;

  // Build an interpreter to run the model
  static tflite::MicroInterpreter static_interpreter(
    model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter); //op resolser to solve any problem
  interpreter = &static_interpreter; //to excute interpriter is use

  // Allocate memory from the tensor_arena for the model's tensors

  //interpreter->AllocateTensors();

  
  TfLiteStatus allocate_status = interpreter->AllocateTensors(); //to alocate 40kb which we have create before
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while(1);
  }
  

  // Assign model input and output buffers (tensors) to pointers
   model_input = interpreter->input(0); 
   model_output = interpreter->output(0);

  // Get information about the memory area to use for the model's input
  // Supported data types:
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/common.h#L226
/*#if DEBUG
  Serial.print("Number of dimensions: ");
  Serial.println(model_input->dims->size);
  Serial.print("Dim 1 size: ");
  Serial.println(model_input->dims->data[0]);
  Serial.print("Dim 2 size: ");
  Serial.println(model_input->dims->data[1]);
  Serial.print("Input type: ");
  Serial.println(model_input->type);
#endif*/


}

void loop() {

 float aX, aY, aZ, gX, gY, gZ, AX, AY, AZ, GX, GY, GZ, at;
 int a=0,b=0,c=1,d=0;

  while (samplesRead == numSamples) {
    if (IMU.accelerationAvailable()) {
      // read the acceleration data
      IMU.readAcceleration(AX, AY, AZ);

      // sum up the absolutes
      float aSum = fabs(AX) + fabs(AY) + fabs(AZ);

      // check if it's above the threshold
      if (aSum >= accelerationThreshold) {
        // reset the sample read count
        samplesRead = 0;
        break;
      }
    }
  }

    while (samplesRead < numSamples) {
    // check if new acceleration AND gyroscope data is available
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      // read the acceleration and gyroscope data
      IMU.readAcceleration(AX, AY, AZ);
      IMU.readGyroscope(GX, GY, GZ);


      aX = AX/200;
      aY = AY/200;
      aZ = AZ/200;
      gX = GX/2500;
      gY = GY/1500;
      gZ = GZ/1500;

      // normalize the IMU data between 0 to 1 and store in the model's
      // input tensor
      model_input->data.f[0] = aX;
      model_input->data.f[1] = aY;
      model_input->data.f[2] = aZ;
      model_input->data.f[3] = gX;
      model_input->data.f[4] = gY;
      model_input->data.f[5] = gZ;

        samplesRead++;

      if (samplesRead == numSamples) {
        // Run inferencing
        TfLiteStatus invokeStatus = interpreter->Invoke();
        if (invokeStatus != kTfLiteOk) {
          Serial.println("Invoke failed!");
          while (1);
          return;
        }

        // Loop through the output tensor values from the model

          float op_pred = model_output->data.f[0];
        // Serial.println(op_pred);
          if(op_pred > 0.5000)
          {
             a=1;
          }

          at= sqrt(AX*AX + AY*AY + AZ*AZ);
          //Serial.println(at);
          if(at>9500){
             b=1;
             digitalWrite(LED_BUILTIN,HIGH);
          }


          /*for(c=1;c<=15000;c++){
            if(digitalRead(5)==LOW){
              a=0;                            //reset button
              b=0;
              digitalWrite(LED_BUILTIN,LOW);
            }
          }*/
          if((a==1)||(b==1))
          {
            //Serial.println("Fall Detected");
            String response=""; 
            mySerial.println("AT");// ASCII code of CTRL+Z
            delay(1000);
            response=mySerial.readString();
            Serial.println(response);
            mySerial.println("AT+CMGF=1");    //Sets the GSM Module in Text Mode
            delay(1000);  // Delay of 1000 milli seconds or 1 second
            mySerial.println("AT+CMGS=\"8888774470\"\r"); // Replace x with mobile number
            delay(1000);
            mySerial.println("Fall Has been Detected Reach Immediately.");
            mySerial.println("Please Reach the following Location Immediately");
            mySerial.println("https://www.google.com/maps/search/?api=1&query=&latitude,longitude");// The SMS text you want to send
            delay(100);
            mySerial.println((char)26);// ASCII code of CTRL+Z
            delay(1000);
          }
      }
    }
   }
}
