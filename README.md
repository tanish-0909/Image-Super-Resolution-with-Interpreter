# Image Super Resolution with Interpreter
 After too many tries I have finally found a way this still works in 2025 where I can integrate ML models into android apps.


Althought I have gone and explained how to do every step in detail. I assume that the reader has atleast basic knowledge of using Android Studio.

 # How to use this repo:
 1. First find the pytorch model you wish to use for image super resolution
 2. Open the repnet-rtsr.ipynb and follow the instructions with your model. I have explained everything step by step. Follow these steps in a kaggle notebook.
 3. Open the "Step by Step guidelines for APP dev.pdf" and follow the 28 steps mentioned in the pdf on Android Studio.
 4. Find any other necessary code in the folder.

# Future plans with this repo:
1. Initialize the interpreter when starting the app instead of when pressing the button.
2. Reduce latency
3. Better UI
4. Integrate CameraX from : https://github.com/tanish-0909/Camera_with_kotlin
5. Make better Image Super Resolution models


### Credits:
1. https://github.com/margaretmz/esrgan-e2e-tflite-tutorial
2. https://www.youtube.com/watch?v=gVJC1j2n9tE&list=PLguFZR_OzRTcAQuANLwsUcB2CGnGQEy2A&index=3&t=1234s&pp=gAQBiAQB
3. https://github.com/lovelyzzkei/MobedSR
4. https://ai.google.dev/edge/api/tflite/java/org/tensorflow/lite/Interpreter
5. https://github.com/ezrealzhang/REPNet_NTIRE23-RTSR
6. https://www.ee.iitm.ac.in/faculty/profile/pravinnair
