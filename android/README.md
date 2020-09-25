# CheckFit application

This application is mostly based on the [TensorFlow Lite Android example](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android).

Main changes:
- Reworked the user interface
- Added audio for live feedback
- Use frontal camera instead of back one
- Added custom trained models (for Plank and Holding Squat)
- Removed image preprocessing steps, crop & rotate (see method Classifier::loadImage), to be more similar with training images (also a button is added to take pictures from inside the application, but it's commented out)