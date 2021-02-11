# ImageClassification
A NuGet package for image classification wrapper using ML.NET.
https://www.nuget.org/packages/ImageClassification/

# Useage

## Create or download data-set to test
You can download the MIT licensed janken-dataset from here.
https://github.com/karaage0703/janken_dataset

## Train and generate the model
```csharp
using ImageClassification;

string dataDir = @"C:\Users\user\Downloads\janken_dataset-master\janken_dataset-master";
string modelPath, pipelinePath;
// Define hyper-paramters such as Epoch or BatchSize.
var hp = new HyperParameter { Epoch = 200, BatchSize = 10, LearningRate = 0.01f, eTrainerArchitecture = eTrainerArchitectures.ResnetV250 };
Trainer.GenerateModel(dataDir, hp, out pipelinePath, out modelPath, true);
```
Once you run the code above, pipeline.zip and model.zip will be created in the dataset folder.

## Classify
To predict an image, pass the pipeline and model.zip output by Trainer.GenerateModel() above, as well as the image file, to the following function.
```csharp
var result = Predictor.ClassifySingleImage(${ the path of pipeline.zip}, ${ the path of model.zip}, ${imagePath});
```

# Acknowledgements
* Most part of the codes are originaly written by [@Hiromasa-Masuda](https://github.com/Hiromasa-Masuda)
* Thank you for [@karaage0703](https://github.com/karaage0703) to create useful dataset
* Thank you for all the effort to create ML.NET and SciSharp
