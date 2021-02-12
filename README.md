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

// Define data-set folder.
string dataDir = @"C:\Users\user\Downloads\janken_dataset-master\janken_dataset-master";
            
// Define hyper-paramters such as Epoch or BatchSize.
var hp = new HyperParameter {
    Epoch = 200,
    BatchSize = 10,
    LearningRate = 0.01f,
    eTrainerArchitecture = eTrainerArchitectures.ResnetV250,
    TestFraction = 0.3f
};

// Train and generate the model.
var resultFile = Trainer.GenerateModel(dataDir, hp);
```
Once you run the code above, pipeline.zip and model.zip will be created in the dataset folder.

## Classify
To predict an image, pass the pipeline and model.zip output by Trainer.GenerateModel() above, as well as the image file, to the following function.
```csharp
// Classify the single image.
string imageToClassify = @"path to the image";
var prediction = Predictor.ClassifySingleImage(resultFile.PipelineSavedPath, resultFile.ModelSavedPath, imageToClassify);
Console.WriteLine($@"Predicted image label is: ""{prediction.PredictedLabel}"". Score:{prediction.Score}");
```

# Acknowledgements
* Most part of the codes are originaly written by [@Hiromasa-Masuda](https://github.com/Hiromasa-Masuda)
* Thank you for [@karaage0703](https://github.com/karaage0703) to create useful dataset
* Thank you for all the effort to create ML.NET and SciSharp
