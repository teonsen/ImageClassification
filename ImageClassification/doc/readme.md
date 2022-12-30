## What is this?

This package is a wrapper library of ML.NET to do image classification.
You will need `ImageClassification.IO` packeage along with this package.

## Usege

### Trainer class

`Trainer` class is a class to train the Model (which will be two zip files) from your image data.
Image data you have to prepare in advence is a foldered image data like below.
Each folder names will be predicted Label names (Apple/Banana/Orange/...).

- Fruits
    - Apple
        - apple001.jpg
        - apple002.jpg
        - ...
    - Banana
    - Orange
    - ...

More images you prepare, better result you'll get.

Next, call `Trainer.GenerateModel` with `dataDir` you prepared.
<pre>
    // Define hyper-paramters such as Epoch and BatchSize.
    var hp = new HyperParameter
    {
        Epoch = 200,
        BatchSize = 10,
        LearningRate = 0.01f,
        eTrainerArchitecture = eTrainerArchitectures.ResnetV250,
        TestFraction = 0.3f,
        ResultsToShow = 20
    };

    // Train and generate the model.
    var results = Trainer.GenerateModel(dataDir, hp);

    // Save the results as HTML file.(optional)
    results.SaveAsHTML();
</pre>

### Classifier class

`Classifier` class is a class to get a prediction from the image file you give (=imageToClassify).
Call `Classifier.GetSingleImagePrediction` along with the Model you've created at previus section.

<pre>
    string imageToClassify = @"C:\your\imageToClassify(apple_or_banana_or_orange).png";
    var p = Classifier.GetSingleImagePrediction(results.Resultfiles.PipelineZip, results.Resultfiles.ModelZip, imageToClassify);
    Console.WriteLine($@"Predicted image label is: ""{p.PredictedLabel}"". Score:{p.HighScore}");
</pre>
