using System;
using System.IO;
//using ImageClassification;
using ImageClassificationGPU;
using ImageClassification.IO;

namespace TestConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            if (args.Length == 1 && Directory.Exists(args[0]))
            {
                // Define data-set folder.
                string dataDir = args[0];
                // Define hyper-paramters such as Epoch or BatchSize.
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
                // Save the results as HTML file.
                results.SaveAsHTML();

                string imageToClassify = @"C:\your\imageToClassify(apple_or_banana_or_orange).png";
                var p = Classifier.GetSingleImagePrediction(results.Resultfiles.PipelineZip, results.Resultfiles.ModelZip, imageToClassify);
                Console.WriteLine($@"Predicted image label is: ""{p.PredictedLabel}"". Score:{p.HighScore}");
            }
        }
    }
}
