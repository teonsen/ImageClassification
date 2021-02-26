#define _USEGPU_
using System;
using System.IO;
#if _USEGPU_
using ImageClassificationGPU;
#else
using ImageClassification;
#endif
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
                    ResultsToShow = 30
                };

                // Train and generate the model.
                var resultFiles = Trainer.GenerateModel(dataDir, hp);
            }
        }

        static void Predict(TrainingResultFiles f)
        {
            var result = Predictor.ClassifySingleImage(f.PipelineSavedPath, f.ModelSavedPath, "path");

        }
    }
}
