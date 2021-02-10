using System;

namespace ImageClassification
{
    internal class ImageData
    {
        internal string Name { get; set; }
        internal string ImagePath { get; set; }
    }

    internal class ImagePrediction : ImageData
    {
        internal string PredictedLabelValue { get; set; }
        internal float[] Score { get; set; }
    }
}

