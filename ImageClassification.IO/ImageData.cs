using System;

namespace ImageClassification.IO
{
    public class ImageData
    {
        public string Name { get; set; }
        public string ImagePath { get; set; }
    }

    public class ImagePrediction : ImageData
    {
        public string PredictedLabelValue { get; set; }
        public float[] Score { get; set; }
    }
}

