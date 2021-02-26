using System;
using System.Collections.Generic;
using System.Text;

namespace ImageClassification.IO
{
    public class PredictionResult
    {
        public string FileName { get; set; }
        public string PredictedLabel { get; set; }
        public float Score { get; set; }
    }
}
