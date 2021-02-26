using System;
using System.Collections.Generic;
using System.IO;

namespace ImageClassification.IO
{
    /// <summary>
    /// Paths to model.zip, pipeline.zip, and result.html which are saved as a result of training
    /// </summary>
    public class TrainingResultFiles
    {
        /// <summary>
        /// The path where the trained model (.zip file) is saved. The trained model will be saved directly under 'imagesFolder'. This file will be needed when predicting images.
        /// </summary>
        public string ModelSavedPath { get; private set; }
        /// <summary>
        /// The path where the pipeline (.zip file) is saved. The pipeline will be saved directly under 'imagesFolder'. This file will be needed when predicting images along with the model.
        /// </summary>
        public string PipelineSavedPath { get; private set; }
        public string ResultHTMLSavedPath { get; private set; }

        public TrainingResultFiles(string imagesFolder)
        {
            ModelSavedPath = Path.Combine(imagesFolder, "model.zip");
            PipelineSavedPath = Path.Combine(imagesFolder, "pipeline.zip");
            ResultHTMLSavedPath = Path.Combine(imagesFolder, $"result{DateTimeOffset.Now:yyyyMMddHHmmss}.html");
        }
    }
}
