using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ImageClassification.IO
{
    public static class Predictor
    {
        /// <summary>
        /// Returns the result of predicting what the image file is, using the saved model trained by Trainer.GenerateModel().(Trainer.GenerateModel()で学習させた保存済みモデルを使用して、画像ファイルが何であるかを予測した結果を返します。)
        /// </summary>
        /// <param name="pipelineZipFilePath"></param>
        /// <param name="trainedModelZipFilePath"></param>
        /// <param name="targetImagePath"></param>
        /// <returns></returns>
        public static PredictionResult ClassifySingleImage(string pipelineZipFilePath, string trainedModelZipFilePath, string targetImagePath)
        {
            MLContext mlContext = new MLContext(seed: 1);
            DataViewSchema dataPrepPipelineSchema, modelSchema;
            // データ準備パイプラインとトレーニング済みモデルを読み込む
            ITransformer dataPrepPipeline = mlContext.Model.Load(pipelineZipFilePath, out dataPrepPipelineSchema);
            ITransformer trainedModel = mlContext.Model.Load(trainedModelZipFilePath, out modelSchema);

            // データのロード
            var data = new List<ImageData>();
            data.Add(new ImageData { ImagePath = targetImagePath });
            var dataView = mlContext.Data.LoadFromEnumerable(data);
            var transformedDataView = dataPrepPipeline.Transform(dataView);
            var prediction = trainedModel.Transform(transformedDataView);
            var predictions = mlContext.Data.CreateEnumerable<ImagePrediction>(prediction, reuseRowObject: true);

            // ラベルと予測文字列のキーバリューを取得
            VBuffer<ReadOnlyMemory<char>> keyValues = default;
            transformedDataView.Schema["Label"].GetKeyValues(ref keyValues);

            var result = predictions.First().Score.Select((s, i) => (Index: i, Label: keyValues.GetItemOrDefault(i), Score: s))
                .OrderByDescending(c => c.Score)
                .Take(1)
                .ToList()[0];
            return new PredictionResult
            {
                FileName = Path.GetFileName(targetImagePath),
                PredictedLabel = result.Label.ToString(),
                Score = result.Score
            };
        }

    }
}
