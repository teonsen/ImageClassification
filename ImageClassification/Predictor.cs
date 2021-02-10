using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ImageClassification
{
    public static class Predictor
    {
        /// <summary>
        /// Returns the result of predicting what the image file is, using the saved model trained by Trainer.GenerateModel().(Trainer.GenerateModel()で学習させた保存済みモデルを使用して、画像ファイルが何であるかを予測した結果を返します。)
        /// </summary>
        /// <param name="pipelineFilePath"></param>
        /// <param name="trainedModelFilePath"></param>
        /// <param name="targetImagePath"></param>
        /// <returns></returns>
        public static PredictionResult ClassifySingleImage(string pipelineFilePath, string trainedModelFilePath, string targetImagePath)
        {
            MLContext mlContext = new MLContext(seed: 1);
            DataViewSchema dataPrepPipelineSchema, modelSchema;
            // データ準備パイプラインとトレーニング済みモデルを読み込む
            ITransformer dataPrepPipeline = mlContext.Model.Load(pipelineFilePath, out dataPrepPipelineSchema);
            ITransformer trainedModel = mlContext.Model.Load(trainedModelFilePath, out modelSchema);

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

    public class PredictionResult
    {
        public string FileName { get; internal set; }
        public string PredictedLabel { get; internal set; }
        public float Score { get; internal set; }
    }
}
