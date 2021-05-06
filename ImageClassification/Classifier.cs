using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using ImageClassification.IO;

namespace ImageClassification
{
    public static class Classifier
    {
        /// <summary>
        /// Returns the result of predicting what the image file is, using the saved model trained by Trainer.GenerateModel().(Trainer.GenerateModel()で学習させた保存済みモデルを使用して、画像ファイルが何であるかを予測した結果を返します。)
        /// </summary>
        /// <param name="pipelineZipFilePath"></param>
        /// <param name="trainedModelZipFilePath"></param>
        /// <param name="targetImageFile"></param>
        /// <returns></returns>
        public static ImageResult GetSingleImagePrediction(string pipelineZipFilePath, string trainedModelZipFilePath, string targetImageFile)
        {
            MLContext mlContext = new MLContext(seed: 1);
            DataViewSchema dataPrepPipelineSchema, modelSchema;
            // データ準備パイプラインとトレーニング済みモデルを読み込む
            ITransformer dataPrepPipeline = mlContext.Model.Load(pipelineZipFilePath, out dataPrepPipelineSchema);
            ITransformer trainedModel = mlContext.Model.Load(trainedModelZipFilePath, out modelSchema);

            // データのロード
            var data = new List<ImageData>();
            data.Add(new ImageData { ImagePath = targetImageFile });
            var dataView = mlContext.Data.LoadFromEnumerable(data);
            var transformedDataView = dataPrepPipeline.Transform(dataView);
            var prediction = trainedModel.Transform(transformedDataView);
            var predictions = mlContext.Data.CreateEnumerable<ImagePrediction>(prediction, reuseRowObject: true);

            // ラベルと予測文字列のキーバリューを取得
            VBuffer<ReadOnlyMemory<char>> keyValues = default;
            transformedDataView.Schema["Label"].GetKeyValues(ref keyValues);
            return new ImageResult(predictions.First(), keyValues);
        }

        public static ImageResult GetSingleImagePrediction(MLContext mlContext, ITransformer dataPrepPipeline, ITransformer trainedModel, string targetImageFile)
        {
            // データのロード
            var data = new List<ImageData>();
            data.Add(new ImageData { ImagePath = targetImageFile });
            var dataView = mlContext.Data.LoadFromEnumerable(data);
            var transformedDataView = dataPrepPipeline.Transform(dataView);
            var prediction = trainedModel.Transform(transformedDataView);
            var predictions = mlContext.Data.CreateEnumerable<ImagePrediction>(prediction, reuseRowObject: true);

            // ラベルと予測文字列のキーバリューを取得
            VBuffer<ReadOnlyMemory<char>> keyValues = default;
            transformedDataView.Schema["Label"].GetKeyValues(ref keyValues);
            return new ImageResult(predictions.First(), keyValues);
        }

        public static List<ImageResult> GetBulkImagePrediction(string pipelineZipFilePath, string trainedModelZipFilePath, string targetImagesFolder)
        {
            MLContext mlContext = new MLContext(seed: 1);
            DataViewSchema dataPrepPipelineSchema, modelSchema;
            // データ準備パイプラインとトレーニング済みモデルを読み込む
            ITransformer dataPrepPipeline = mlContext.Model.Load(pipelineZipFilePath, out dataPrepPipelineSchema);
            ITransformer trainedModel = mlContext.Model.Load(trainedModelZipFilePath, out modelSchema);

            // データのロード
            var dataFiles = Directory.EnumerateFiles(targetImagesFolder, "*", SearchOption.AllDirectories).Where(s =>
            s.EndsWith(".jpg", StringComparison.CurrentCultureIgnoreCase) || s.EndsWith(".png", StringComparison.CurrentCultureIgnoreCase));
            var dataSet = dataFiles.Select(f => new ImageData
            {
                ImagePath = f,
                Name = Directory.GetParent(f).Name
            });
            var dataView = mlContext.Data.LoadFromEnumerable(dataSet);
            var transformedDataView = dataPrepPipeline.Transform(dataView);
            var prediction = trainedModel.Transform(transformedDataView);
            var predictions = mlContext.Data.CreateEnumerable<ImagePrediction>(prediction, reuseRowObject: true);

            // ラベルと予測文字列のキーバリューを取得
            VBuffer<ReadOnlyMemory<char>> keyValues = default;
            transformedDataView.Schema["Label"].GetKeyValues(ref keyValues);

            var allImgResults = new List<ImageResult>();
            foreach (var p in predictions)
            {
                allImgResults.Add(new ImageResult(p, keyValues));
            }
            var imgResults = (from a in allImgResults orderby a.HighScore descending select a);
            return imgResults.ToList();
        }

    }
}
