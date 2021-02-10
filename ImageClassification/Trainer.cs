using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Vision;

namespace ImageClassification
{
    public static class Trainer
    {
        // This code is based on the articles below.
        // https://qiita.com/hiromasa-masuda/items/e1a3ea60f0c838f17589
        // https://docs.microsoft.com/ja-jp/dotnet/machine-learning/how-to-guides/save-load-machine-learning-models-ml-net

        /// <summary>
        /// This is a sample of image classification using transfer learning.(転移学習により画像の分類を行うサンプルです。)
        /// </summary>
        /// <param name="imagesFolder">Specify the root folder where the images are foldered for each type.(それぞれの種類にフォルダ分けされた画像があるルートフォルダを指定してください。)</param>
        /// <param name="createdPipelinePath">The path where the pipeline (.zip file) is saved. The pipeline will be saved directly under 'imagesFolder'. This file will be needed when predicting images along with the model.</param>
        /// <param name="createdModelPath">The path where the trained model (.zip file) is saved. The trained model will be saved directly under 'imagesFolder'. This file will be needed when predicting images.</param>
        /// <param name="saveResultHTML">Specify "true" if you want to output the learning results as an HTML file.(学習結果をHTMLファイルとして出力する場合はtrueを指定してください。)</param>
        public static void GenerateModel(string imagesFolder, HyperParameter hp, out string createdPipelinePath, out string createdModelPath, bool saveResultHTML = false)
        {
            createdModelPath = Path.Combine(imagesFolder, "model.zip");
            createdPipelinePath = Path.Combine(imagesFolder, "data_preparation_pipeline.zip");
            string resultHTMLPath = Path.Combine(imagesFolder, $"Result{DateTimeOffset.Now:yyyyMMddHHmmss}.html");

            // データフォルダの指定
            // データフォルダ下にあるフォルダ名をラベル名とする
            var dataFiles = Directory.GetFiles(imagesFolder, "*", SearchOption.AllDirectories).ToList().FindAll(a => a.ToLower().EndsWith(".jpg") || a.ToLower().EndsWith(".png"));

            // データセットの作成
            var dataSet = dataFiles.Select(f => new ImageData
            {
                ImagePath = f,
                Name = Directory.GetParent(f).Name
            });

            // コンテキストの生成
            MLContext mlContext = new MLContext(seed: 1);

            // データのロード
            IDataView dataView = mlContext.Data.LoadFromEnumerable(dataSet);
            // データセットをシャッフル
            IDataView shuffledDataView = mlContext.Data.ShuffleRows(dataView);

            // データ前処理
            // データセットの加工
            var imagesPipeline = mlContext.Transforms.Conversion.MapValueToKey(
                // ラベル名を数値に変換して列名を Label とする
                inputColumnName: nameof(ImageData.Name),
                outputColumnName: "Label",
                keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
                .Append(mlContext.Transforms.LoadRawImageBytes(
                    // パスから画像をロード
                    inputColumnName: nameof(ImageData.ImagePath),
                    imageFolder: null,
                    outputColumnName: "RawImageBytes"));

            // データセット用に Transformer を生成 (データ準備パイプライン)
            var dataPrepTransformer = imagesPipeline.Fit(shuffledDataView);
            // Transformer をデータセットに適用
            var transformedDataView = dataPrepTransformer.Transform(shuffledDataView);

            // データセットを学習データ、検証データ、評価データに分割
            // データセットを 7:3 に分割(データセットの 70% を学習データとする)
            var trainValidationTestSplit = mlContext.Data.TrainTestSplit(transformedDataView, testFraction: 0.3);
            IDataView trainDataView70 = trainValidationTestSplit.TrainSet;
            IDataView testDataView30 = trainValidationTestSplit.TestSet;
            // 検証/評価データセットを 8:2 に分割
            var validationTestSplit = mlContext.Data.TrainTestSplit(testDataView30, testFraction: 0.2);

            // データセットの 24% を検証データとする
            IDataView validationDataView24 = validationTestSplit.TrainSet;
            // データセットの 6% を評価データとする
            IDataView testDataView6 = validationTestSplit.TestSet;

            // 学習の定義
            var classifierOptions = new ImageClassificationTrainer.Options()
            {
                LabelColumnName = "Label", //ラベル列
                FeatureColumnName = "RawImageBytes", // 特徴列
                //Arch = ImageClassificationTrainer.Architecture.ResnetV250, //転移学習モデルの選択
                Arch = (ImageClassificationTrainer.Architecture)hp.eTrainerArchitecture, //転移学習モデルの選択
                Epoch = hp.Epoch,
                BatchSize = hp.BatchSize,
                LearningRate = hp.LearningRate,
                ValidationSet = validationDataView24, // 検証データを設定
                MetricsCallback = (metrics) => Console.WriteLine(metrics),
                WorkspacePath = @".\Workspace",
            };

            var pipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(classifierOptions)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(
                    // 推論結果のラベルを数値から予測文字列に変換
                    inputColumnName: "PredictedLabel",
                    outputColumnName: nameof(ImagePrediction.PredictedLabelValue)));

            // 学習の実行
            ITransformer model = pipeline.Fit(trainDataView70);

            // データ準備パイプラインをファイルに保存
            mlContext.Model.Save(dataPrepTransformer, trainDataView70.Schema, createdPipelinePath);

            // 学習モデルをファイルに保存
            mlContext.Model.Save(model, trainDataView70.Schema, createdModelPath);

            // テストデータで推論を実行
            IDataView prediction = model.Transform(testDataView6);
            IEnumerable<ImagePrediction> predictions = mlContext.Data.CreateEnumerable<ImagePrediction>(prediction, reuseRowObject: true).Take(10);

            if (saveResultHTML)
            {
                // 結果を保存
                SaveResultHTML(mlContext, trainDataView70, prediction, predictions, createdModelPath, resultHTMLPath);
            }
        }

        private static void SaveResultHTML(MLContext context, IDataView trainData, IDataView prediction, IEnumerable<ImagePrediction> predictions, string modelFilePath, string resultHTMLPath)
        {
            // テストデータでの推論結果をもとに評価指標を計算
            var metrics = context.MulticlassClassification.Evaluate(prediction);

            // ラベルと予測文字列のキーバリューを取得
            VBuffer<ReadOnlyMemory<char>> keyValues = default;
            trainData.Schema["Label"].GetKeyValues(ref keyValues);

            // HTML で評価結果を書き出し
            using (var writer = new StreamWriter(resultHTMLPath))
            {
                writer.WriteLine($"<html><head><title>{Path.GetFileName(modelFilePath)}</title>");
                writer.WriteLine("<link rel=\"stylesheet\" href=\"https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css\" integrity=\"sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2\" crossorigin=\"anonymous\">");
                writer.WriteLine("</head><body>");

                writer.WriteLine($"<h1>Metrics for {Path.GetFileName(modelFilePath)}</h1>");
                // メトリックの書き出し
                writer.WriteLine("<div><table class=\"table table-striped\">");
                writer.WriteLine($"<tr><td>MicroAccuracy</td><td>{metrics.MicroAccuracy:0.000}</td></tr></tr>");
                writer.WriteLine($"<tr><td>MacroAccuracy</td><td>{metrics.MacroAccuracy:0.000}</td></tr></tr>");
                writer.WriteLine($"<tr><td>Precision</td><td>{metrics.ConfusionMatrix.PerClassPrecision.Average():0.000}</td></tr></tr>");
                writer.WriteLine($"<tr><td>Recall</td><td>{metrics.ConfusionMatrix.PerClassRecall.Average():0.000}</td></tr></tr>");
                writer.WriteLine($"<tr><td>LogLoss</td><td>{metrics.LogLoss:0.000}</td></tr></tr>");
                writer.WriteLine($"<tr><td>LogLossReduction</td><td>{metrics.LogLossReduction:0.000}</td></tr></tr>");

                // クラス毎の適合率
                writer.WriteLine("<tr><td>PerClassPrecision</td><td>");
                metrics.ConfusionMatrix.PerClassPrecision
                .Select((p, i) => (Precision: p, Index: i))
                .ToList()
                .ForEach(p =>
                    writer.WriteLine($"{keyValues.GetItemOrDefault(p.Index)}: {p.Precision:0.000}<br />"));
                writer.WriteLine("</td></tr>");

                // クラス毎の再現率
                writer.WriteLine("<tr><td>PerClassRecall</td><td>");
                metrics.ConfusionMatrix.PerClassRecall
                .Select((p, i) => (Recall: p, Index: i))
                .ToList()
                .ForEach(p =>
                    writer.WriteLine($"{keyValues.GetItemOrDefault(p.Index)}: {p.Recall:0.000}<br />"));
                writer.WriteLine("</td></tr></table></div>");

                // 評価データ毎の分類結果
                writer.WriteLine($"<h1>Predictions</h1>");
                writer.WriteLine($"<div><table class=\"table table-bordered\">");

                foreach (var p in predictions)
                {
                    writer.WriteLine($"<tr><td>");
                    // 画像ファイル名
                    writer.WriteLine($"{Path.GetFileName(p.ImagePath)}<br />");
                    // 正解ラベル
                    writer.WriteLine($"Actual Value: {p.Name}<br />");
                    // 推論結果
                    writer.WriteLine($"Predicted Value: {p.PredictedLabelValue}<br />");
                    // 画像
                    writer.WriteLine($"<img class=\"img-fluid\" src=\"{p.ImagePath}\" /></td>");
                    // クラス毎の推論結果
                    writer.WriteLine($"<td>");
                    p.Score.Select((s, i) => (Index: i, Label: keyValues.GetItemOrDefault(i), Score: s))
                    .OrderByDescending(c => c.Score)
                    .Take(10) // 上位 10 件
                    .ToList()
                    .ForEach(c =>
                    {
                        writer.WriteLine($"{c.Label}: {c.Score:P}<br />");
                    });

                    writer.WriteLine("</td></tr>");
                }
                writer.WriteLine("</table></div>");
                writer.WriteLine("</body></html>");
            }
        }

    }

}
