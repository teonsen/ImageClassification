﻿using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Transforms;
using Microsoft.ML.Vision;
using ImageClassification.IO;

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
        /// <param name="imagesFolder">Specify the root folder where the images are foldered for each label.(それぞれの種類にフォルダ分けされた画像があるルートフォルダを指定してください。)</param>
        /// <param name="hp">Hyper parameters</param>
        /// <returns></returns>
        public static TrainingResults GenerateModel(string imagesFolder, HyperParameter hp)
        {
            // コンテキストの生成
            MLContext mlContext = new MLContext(seed: 1);
            // データをロードしてシャッフルする
            var shuffledDataView = DataLoader.GetShuffledDataView(mlContext, imagesFolder);

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
            var trainValidationTestSplit = mlContext.Data.TrainTestSplit(transformedDataView, testFraction: hp.TestFraction);
            IDataView trainDataView = trainValidationTestSplit.TrainSet;
            IDataView testDataView = trainValidationTestSplit.TestSet;
            // 検証/評価データセットを 8:2 に分割
            var validationTestSplit = mlContext.Data.TrainTestSplit(testDataView, testFraction: 0.2);

            // データセットの 24% を検証データとする
            IDataView validationDataView24 = validationTestSplit.TrainSet;
            // データセットの 6% を評価データとする
            IDataView testDataView6 = validationTestSplit.TestSet;

            // 学習の定義
            var classifierOptions = new ImageClassificationTrainer.Options()
            {
                LabelColumnName = "Label", //ラベル列
                FeatureColumnName = "RawImageBytes", // 特徴列
                //Arch = ImageClassificationTrainer.Architecture.ResnetV250,
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
            ITransformer model = pipeline.Fit(trainDataView);

            var resultFiles = new ResultFiles(imagesFolder);
            // データ準備パイプラインをファイルに保存
            mlContext.Model.Save(dataPrepTransformer, trainDataView.Schema, resultFiles.PipelineZip);

            // 学習モデルをファイルに保存
            mlContext.Model.Save(model, trainDataView.Schema, resultFiles.ModelZip);

            // テストデータで推論を実行
            IDataView prediction = model.Transform(testDataView6);
            IEnumerable<ImagePrediction> predictions = mlContext.Data.CreateEnumerable<ImagePrediction>(prediction, reuseRowObject: true);

            // テストデータでの推論結果をもとに評価指標を計算
            var mcm = mlContext.MulticlassClassification.Evaluate(prediction);
            return new TrainingResults(mcm, trainDataView, predictions, resultFiles, hp.ResultsToShow);
        }

    }

}
