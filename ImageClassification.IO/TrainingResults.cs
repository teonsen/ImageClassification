using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ImageClassification.IO
{
    /// <summary>
    /// Paths to model.zip, pipeline.zip, and result.html which are saved as a result of training
    /// </summary>
    public class ResultFiles
    {
        /// <summary>
        /// The path where the trained model (.zip file) is saved. The trained model will be saved directly under 'imagesFolder'. This file will be needed when predicting images.
        /// </summary>
        public string ModelZip { get; private set; }
        /// <summary>
        /// The path where the pipeline (.zip file) is saved. The pipeline will be saved directly under 'imagesFolder'. This file will be needed when predicting images along with the model.
        /// </summary>
        public string PipelineZip { get; private set; }
        public string ResultHTML { get; private set; }
        public ResultFiles(string imagesFolder)
        {
            ModelZip = Path.Combine(imagesFolder, "model.zip");
            PipelineZip = Path.Combine(imagesFolder, "pipeline.zip");
            ResultHTML = Path.Combine(imagesFolder, $"result{DateTimeOffset.Now:yyyyMMddHHmmss}.html");
        }
    }

    public class TrainingResults
    {
        public ResultFiles Resultfiles { get; }
        public double MicroAccuracy { get; }
        public double MacroAccuracy { get; }
        public double Precision { get; }
        public double Recall { get; }
        public double LogLoss { get; }
        public double LogLossReduction { get; }
        public List<ImageResult> ImageResults { get; }
        public List<PerClassResult> PerClassPrecisions { get; }
        public List<PerClassResult> PerClassRecalls { get; }
        public int CorrectAnswers { get; }
        public int AnswersOverScore95 { get; }
        public int CorrectAnswersOverScore95 { get; }
        public float CorrectAnswerRate { get; }
        public float CorrectAnswerRateOverScore95 { get; }

        public TrainingResults(MulticlassClassificationMetrics metrics, IDataView trainData, IEnumerable<ImagePrediction> predictions, ResultFiles resultFiles, int resultsToShow)
        {
            Resultfiles = resultFiles;
            MicroAccuracy = metrics.MicroAccuracy;
            MacroAccuracy = metrics.MacroAccuracy;
            Precision = metrics.ConfusionMatrix.PerClassPrecision.Average();
            Recall = metrics.ConfusionMatrix.PerClassRecall.Average();
            LogLoss = metrics.LogLoss;
            LogLossReduction = metrics.LogLossReduction;

            // ラベルと予測文字列のキーバリューを取得
            VBuffer<ReadOnlyMemory<char>> keyValues = default;
            trainData.Schema["Label"].GetKeyValues(ref keyValues);

            var allImgResults = new List<ImageResult>();
            foreach (var p in predictions)
            {
                allImgResults.Add(new ImageResult(p, keyValues));
            }
            var imgResults = (from a in allImgResults orderby a.HighScore descending select a).Take(resultsToShow);
            ImageResults = imgResults.ToList();

            // 正答数
            CorrectAnswers = (from a in imgResults where a.IsCorrect select a).Count();
            // Scoreが95以上の数
            AnswersOverScore95 = (from a in imgResults where a.HighScore >= 0.95f select a).Count();
            // Scoreが95以上かつ正答した数
            CorrectAnswersOverScore95 = (from a in imgResults where a.IsCorrect && a.HighScore >= 0.95f select a).Count();
            CorrectAnswerRate = CorrectAnswers / (float)resultsToShow;
            CorrectAnswerRateOverScore95 = CorrectAnswersOverScore95 / (float)AnswersOverScore95;

            PerClassPrecisions = new List<PerClassResult>();
            metrics.ConfusionMatrix.PerClassPrecision
            .Select((p, i) => (Precision: p, Index: i))
            .ToList()
            .ForEach(p =>
                PerClassPrecisions.Add(new PerClassResult(keyValues.GetItemOrDefault(p.Index).ToString(), p.Precision)));

            PerClassRecalls = new List<PerClassResult>();
            metrics.ConfusionMatrix.PerClassRecall
            .Select((p, i) => (Recall: p, Index: i))
            .ToList()
            .ForEach(p =>
                PerClassRecalls.Add(new PerClassResult(keyValues.GetItemOrDefault(p.Index).ToString(), p.Recall)));
        }

        public class PerClassResult
        {
            public string Label { get; }
            public double Value { get; }

            public PerClassResult(string label, double value)
            {
                Label = label;
                Value = value;
            }
        }

        public class ImageResult
        {
            internal string Name { get; private set; }
            internal string PredictedLabel { get; private set; }
            internal string ImagePath { get; private set; }
            internal string FileName { get; private set; }
            internal bool IsCorrect { get; private set; }
            internal float HighScore { get; private set; }
            internal List<SubResult> LabelsScore { get; private set; }

            public ImageResult(ImagePrediction p, VBuffer<ReadOnlyMemory<char>> keyValues)
            {
                Name = p.Name;
                PredictedLabel = p.PredictedLabelValue;
                IsCorrect = Name.Equals(PredictedLabel);
                ImagePath = p.ImagePath;
                FileName = Path.GetFileName(ImagePath);
                LabelsScore = new List<SubResult>();
                p.Score.Select((s, i) => (Index: i, Label: keyValues.GetItemOrDefault(i), Score: s))
                .OrderByDescending(c => c.Score)
                .Take(10) // 上位 10 件(ラベル)
                .ToList()
                .ForEach(c =>
                {
                    LabelsScore.Add(new SubResult(c.Label.ToString(), c.Score));
                });
                HighScore = LabelsScore[0].fScore;
            }
        }

        public class SubResult
        {
            internal string Label { get; private set; }
            internal string sScore { get; private set; }
            internal float fScore { get; private set; }
            public SubResult(string label, float score)
            {
                Label = label;
                fScore = score;
                sScore = $"{score:P}";
            }
        }

        public void SaveAsHTML(string savePath = "")
        {
            savePath = savePath.Length == 0 ? Resultfiles.ResultHTML : savePath;
            // HTML で評価結果を書き出し
            using (var writer = new StreamWriter(savePath))
            {
                writer.WriteLine($"<html><head><title>{Path.GetFileName(Resultfiles.ModelZip)}</title>");
                writer.WriteLine("<link rel=\"stylesheet\" href=\"https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css\" integrity=\"sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2\" crossorigin=\"anonymous\">");
                writer.WriteLine("</head><body>");

                writer.WriteLine($"<h1>Metrics for {Path.GetFileName(Resultfiles.ModelZip)}</h1>");
                // メトリックの書き出し
                writer.WriteLine("<div><table class=\"table table-striped\">");
                writer.WriteLine($"<tr><td>MicroAccuracy</td><td>{MicroAccuracy:0.000}</td></tr></tr>");
                writer.WriteLine($"<tr><td>MacroAccuracy</td><td>{MacroAccuracy:0.000}</td></tr></tr>");
                writer.WriteLine($"<tr><td>Precision</td><td>{Precision:0.000}</td></tr></tr>");
                writer.WriteLine($"<tr><td>Recall</td><td>{Recall:0.000}</td></tr></tr>");
                writer.WriteLine($"<tr><td>LogLoss</td><td>{LogLoss:0.000}</td></tr></tr>");
                writer.WriteLine($"<tr><td>LogLossReduction</td><td>{LogLossReduction:0.000}</td></tr></tr>");

                // クラス毎の適合率
                writer.WriteLine("<tr><td>PerClassPrecision</td><td>");
                foreach (var p in PerClassPrecisions)
                    writer.WriteLine($"{p.Label}: {p.Value:0.000}<br />");
                writer.WriteLine("</td></tr>");

                // クラス毎の再現率
                writer.WriteLine("<tr><td>PerClassRecall</td><td>");
                foreach (var r in PerClassRecalls)
                    writer.WriteLine($"{r.Label}: {r.Value:0.000}<br />");

                writer.WriteLine("</td></tr></table></div>");

                // 評価データ毎の分類結果
                writer.WriteLine($"<h1>Predictions</h1>");
                writer.WriteLine($"<div><table class=\"table table-bordered\">");

                int n = 1;
                foreach (var p in ImageResults)
                {
                    writer.WriteLine($"<tr><td>");
                    // 画像ファイル名
                    writer.WriteLine($"[No.{n}] {p.FileName}<br />");
                    // 正解ラベル
                    writer.WriteLine($"Actual Value: {p.Name}<br />");
                    string color = p.IsCorrect ? "green" : "red";
                    // 推論結果
                    writer.WriteLine($@"<span style=""background: linear-gradient(transparent 50%, {color} 100%);"">Predicted Value:{p.PredictedLabel}</span><br />");
                    // 画像
                    writer.WriteLine($"<img class=\"img-fluid\" src=\"{p.ImagePath}\" /></td>");
                    // クラス毎の推論結果
                    writer.WriteLine($"<td>");
                    foreach (var s in p.LabelsScore)
                    {
                        if (s.fScore >= 0.95f)
                        {
                            writer.WriteLine($@"<span style=""background: linear-gradient(transparent 50%, {color} 100%);"">{s.Label}: {s.sScore}</span><br />");
                        }
                        else
                        {
                            writer.WriteLine($"{s.Label}: {s.sScore}<br />");
                        }
                    };
                    writer.WriteLine("</td></tr>");
                    n++;
                }
                writer.WriteLine("</table></div>");
                writer.WriteLine("</body></html>");
            }
        }

    }


}
