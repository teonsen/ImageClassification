using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ImageClassification.IO
{
    public static class ResultHTML
    {
        public static void Save(MLContext context,
                                IDataView trainData,
                                IDataView prediction,
                                IEnumerable<ImagePrediction> predictions,
                                string modelFilePath,
                                string resultHTMLPath,
                                int resultsToShow)
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

                var resultList = new List<MainResult>();
                foreach (var p in predictions)
                {
                    resultList.Add(new MainResult(p, keyValues));
                }
                resultList.Sort((a, b) => b.HighScore.CompareTo(a.HighScore));

                int n = 1;
                foreach (var p in resultList)
                {
                    writer.WriteLine($"<tr><td>");
                    // 画像ファイル名
                    writer.WriteLine($"[No.{n}] {Path.GetFileName(p.ImagePath)}<br />");
                    // 正解ラベル
                    writer.WriteLine($"Actual Value: {p.Name}<br />");
                    string color = p.Name.Equals(p.PredictedLabel) ? "green" : "red";
                    // 推論結果
                    writer.WriteLine($@"<span style=""background: linear-gradient(transparent 50%, {color} 100%);"">Predicted Value:{p.PredictedLabel}</span><br />");
                    // 画像
                    writer.WriteLine($"<img class=\"img-fluid\" src=\"{p.ImagePath}\" /></td>");
                    // クラス毎の推論結果
                    writer.WriteLine($"<td>");
                    foreach (var s in p.PredictionList)
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
                    if (n > resultsToShow) break;
                }
                writer.WriteLine("</table></div>");
                writer.WriteLine("</body></html>");
            }
        }

        class MainResult
        {
            internal string Name { get; set; }
            internal string PredictedLabel { get; set; }
            internal string ImagePath { get; set; }
            internal string HighScore { get; set; }
            internal List<SubResult> PredictionList { get; set; }

            public MainResult(ImagePrediction p, VBuffer<ReadOnlyMemory<char>> keyValues)
            {
                Name = p.Name;
                PredictedLabel = p.PredictedLabelValue;
                ImagePath = p.ImagePath;
                PredictionList = new List<SubResult>();
                p.Score.Select((s, i) => (Index: i, Label: keyValues.GetItemOrDefault(i), Score: s))
                .OrderByDescending(c => c.Score)
                .Take(10) // 上位 10 件(ラベル)
                .ToList()
                .ForEach(c =>
                {
                    PredictionList.Add(new SubResult(c.Label.ToString(), c.Score));
                });
                HighScore = $"{PredictionList[0].fScore:P}";
            }
        }

        class SubResult
        {
            internal string Label { get; set; }
            internal string sScore { get; set; }
            internal float fScore { get; private set; }
            public SubResult(string label, float score)
            {
                Label = label;
                fScore = score;
                sScore = $"{score:P}";
            }
        }

    }
}
