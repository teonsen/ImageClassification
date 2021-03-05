using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ImageClassification.IO
{
    public static class DataLoader
    {
        public static IDataView GetShuffledDataView(MLContext mlContext, string imagesFolder)
        {
            // データフォルダの指定
            // データフォルダ下にあるフォルダ名をラベル名とする
            var dataFiles = Directory.EnumerateFiles(imagesFolder, "*", SearchOption.AllDirectories).Where(s =>
            s.EndsWith(".jpg", StringComparison.CurrentCultureIgnoreCase) || s.EndsWith(".png", StringComparison.CurrentCultureIgnoreCase));

            // データセットの作成
            var dataSet = dataFiles.Select(f => new ImageData
            {
                ImagePath = f,
                Name = Directory.GetParent(f).Name
            });

            // データのロード
            IDataView dataView = mlContext.Data.LoadFromEnumerable(dataSet);
            // データセットをシャッフル
            IDataView shuffledDataView = mlContext.Data.ShuffleRows(dataView);
            return shuffledDataView;
        }

    }
}
