using System;
using System.Collections.Generic;
using System.Text;

namespace ImageClassification
{
    public class HyperParameter
    {
        /// <summary>
        /// エポック数（世代数）：学習させる回数
        /// </summary>
        public int Epoch { get; set; } = 200;

        /// <summary>
        /// バッチサイズ：１度に学習させるデータの数
        /// </summary>
        public int BatchSize { get; set; } = 10;

        /// <summary>
        /// 学習率：学習が進むスピード
        /// </summary>
        public float LearningRate { get; set; } = 0.01f;

        public eTrainerArchitectures eTrainerArchitecture { get; set; } = eTrainerArchitectures.ResnetV250;
    }

    public enum eTrainerArchitectures
    {
        ResnetV2101, // 0
        InceptionV3, // 1
        MobilenetV2, // 2
        ResnetV250   // 3 転移学習
    }

}
