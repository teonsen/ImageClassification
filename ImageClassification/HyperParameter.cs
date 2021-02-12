using System;
using System.Collections.Generic;
using System.Text;

namespace ImageClassification
{
    public class HyperParameter
    {
        /// <summary>
        /// Number of training iterations. The default value for Epoch is 200.
        /// エポック数（世代数）：学習させる回数
        /// </summary>
        public int Epoch { get; set; } = 200;

        /// <summary>
        /// Number of samples to use for mini-batch training. The default value for BatchSize is 10.
        /// バッチサイズ：１度に学習させるデータの数
        /// </summary>
        public int BatchSize { get; set; } = 10;

        /// <summary>
        /// Learning rate to use during optimization. The default value for Learning Rate is 0.01.
        /// 学習率：学習が進むスピード
        /// </summary>
        public float LearningRate { get; set; } = 0.01f;

        /// <summary>
        /// Specifies the model architecture to be used in the case of image classification training using transfer learning. The default Architecture is Resnet_v2_50.
        /// 転移学習モデルの選択
        /// </summary>
        public eTrainerArchitectures eTrainerArchitecture { get; set; } = eTrainerArchitectures.ResnetV250;

        /// <summary>
        /// The fraction of data into the test set.
        /// データセットの学習・検証/評価の分割比率。0.3の場合は70%が学習データで、残りが検証/評価データとなる。
        /// </summary>
        public double TestFraction { get; set; } = 0.3f;
    }

    public enum eTrainerArchitectures
    {
        ResnetV2101, // 0
        InceptionV3, // 1
        MobilenetV2, // 2
        ResnetV250   // 3
    }

}
