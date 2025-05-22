#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>

#include "TLSDataProcessor.hpp"
#include "SimpleCNN.hpp"

// 超参数
const float LEARNING_RATE = 0.01f;
const int EPOCHS = 50;
const int BATCH_SIZE = 32;
const std::string MODEL_PATH = "../data/tls_model.bin";

int main()
{
    try
    {
        std::cout << "========== TLS Traffic Classification using CNN ==========" << std::endl;

        // 加载并预处理数据
        std::cout << "[INFO] Loading and preprocessing data..." << std::endl;
        TLSDataProcessor dataProcessor("../data/tls_features.csv");

        int featureDim = dataProcessor.getFeatureDim();
        int numLabels = dataProcessor.getNumLabels();

        std::cout << "[INFO] Feature dimension: " << featureDim << std::endl;
        std::cout << "[INFO] Number of classes: " << numLabels << std::endl;

        // 创建CNN模型
        SimpleCNN model(featureDim, numLabels);

        // 获取训练集和测试集
        const auto &trainSamples = dataProcessor.getTrainSamples();
        const auto &testSamples = dataProcessor.getTestSamples();

        std::cout << "[INFO] Starting training..." << std::endl;

        // 训练
        auto startTime = std::chrono::high_resolution_clock::now();

        for (int epoch = 0; epoch < EPOCHS; ++epoch)
        {
            float epochLoss = 0.0f;
            int numBatches = 0;

            // 随机打乱训练样本
            std::vector<Sample> shuffledSamples = trainSamples;
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(shuffledSamples.begin(), shuffledSamples.end(), g);

            // 批次训练
            for (size_t i = 0; i < shuffledSamples.size(); i += BATCH_SIZE)
            {
                size_t batchSize = std::min(BATCH_SIZE, static_cast<int>(shuffledSamples.size() - i));
                std::vector<Sample> batch(shuffledSamples.begin() + i,
                                          shuffledSamples.begin() + i + batchSize);

                float batchLoss = model.trainBatch(batch, LEARNING_RATE);
                epochLoss += batchLoss;
                numBatches++;
            }

            epochLoss /= numBatches;

            // 评估
            float trainAcc = model.evaluate(trainSamples);
            float testAcc = model.evaluate(testSamples);

            std::cout << "Epoch " << std::setw(3) << epoch + 1
                      << ", Loss: " << std::fixed << std::setprecision(4) << epochLoss
                      << ", Train Acc: " << std::fixed << std::setprecision(2) << (trainAcc * 100) << "%"
                      << ", Test Acc: " << std::fixed << std::setprecision(2) << (testAcc * 100) << "%"
                      << std::endl;

            // 提前停止
            if (trainAcc > 0.99f && testAcc > 0.95f)
            {
                std::cout << "[INFO] Early stopping at epoch " << epoch + 1 << std::endl;
                break;
            }
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
        std::cout << "[INFO] Training completed in " << duration << " seconds." << std::endl;

        // 最终评估
        float finalTrainAcc = model.evaluate(trainSamples);
        float finalTestAcc = model.evaluate(testSamples);

        std::cout << "Final Train Accuracy: " << std::fixed << std::setprecision(2)
                  << (finalTrainAcc * 100) << "%" << std::endl;
        std::cout << "Final Test Accuracy: " << std::fixed << std::setprecision(2)
                  << (finalTestAcc * 100) << "%" << std::endl;

        // 保存模型
        model.saveModel(MODEL_PATH);

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }
}