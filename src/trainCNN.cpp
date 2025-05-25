#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <random>

#include "TLSDataProcessor.hpp"
#include "SimpleCNN.hpp"

// 超参数调整
const float LEARNING_RATE = 0.01f; // 提高初始学习率
const int EPOCHS = 100;
const int BATCH_SIZE = 8; // 进一步减小批次大小

const std::string MODEL_PATH = "../model/tls_model.bin";

int main()
{
    try
    {
        std::cout << "========== TLS Traffic Classification using CNN ==========" << std::endl;

        // 加载并预处理数据
        std::cout << "[INFO] Loading and preprocessing data..." << std::endl;
        TLSDataProcessor data_processor("../output/tls_features.csv");

        int feature_dim = data_processor.get_feature_dim();
        int num_labels = data_processor.get_num_labels();

        std::cout << "[INFO] Feature dimension: " << feature_dim << std::endl;
        std::cout << "[INFO] Number of classes: " << num_labels << std::endl;

        // 创建CNN模型
        SimpleCNN model(feature_dim, num_labels);

        // 获取训练集和测试集
        const auto &train_samples = data_processor.get_train_samples();
        const auto &test_samples = data_processor.get_test_samples();

        std::cout << "[INFO] Starting training..." << std::endl;

        float learning_rate = LEARNING_RATE;
        float best_test_acc = 0.0f;
        int patience = 0;
        const int max_patience = 10;

        // 训练
        auto start_time = std::chrono::high_resolution_clock::now();

        for (int epoch = 0; epoch < EPOCHS; ++epoch)
        {
            float epoch_loss = 0.0f;
            int num_batches = 0;

            // 随机打乱训练样本
            std::vector<Sample> shuffled_samples = train_samples;
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(shuffled_samples.begin(), shuffled_samples.end(), g);

            // 批次训练
            for (size_t i = 0; i < shuffled_samples.size(); i += BATCH_SIZE)
            {
                size_t batch_end = std::min(i + BATCH_SIZE, shuffled_samples.size());
                std::vector<Sample> batch(shuffled_samples.begin() + i,
                                          shuffled_samples.begin() + batch_end);

                float batch_loss = model.train_batch(batch, learning_rate);

                // 检查NaN
                if (std::isnan(batch_loss) || std::isinf(batch_loss))
                {
                    std::cout << "[WARNING] NaN/Inf detected in batch loss at epoch " << epoch + 1 << std::endl;
                    std::cout << "[INFO] Resetting learning rate and continuing..." << std::endl;
                    learning_rate = LEARNING_RATE * 0.1f; // 大幅降低学习率
                    continue;
                }

                epoch_loss += batch_loss;
                num_batches++;
            }

            if (num_batches > 0)
            {
                epoch_loss /= num_batches;
            }

            // 每5轮评估一次
            if (epoch % 5 == 0 || epoch == EPOCHS - 1)
            {
                float train_acc = model.evaluate(train_samples);
                float test_acc = model.evaluate(test_samples);

                std::cout << "Epoch " << std::setw(3) << epoch + 1
                          << ", Loss: " << std::fixed << std::setprecision(4) << epoch_loss
                          << ", Train Acc: " << std::fixed << std::setprecision(2) << (train_acc * 100) << "%"
                          << ", Test Acc: " << std::fixed << std::setprecision(2) << (test_acc * 100) << "%"
                          << ", LR: " << std::scientific << learning_rate << std::endl;

                // 早停机制
                if (test_acc > best_test_acc)
                {
                    best_test_acc = test_acc;
                    patience = 0;
                }
                else
                {
                    patience++;
                }

                if (patience >= max_patience)
                {
                    std::cout << "[INFO] Early stopping due to no improvement" << std::endl;
                    break;
                }

                // 提前停止
                if (train_acc > 0.95f && test_acc > 0.90f)
                {
                    std::cout << "[INFO] Early stopping at epoch " << epoch + 1 << std::endl;
                    break;
                }
            }

            // 改进的学习率衰减
            if (epoch > 0 && epoch % 20 == 0)
            {
                learning_rate *= 0.9f;                          // 减缓衰减速度
                learning_rate = std::max(learning_rate, 1e-6f); // 设置最小学习率
                std::cout << "[INFO] Learning rate reduced to " << std::scientific << learning_rate << std::endl;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
        std::cout << "[INFO] Training completed in " << duration << " seconds." << std::endl;

        // 最终评估
        float final_train_acc = model.evaluate(train_samples);
        float final_test_acc = model.evaluate(test_samples);

        std::cout << "Final Train Accuracy: " << std::fixed << std::setprecision(2)
                  << (final_train_acc * 100) << "%" << std::endl;
        std::cout << "Final Test Accuracy: " << std::fixed << std::setprecision(2)
                  << (final_test_acc * 100) << "%" << std::endl;

        // 保存模型
        model.save_model(MODEL_PATH);

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }
}