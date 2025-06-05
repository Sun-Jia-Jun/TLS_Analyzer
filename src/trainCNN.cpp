#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <random>

#include "TLSDataProcessor.hpp"
#include "SimpleCNN.hpp"

// 优化后的超参数
const float LEARNING_RATE = 0.001f; // 适中的学习率
const int EPOCHS = 300;             // 减少训练轮数
const int BATCH_SIZE = 4;           // 小批量训练

const std::string MODEL_PATH = "../model/tls_model.bin";

int main(int argc, char *argv[])
{
    try
    {
        bool continue_training = false;
        if (argc > 1 && (std::string(argv[1]) == "--continue" || std::string(argv[1]) == "-c"))
        {
            continue_training = true;
        }

        std::cout << "============= TLS Traffic Classification =============" << std::endl;

        // 加载和预处理数据
        std::cout << "[INFO] Loading and preprocessing data..." << std::endl;
        TLSDataProcessor data_processor("../output/tls_features.csv");

        int feature_dim = data_processor.get_feature_dim();
        int num_labels = data_processor.get_num_labels();

        std::cout << "[INFO] Flattened Feature dimension: " << feature_dim << std::endl;
        std::cout << "[INFO] Number of classes: " << num_labels << std::endl;

        // 创建模型
        SimpleCNN model(feature_dim, num_labels);

        // 尝试加载已有模型
        if (continue_training)
        {
            try
            {
                model = SimpleCNN::load_model(MODEL_PATH, feature_dim, num_labels);
                std::cout << "[INFO] Continuing from existing model" << std::endl;
            }
            catch (const std::exception &e)
            {
                std::cout << "[INFO] Starting with new model: " << e.what() << std::endl;
            }
        }

        const auto &train_samples = data_processor.get_train_samples();
        const auto &test_samples = data_processor.get_test_samples();

        std::cout << "[INFO] Starting training..." << std::endl;

        float learning_rate = LEARNING_RATE;
        float best_test_acc = 0.0f;
        int patience = 0;
        const int max_patience = 30;

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int epoch = 0; epoch < EPOCHS; ++epoch)
        {
            float epoch_loss = 0.0f;
            int num_batches = 0;

            // 随机打乱训练数据
            std::vector<Sample> shuffled_samples = train_samples;
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(shuffled_samples.begin(), shuffled_samples.end(), g);

            // 批量训练
            for (size_t i = 0; i < shuffled_samples.size(); i += BATCH_SIZE)
            {
                std::vector<Sample> batch;
                size_t batch_end = std::min(i + BATCH_SIZE, shuffled_samples.size());
                batch.assign(shuffled_samples.begin() + i, shuffled_samples.begin() + batch_end);

                float batch_loss = model.train_batch(batch, learning_rate);

                if (!std::isnan(batch_loss) && !std::isinf(batch_loss) && batch_loss < 10.0f)
                {
                    epoch_loss += batch_loss;
                    num_batches++;
                }
            }

            if (num_batches > 0)
            {
                epoch_loss /= num_batches;
            }

            // 每10轮评估一次
            if (epoch % 10 == 0 || epoch == EPOCHS - 1)
            {
                float train_acc = model.evaluate(train_samples);
                float test_acc = model.evaluate(test_samples);

                std::cout << "Epoch " << std::setw(3) << epoch + 1
                          << ", Loss: " << std::fixed << std::setprecision(4) << epoch_loss
                          << ", Train: " << std::fixed << std::setprecision(1) << (train_acc * 100) << "%"
                          << ", Test: " << std::fixed << std::setprecision(1) << (test_acc * 100) << "%"
                          << ", LR: " << std::scientific << std::setprecision(1) << learning_rate << std::endl;

                // 保存最佳模型
                if (test_acc > best_test_acc)
                {
                    best_test_acc = test_acc;
                    patience = 0;
                    model.save_model(MODEL_PATH);
                    std::cout << "[INFO] New best test accuracy: "
                              << std::fixed << std::setprecision(1) << (test_acc * 100) << "%" << std::endl;
                }
                else
                {
                    patience++;
                }

                // 早停机制
                if (patience >= max_patience)
                {
                    std::cout << "[INFO] Early stopping - no improvement for "
                              << max_patience << " evaluations" << std::endl;
                    break;
                }

                // 达到目标准确率
                if (test_acc > 0.85f && train_acc > 0.85f)
                {
                    std::cout << "[INFO] Target accuracy reached!" << std::endl;
                    break;
                }
            }

            // 学习率衰减
            if (epoch > 0 && epoch % 50 == 0)
            {
                learning_rate *= 0.8f;
                learning_rate = std::max(learning_rate, 1e-5f);
                std::cout << "[INFO] Learning rate decreased to: "
                          << std::scientific << learning_rate << std::endl;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

        std::cout << "\n========== Training Summary ==========" << std::endl;
        std::cout << "Training time: " << duration << " seconds" << std::endl;
        std::cout << "Best test accuracy: " << std::fixed << std::setprecision(1)
                  << (best_test_acc * 100) << "%" << std::endl;

        // 最终评估
        float final_train_acc = model.evaluate(train_samples);
        float final_test_acc = model.evaluate(test_samples);

        std::cout << "Final train accuracy: " << std::fixed << std::setprecision(1)
                  << (final_train_acc * 100) << "%" << std::endl;
        std::cout << "Final test accuracy: " << std::fixed << std::setprecision(1)
                  << (final_test_acc * 100) << "%" << std::endl;

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }
}