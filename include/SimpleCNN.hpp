#ifndef _SIMPLE_CNN_HPP_
#define _SIMPLE_CNN_HPP_

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <chrono>

#include "TLSDataProcessor.hpp"

// 激活函数工具类
class Activation
{
public:
    // ReLU前向传播
    static std::vector<float> relu(const std::vector<float> &x)
    {
        std::vector<float> result(x.size());
        for (size_t i = 0; i < x.size(); ++i)
        {
            result[i] = std::max(0.0f, x[i]);
        }
        return result;
    }

    // Softmax激活 - 增强数值稳定性
    static std::vector<float> softmax(const std::vector<float> &x)
    {
        std::vector<float> result(x.size());

        // 为了数值稳定性，减去最大值
        float max_val = *std::max_element(x.begin(), x.end());

        float sum = 0.0f;
        for (size_t i = 0; i < x.size(); ++i)
        {
            result[i] = std::exp(std::min(x[i] - max_val, 80.0f)); // 限制指数值
            sum += result[i];
        }

        // 归一化，防止除零
        sum = std::max(sum, 1e-7f);
        for (size_t i = 0; i < x.size(); ++i)
        {
            result[i] /= sum;
        }

        return result;
    }
};

// 全连接层
class FCLayer
{
private:
    int input_size;
    int output_size;

    std::vector<std::vector<float>> weights;
    std::vector<float> biases;

    std::vector<float> input;
    std::vector<float> output;

public:
    // 在FCLayer构造函数中使用更保守的初始化
    FCLayer(int input_size, int output_size) : input_size(input_size), output_size(output_size)
    {
        // 使用更小的初始化方差
        float scale = std::sqrt(1.0f / input_size); // Xavier初始化，比He初始化更保守

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, scale);

        // 初始化权重
        weights.resize(output_size, std::vector<float>(input_size));
        for (int o = 0; o < output_size; ++o)
        {
            for (int i = 0; i < input_size; ++i)
            {
                weights[o][i] = dist(gen);
            }
        }

        // 初始化偏置为0
        biases.resize(output_size, 0.0f);
    }

    // 前向传播
    std::vector<float> forward(const std::vector<float> &input)
    {
        this->input = input;
        output.resize(output_size);

        for (int o = 0; o < output_size; ++o)
        {
            float sum = biases[o];
            for (int i = 0; i < input_size; ++i)
            {
                sum += weights[o][i] * input[i];
            }
            output[o] = sum;
        }

        return output;
    }

    // 反向传播
    std::vector<float> backward(const std::vector<float> &gradient, float learning_rate)
    {
        std::vector<float> input_gradient(input_size, 0.0f);

        // 更新权重和偏置
        for (int o = 0; o < output_size; ++o)
        {
            for (int i = 0; i < input_size; ++i)
            {
                weights[o][i] -= learning_rate * gradient[o] * input[i];
                input_gradient[i] += gradient[o] * weights[o][i];
            }
            biases[o] -= learning_rate * gradient[o];
        }

        return input_gradient;
    }

    // 权重访问方法
    const std::vector<std::vector<float>> &get_weights() const { return weights; }
    const std::vector<float> &get_biases() const { return biases; }
    std::vector<std::vector<float>> &get_mutable_weights() { return weights; }
    std::vector<float> &get_mutable_biases() { return biases; }

    int get_input_size() const { return input_size; }
    int get_output_size() const { return output_size; }
};

// 简化的神经网络（只使用全连接层）
class SimpleCNN
{
private:
    int input_dim;
    int num_labels;

    // 更小的网络结构
    FCLayer fc1;
    FCLayer fc2;

    // 存储中间激活值
    std::vector<float> fc1_input, fc1_output;

public:
    SimpleCNN(int input_dim, int num_labels)
        : input_dim(input_dim), num_labels(num_labels),
          fc1(input_dim, 16), // 大幅减少隐藏层神经元：352 -> 16
          fc2(16, num_labels) // 16 -> 4
    {
        std::cout << "[INFO] Simplified Neural Network:" << std::endl;
        std::cout << "  Input: " << input_dim << std::endl;
        std::cout << "  Hidden: " << input_dim << " -> 16" << std::endl;
        std::cout << "  Output: 16 -> " << num_labels << std::endl;
    }

    // 前向传播
    std::vector<float> forward(const std::vector<float> &input)
    {
        // 输入验证
        for (float val : input)
        {
            if (std::isnan(val) || std::isinf(val))
            {
                throw std::runtime_error("Invalid input detected");
            }
        }

        // 第一层：input -> fc1 -> relu
        fc1_input = input;
        auto fc1_raw = fc1.forward(input);
        fc1_output = Activation::relu(fc1_raw);

        // 输出层：fc1_output -> fc2 -> softmax
        auto logits = fc2.forward(fc1_output);
        return Activation::softmax(logits);
    }

    // 损失计算
    float compute_loss(const std::vector<float> &output, int label)
    {
        if (label < 0 || label >= static_cast<int>(output.size()))
        {
            throw std::runtime_error("Invalid label: " + std::to_string(label));
        }

        float prob = std::max(output[label], 1e-7f); // 避免log(0)
        float loss = -std::log(prob);

        // 限制损失值防止爆炸
        return std::min(loss, 10.0f);
    }

    // 改进的训练方法
    float train_batch(const std::vector<Sample> &batch, float learning_rate)
    {
        float total_loss = 0.0f;
        int valid_samples = 0;

        for (const auto &sample : batch)
        {
            try
            {
                // 前向传播
                auto output = forward(sample.features);

                // 计算损失
                float loss = compute_loss(output, sample.label);

                // 严格的损失检查
                if (std::isnan(loss) || std::isinf(loss) || loss > 5.0f)
                {
                    std::cout << "[WARNING] Skipping sample with loss: " << loss << std::endl;
                    continue;
                }

                total_loss += loss;
                valid_samples++;

                // 计算输出层梯度
                std::vector<float> gradient(num_labels, 0.0f);
                for (int i = 0; i < num_labels; ++i)
                {
                    gradient[i] = output[i];
                }
                gradient[sample.label] -= 1.0f;

                // 严格的梯度裁剪
                clip_gradients(gradient, 1.0f);

                // 简化的反向传播（只有两层）
                auto fc2_grad = fc2.backward(gradient, learning_rate);
                clip_gradients(fc2_grad, 1.0f);

                auto fc1_relu_grad = apply_relu_gradient(fc2_grad, fc1_output);
                fc1.backward(fc1_relu_grad, learning_rate);
            }
            catch (const std::exception &e)
            {
                std::cout << "[WARNING] Error in sample: " << e.what() << std::endl;
                continue;
            }
        }

        return valid_samples > 0 ? total_loss / valid_samples : 0.0f;
    }

    // 模型评估
    float evaluate(const std::vector<Sample> &samples)
    {
        int correct = 0;
        int total = 0;

        for (const auto &sample : samples)
        {
            try
            {
                auto output = forward(sample.features);
                int predicted = std::max_element(output.begin(), output.end()) - output.begin();
                if (predicted == sample.label)
                {
                    correct++;
                }
                total++;
            }
            catch (const std::exception &e)
            {
                // 跳过无法预测的样本
                continue;
            }
        }

        return total > 0 ? static_cast<float>(correct) / total : 0.0f;
    }

    // 修复模型保存（只保存两层）
    void save_model(const std::string &path)
    {
        std::ofstream ofs(path, std::ios::binary);
        if (!ofs.is_open())
        {
            throw std::runtime_error("Failed to save model to: " + path);
        }

        // 保存配置
        ofs.write(reinterpret_cast<const char *>(&input_dim), sizeof(input_dim));
        ofs.write(reinterpret_cast<const char *>(&num_labels), sizeof(num_labels));

        // 只保存两层
        save_fc_weights(ofs, fc1);
        save_fc_weights(ofs, fc2);

        ofs.close();
        std::cout << "[INFO] Model saved to " << path << std::endl;
    }

    // 修复模型加载（只加载两层）
    static SimpleCNN load_model(const std::string &path, int input_dim, int num_labels)
    {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs.is_open())
        {
            std::cout << "[WARNING] Model file not found, creating new model" << std::endl;
            return SimpleCNN(input_dim, num_labels);
        }

        // 读取配置
        int saved_input_dim, saved_num_labels;
        ifs.read(reinterpret_cast<char *>(&saved_input_dim), sizeof(saved_input_dim));
        ifs.read(reinterpret_cast<char *>(&saved_num_labels), sizeof(saved_num_labels));

        if (saved_input_dim != input_dim || saved_num_labels != num_labels)
        {
            std::cout << "[WARNING] Model dimension mismatch, creating new model" << std::endl;
            ifs.close();
            return SimpleCNN(input_dim, num_labels);
        }

        SimpleCNN model(input_dim, num_labels);

        try
        {
            // 只加载两层
            load_fc_weights(ifs, model.fc1);
            load_fc_weights(ifs, model.fc2);

            ifs.close();
            std::cout << "[INFO] Model loaded from " << path << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cout << "[WARNING] Failed to load weights: " << e.what() << std::endl;
            ifs.close();
            return SimpleCNN(input_dim, num_labels);
        }

        return model;
    }

private:
    // 梯度裁剪
    void clip_gradients(std::vector<float> &gradients, float max_norm)
    {
        float norm = 0.0f;
        for (float grad : gradients)
        {
            norm += grad * grad;
        }
        norm = std::sqrt(norm);

        if (norm > max_norm)
        {
            float scale = max_norm / norm;
            for (float &grad : gradients)
            {
                grad *= scale;
            }
        }
    }

    // 应用ReLU梯度
    std::vector<float> apply_relu_gradient(const std::vector<float> &upstream_grad,
                                           const std::vector<float> &activation_output)
    {
        std::vector<float> result(upstream_grad.size());
        for (size_t i = 0; i < upstream_grad.size(); ++i)
        {
            result[i] = (activation_output[i] > 0) ? upstream_grad[i] : 0.0f;
        }
        return result;
    }

    // 保存全连接层权重
    void save_fc_weights(std::ofstream &ofs, const FCLayer &layer)
    {
        const auto &weights = layer.get_weights();
        const auto &biases = layer.get_biases();

        // 保存维度信息
        int output_size = layer.get_output_size();
        int input_size = layer.get_input_size();

        ofs.write(reinterpret_cast<const char *>(&output_size), sizeof(output_size));
        ofs.write(reinterpret_cast<const char *>(&input_size), sizeof(input_size));

        // 保存权重
        for (const auto &row : weights)
        {
            for (float weight : row)
            {
                ofs.write(reinterpret_cast<const char *>(&weight), sizeof(weight));
            }
        }

        // 保存偏置
        for (float bias : biases)
        {
            ofs.write(reinterpret_cast<const char *>(&bias), sizeof(bias));
        }
    }

    // 加载全连接层权重
    static void load_fc_weights(std::ifstream &ifs, FCLayer &layer)
    {
        int output_size, input_size;
        ifs.read(reinterpret_cast<char *>(&output_size), sizeof(output_size));
        ifs.read(reinterpret_cast<char *>(&input_size), sizeof(input_size));

        // 验证维度
        if (output_size != layer.get_output_size() || input_size != layer.get_input_size())
        {
            throw std::runtime_error("Layer dimension mismatch during loading");
        }

        auto &weights = layer.get_mutable_weights();
        auto &biases = layer.get_mutable_biases();

        // 加载权重
        for (auto &row : weights)
        {
            for (float &weight : row)
            {
                ifs.read(reinterpret_cast<char *>(&weight), sizeof(weight));
            }
        }

        // 加载偏置
        for (float &bias : biases)
        {
            ifs.read(reinterpret_cast<char *>(&bias), sizeof(bias));
        }
    }
};

#endif // _SIMPLE_CNN_HPP_