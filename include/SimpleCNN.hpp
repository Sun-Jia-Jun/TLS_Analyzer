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

    // ReLU导数
    static std::vector<float> relu_derivative(const std::vector<float> &x)
    {
        std::vector<float> result(x.size());
        for (size_t i = 0; i < x.size(); ++i)
        {
            result[i] = (x[i] > 0) ? 1.0f : 0.0f;
        }
        return result;
    }

    // Softmax激活
    static std::vector<float> softmax(const std::vector<float> &x)
    {
        std::vector<float> result(x.size());

        // 为了数值稳定性，减去最大值
        float max_val = *std::max_element(x.begin(), x.end());

        float sum = 0.0f;
        for (size_t i = 0; i < x.size(); ++i)
        {
            result[i] = std::exp(x[i] - max_val);
            sum += result[i];
        }

        // 归一化
        for (size_t i = 0; i < x.size(); ++i)
        {
            result[i] /= sum;
        }

        return result;
    }
};

// 卷积层
class ConvLayer
{
private:
    int input_channels;  // 输入通道数
    int output_channels; // 输出通道数
    int kernel_size;     // 卷积核大小
    int stride;          // 步长
    int padding;         // 填充

    std::vector<std::vector<std::vector<float>>> weights; // 权重
    std::vector<float> biases;                            // 偏置

    // 用于存储中间结果
    std::vector<float> input;
    std::vector<float> output;

public:
    ConvLayer(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0)
        : input_channels(in_channels), output_channels(out_channels),
          kernel_size(kernel_size), stride(stride), padding(padding)
    {

        // 随机初始化权重
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 0.1f);

        // 初始化卷积核
        weights.resize(output_channels);
        for (int oc = 0; oc < output_channels; ++oc)
        {
            weights[oc].resize(input_channels);
            for (int ic = 0; ic < input_channels; ++ic)
            {
                weights[oc][ic].resize(kernel_size);
                for (int k = 0; k < kernel_size; ++k)
                {
                    weights[oc][ic][k] = dist(gen);
                }
            }
        }

        // 初始化偏置
        biases.resize(output_channels, 0.0f);
    }

    // 前向传播
    std::vector<float> forward(const std::vector<float> &input)
    {
        this->input = input;

        int input_width = input.size() / input_channels;
        int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

        // 初始化输出
        output.resize(output_channels * output_width, 0.0f);

        // 对每个输出通道进行卷积
        for (int oc = 0; oc < output_channels; ++oc)
        {
            for (int ow = 0; ow < output_width; ++ow)
            {
                float sum = biases[oc];

                // 卷积核位置
                int start_w = ow * stride - padding;

                // 对每个输入通道进行卷积
                for (int ic = 0; ic < input_channels; ++ic)
                {
                    for (int k = 0; k < kernel_size; ++k)
                    {
                        int w = start_w + k;

                        // 检查边界
                        if (w >= 0 && w < input_width)
                        {
                            int input_idx = ic * input_width + w;
                            sum += input[input_idx] * weights[oc][ic][k];
                        }
                    }
                }

                output[oc * output_width + ow] = sum;
            }
        }

        return output;
    }

    // 反向传播
    std::vector<float> backward(const std::vector<float> &gradient, float learning_rate)
    {
        int input_width = input.size() / input_channels;
        int output_width = output.size() / output_channels;

        // 初始化输入梯度
        std::vector<float> input_gradient(input.size(), 0.0f);

        // 更新卷积核权重
        for (int oc = 0; oc < output_channels; ++oc)
        {
            for (int ic = 0; ic < input_channels; ++ic)
            {
                for (int k = 0; k < kernel_size; ++k)
                {
                    float weight_gradient = 0.0f;

                    for (int ow = 0; ow < output_width; ++ow)
                    {
                        int start_w = ow * stride - padding;
                        int w = start_w + k;

                        if (w >= 0 && w < input_width)
                        {
                            int gradient_idx = oc * output_width + ow;
                            int input_idx = ic * input_width + w;
                            weight_gradient += gradient[gradient_idx] * input[input_idx];
                        }
                    }

                    weights[oc][ic][k] -= learning_rate * weight_gradient;
                }
            }
        }

        // 更新偏置
        for (int oc = 0; oc < output_channels; ++oc)
        {
            float bias_gradient = 0.0f;
            for (int ow = 0; ow < output_width; ++ow)
            {
                bias_gradient += gradient[oc * output_width + ow];
            }
            biases[oc] -= learning_rate * bias_gradient;
        }

        // 计算输入梯度（用于反向传播到前一层）
        for (int ic = 0; ic < input_channels; ++ic)
        {
            for (int iw = 0; iw < input_width; ++iw)
            {
                float g = 0.0f;

                for (int oc = 0; oc < output_channels; ++oc)
                {
                    for (int k = 0; k < kernel_size; ++k)
                    {
                        int ow = (iw + padding - k) / stride;

                        if ((iw + padding - k) % stride == 0 && ow >= 0 && ow < output_width)
                        {
                            g += gradient[oc * output_width + ow] * weights[oc][ic][k];
                        }
                    }
                }

                input_gradient[ic * input_width + iw] = g;
            }
        }

        return input_gradient;
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
    FCLayer(int input_size, int output_size) : input_size(input_size), output_size(output_size)
    {
        // Xavier初始化
        float scale = std::sqrt(6.0f / (input_size + output_size));

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-scale, scale);

        // 初始化权重
        weights.resize(output_size, std::vector<float>(input_size));
        for (int o = 0; o < output_size; ++o)
        {
            for (int i = 0; i < input_size; ++i)
            {
                weights[o][i] = dist(gen);
            }
        }

        // 初始化偏置
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
};

// 简单CNN网络
class SimpleCNN
{
private:
    int input_dim;
    int num_labels;

    ConvLayer conv1;
    FCLayer fc1;
    FCLayer fc2;

public:
    SimpleCNN(int input_dim, int num_labels)
        : input_dim(input_dim), num_labels(num_labels),
          conv1(1, 16, 5, 2),            // 输入通道1，输出通道16，卷积核大小5，步长2
          fc1((input_dim / 2) * 16, 64), // 降采样后的特征维度 * 通道数 -> 64
          fc2(64, num_labels)            // 64 -> 类别数
    {
    }

    // 前向传播
    std::vector<float> forward(const std::vector<float> &input)
    {
        // 卷积层 + ReLU
        auto conv1_out = Activation::relu(conv1.forward(input));

        // 全连接层 + ReLU
        auto fc1_out = Activation::relu(fc1.forward(conv1_out));

        // 输出层 + Softmax
        auto logits = fc2.forward(fc1_out);
        return Activation::softmax(logits);
    }

    // 计算交叉熵损失
    float compute_loss(const std::vector<float> &output, int label)
    {
        return -std::log(std::max(output[label], 1e-7f));
    }

    // 训练一个批次
    float train_batch(const std::vector<Sample> &batch, float learning_rate)
    {
        float total_loss = 0.0f;

        for (const auto &sample : batch)
        {
            // 前向传播
            auto output = forward(sample.features);

            // 计算损失
            total_loss += compute_loss(output, sample.label);

            // 计算输出层梯度
            std::vector<float> gradient(num_labels, 0.0f);
            for (int i = 0; i < num_labels; ++i)
            {
                gradient[i] = output[i];
            }
            gradient[sample.label] -= 1.0f;

            // 反向传播
            auto fc2_gradient = fc2.backward(gradient, learning_rate);
            auto fc1_gradient = fc1.backward(fc2_gradient, learning_rate);
            conv1.backward(fc1_gradient, learning_rate);
        }

        return total_loss / batch.size();
    }

    // 评估
    float evaluate(const std::vector<Sample> &samples)
    {
        int correct = 0;

        for (const auto &sample : samples)
        {
            auto output = forward(sample.features);

            // 找到概率最大的类别
            int predicted_label = std::max_element(output.begin(), output.end()) - output.begin();

            if (predicted_label == sample.label)
            {
                correct++;
            }
        }

        return static_cast<float>(correct) / samples.size();
    }

    // 保存模型
    void save_model(const std::string &filename)
    {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open file for saving model: " + filename);
        }

        // 这里简单实现，实际应该序列化所有层的参数
        file.write(reinterpret_cast<const char *>(&input_dim), sizeof(input_dim));
        file.write(reinterpret_cast<const char *>(&num_labels), sizeof(num_labels));

        // 各层参数也需要保存
        // ...

        std::cout << "[INFO] Model saved to " << filename << std::endl;
    }

    // 加载模型
    static SimpleCNN load_model(const std::string &filename)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open file for loading model: " + filename);
        }

        int input_dim, num_labels;
        file.read(reinterpret_cast<char *>(&input_dim), sizeof(input_dim));
        file.read(reinterpret_cast<char *>(&num_labels), sizeof(num_labels));

        SimpleCNN model(input_dim, num_labels);

        // 加载各层参数
        // ...

        std::cout << "[INFO] Model loaded from " << filename << std::endl;
        return model;
    }
};

#endif // _SIMPLE_CNN_HPP_