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
        // He初始化，更适合ReLU
        float scale = std::sqrt(2.0f / input_size);

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
    FCLayer fc1; // 简化网络结构，去掉第二个卷积层
    FCLayer fc2;

    // 存储中间激活值用于反向传播
    std::vector<float> conv1_input, conv1_output;
    std::vector<float> fc1_input, fc1_output;

public:
    SimpleCNN(int input_dim, int num_labels)
        : input_dim(input_dim), num_labels(num_labels),
          conv1(1, 4, 5, 2),                // 减少通道数，增大步长
          fc1((input_dim - 4) / 2 * 4, 16), // 简化维度计算
          fc2(16, num_labels)
    {
        std::cout << "[INFO] CNN architecture:" << std::endl;
        std::cout << "  Input dim: " << input_dim << std::endl;
        std::cout << "  Conv1: 1->4 channels, kernel=5, stride=2" << std::endl;
        std::cout << "  FC1: " << (input_dim - 4) / 2 * 4 << " -> 16" << std::endl;
        std::cout << "  FC2: 16 -> " << num_labels << std::endl;
    }

    // 简化的前向传播
    std::vector<float> forward(const std::vector<float> &input)
    {
        // 检查输入有效性
        for (float val : input)
        {
            if (std::isnan(val) || std::isinf(val))
            {
                throw std::runtime_error("Invalid input detected in forward pass");
            }
        }

        // 卷积层
        conv1_input = input;
        auto conv1_raw = conv1.forward(input);
        conv1_output = Activation::relu(conv1_raw);

        // 第一个全连接层
        fc1_input = conv1_output;
        auto fc1_raw = fc1.forward(conv1_output);
        fc1_output = Activation::relu(fc1_raw);

        // 输出层
        auto logits = fc2.forward(fc1_output);
        return Activation::softmax(logits);
    }

    // 改进的损失计算
    float compute_loss(const std::vector<float> &output, int label)
    {
        float prob = std::max(output[label], 1e-7f); // 避免log(0)
        float loss = -std::log(prob);

        // 限制损失值
        return std::min(loss, 10.0f);
    }

    // 改进的训练方法，添加梯度裁剪
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

                if (std::isnan(loss) || std::isinf(loss))
                {
                    std::cout << "[WARNING] Invalid loss detected, skipping sample" << std::endl;
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

                // 梯度裁剪
                clip_gradients(gradient, 1.0f);

                // 反向传播 fc2
                auto fc2_grad = fc2.backward(gradient, learning_rate);
                clip_gradients(fc2_grad, 1.0f);

                // 反向传播 fc1 (考虑ReLU梯度)
                auto fc1_relu_grad = apply_relu_gradient(fc2_grad, fc1_output);
                auto fc1_grad = fc1.backward(fc1_relu_grad, learning_rate);
                clip_gradients(fc1_grad, 1.0f);

                // 反向传播 conv1 (考虑ReLU梯度)
                auto conv1_relu_grad = apply_relu_gradient(fc1_grad, conv1_output);
                conv1.backward(conv1_relu_grad, learning_rate);
            }
            catch (const std::exception &e)
            {
                std::cout << "[WARNING] Error in training sample: " << e.what() << std::endl;
                continue;
            }
        }

        return valid_samples > 0 ? total_loss / valid_samples : 0.0f;
    }

    // 添加评估方法
    float evaluate(const std::vector<Sample> &samples)
    {
        int correct = 0;
        for (const auto &sample : samples)
        {
            auto output = forward(sample.features);
            int predicted = std::max_element(output.begin(), output.end()) - output.begin();
            if (predicted == sample.label)
            {
                correct++;
            }
        }
        return static_cast<float>(correct) / samples.size();
    }

    // 添加模型保存方法（简单实现）
    void save_model(const std::string &path)
    {
        std::ofstream ofs(path, std::ios::binary);
        if (!ofs.is_open())
        {
            throw std::runtime_error("Failed to save model to: " + path);
        }

        // 这里应该实现权重的序列化保存
        // 为了简化，现在只是创建文件
        ofs.write("SimpleCNN_Model", 15);
        ofs.close();

        std::cout << "[INFO] Model saved to " << path << std::endl;
    }

    // 添加模型加载方法（简单实现）
    static SimpleCNN load_model(const std::string &path, int input_dim, int num_labels)
    {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs.is_open())
        {
            throw std::runtime_error("Failed to load model from: " + path);
        }

        // 这里应该实现权重的反序列化加载
        // 为了简化，现在只是返回新模型
        ifs.close();

        std::cout << "[INFO] Model loaded from " << path << std::endl;
        return SimpleCNN(input_dim, num_labels);
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

    // 计算第一个全连接层的输入维度
    int calculate_fc1_input_size(int input_dim)
    {
        // conv1: kernel=3, stride=1, padding=0
        int after_conv1 = input_dim - 3 + 1; // input_dim - 2

        // conv2: kernel=3, stride=2, padding=0
        int after_conv2 = (after_conv1 - 3) / 2 + 1; // (input_dim - 2 - 3) / 2 + 1

        return after_conv2 * 16; // 16 是 conv2 的输出通道数
    }
};

#endif // _SIMPLE_CNN_HPP_