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
    static std::vector<float> reluDerivative(const std::vector<float> &x)
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
        float maxVal = *std::max_element(x.begin(), x.end());

        float sum = 0.0f;
        for (size_t i = 0; i < x.size(); ++i)
        {
            result[i] = std::exp(x[i] - maxVal);
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
    int inputChannels;  // 输入通道数
    int outputChannels; // 输出通道数
    int kernelSize;     // 卷积核大小
    int stride;         // 步长
    int padding;        // 填充

    std::vector<std::vector<std::vector<float>>> weights; // 权重
    std::vector<float> biases;                            // 偏置

    // 用于存储中间结果
    std::vector<float> input;
    std::vector<float> output;

public:
    ConvLayer(int inChannels, int outChannels, int kernelSize, int stride = 1, int padding = 0)
        : inputChannels(inChannels), outputChannels(outChannels),
          kernelSize(kernelSize), stride(stride), padding(padding)
    {

        // 随机初始化权重
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 0.1f);

        // 初始化卷积核
        weights.resize(outputChannels);
        for (int oc = 0; oc < outputChannels; ++oc)
        {
            weights[oc].resize(inputChannels);
            for (int ic = 0; ic < inputChannels; ++ic)
            {
                weights[oc][ic].resize(kernelSize);
                for (int k = 0; k < kernelSize; ++k)
                {
                    weights[oc][ic][k] = dist(gen);
                }
            }
        }

        // 初始化偏置
        biases.resize(outputChannels, 0.0f);
    }

    // 前向传播
    std::vector<float> forward(const std::vector<float> &input)
    {
        this->input = input;

        int inputWidth = input.size() / inputChannels;
        int outputWidth = (inputWidth + 2 * padding - kernelSize) / stride + 1;

        // 初始化输出
        output.resize(outputChannels * outputWidth, 0.0f);

        // 对每个输出通道进行卷积
        for (int oc = 0; oc < outputChannels; ++oc)
        {
            for (int ow = 0; ow < outputWidth; ++ow)
            {
                float sum = biases[oc];

                // 卷积核位置
                int startW = ow * stride - padding;

                // 对每个输入通道进行卷积
                for (int ic = 0; ic < inputChannels; ++ic)
                {
                    for (int k = 0; k < kernelSize; ++k)
                    {
                        int w = startW + k;

                        // 检查边界
                        if (w >= 0 && w < inputWidth)
                        {
                            int inputIdx = ic * inputWidth + w;
                            sum += input[inputIdx] * weights[oc][ic][k];
                        }
                    }
                }

                output[oc * outputWidth + ow] = sum;
            }
        }

        return output;
    }

    // 反向传播
    std::vector<float> backward(const std::vector<float> &gradient, float learningRate)
    {
        int inputWidth = input.size() / inputChannels;
        int outputWidth = output.size() / outputChannels;

        // 初始化输入梯度
        std::vector<float> inputGradient(input.size(), 0.0f);

        // 更新卷积核权重
        for (int oc = 0; oc < outputChannels; ++oc)
        {
            for (int ic = 0; ic < inputChannels; ++ic)
            {
                for (int k = 0; k < kernelSize; ++k)
                {
                    float weightGradient = 0.0f;

                    for (int ow = 0; ow < outputWidth; ++ow)
                    {
                        int startW = ow * stride - padding;
                        int w = startW + k;

                        if (w >= 0 && w < inputWidth)
                        {
                            int gradientIdx = oc * outputWidth + ow;
                            int inputIdx = ic * inputWidth + w;
                            weightGradient += gradient[gradientIdx] * input[inputIdx];
                        }
                    }

                    weights[oc][ic][k] -= learningRate * weightGradient;
                }
            }
        }

        // 更新偏置
        for (int oc = 0; oc < outputChannels; ++oc)
        {
            float biasGradient = 0.0f;
            for (int ow = 0; ow < outputWidth; ++ow)
            {
                biasGradient += gradient[oc * outputWidth + ow];
            }
            biases[oc] -= learningRate * biasGradient;
        }

        // 计算输入梯度（用于反向传播到前一层）
        for (int ic = 0; ic < inputChannels; ++ic)
        {
            for (int iw = 0; iw < inputWidth; ++iw)
            {
                float g = 0.0f;

                for (int oc = 0; oc < outputChannels; ++oc)
                {
                    for (int k = 0; k < kernelSize; ++k)
                    {
                        int ow = (iw + padding - k) / stride;

                        if ((iw + padding - k) % stride == 0 && ow >= 0 && ow < outputWidth)
                        {
                            g += gradient[oc * outputWidth + ow] * weights[oc][ic][k];
                        }
                    }
                }

                inputGradient[ic * inputWidth + iw] = g;
            }
        }

        return inputGradient;
    }
};

// 全连接层
class FCLayer
{
private:
    int inputSize;
    int outputSize;

    std::vector<std::vector<float>> weights;
    std::vector<float> biases;

    std::vector<float> input;
    std::vector<float> output;

public:
    FCLayer(int inputSize, int outputSize) : inputSize(inputSize), outputSize(outputSize)
    {
        // Xavier初始化
        float scale = std::sqrt(6.0f / (inputSize + outputSize));

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-scale, scale);

        // 初始化权重
        weights.resize(outputSize, std::vector<float>(inputSize));
        for (int o = 0; o < outputSize; ++o)
        {
            for (int i = 0; i < inputSize; ++i)
            {
                weights[o][i] = dist(gen);
            }
        }

        // 初始化偏置
        biases.resize(outputSize, 0.0f);
    }

    // 前向传播
    std::vector<float> forward(const std::vector<float> &input)
    {
        this->input = input;
        output.resize(outputSize);

        for (int o = 0; o < outputSize; ++o)
        {
            float sum = biases[o];
            for (int i = 0; i < inputSize; ++i)
            {
                sum += weights[o][i] * input[i];
            }
            output[o] = sum;
        }

        return output;
    }

    // 反向传播
    std::vector<float> backward(const std::vector<float> &gradient, float learningRate)
    {
        std::vector<float> inputGradient(inputSize, 0.0f);

        // 更新权重和偏置
        for (int o = 0; o < outputSize; ++o)
        {
            for (int i = 0; i < inputSize; ++i)
            {
                weights[o][i] -= learningRate * gradient[o] * input[i];
                inputGradient[i] += gradient[o] * weights[o][i];
            }
            biases[o] -= learningRate * gradient[o];
        }

        return inputGradient;
    }
};

// 简单CNN网络
class SimpleCNN
{
private:
    int inputDim;
    int numLabels;

    ConvLayer conv1;
    FCLayer fc1;
    FCLayer fc2;

public:
    SimpleCNN(int inputDim, int numLabels)
        : inputDim(inputDim), numLabels(numLabels),
          conv1(1, 16, 5, 2),           // 输入通道1，输出通道16，卷积核大小5，步长2
          fc1((inputDim / 2) * 16, 64), // 降采样后的特征维度 * 通道数 -> 64
          fc2(64, numLabels)            // 64 -> 类别数
    {
    }

    // 前向传播
    std::vector<float> forward(const std::vector<float> &input)
    {
        // 卷积层 + ReLU
        auto conv1Out = Activation::relu(conv1.forward(input));

        // 全连接层 + ReLU
        auto fc1Out = Activation::relu(fc1.forward(conv1Out));

        // 输出层 + Softmax
        auto logits = fc2.forward(fc1Out);
        return Activation::softmax(logits);
    }

    // 计算交叉熵损失
    float computeLoss(const std::vector<float> &output, int label)
    {
        return -std::log(std::max(output[label], 1e-7f));
    }

    // 训练一个批次
    float trainBatch(const std::vector<Sample> &batch, float learningRate)
    {
        float totalLoss = 0.0f;

        for (const auto &sample : batch)
        {
            // 前向传播
            auto output = forward(sample.features);

            // 计算损失
            totalLoss += computeLoss(output, sample.label);

            // 计算输出层梯度
            std::vector<float> gradient(numLabels, 0.0f);
            for (int i = 0; i < numLabels; ++i)
            {
                gradient[i] = output[i];
            }
            gradient[sample.label] -= 1.0f;

            // 反向传播
            auto fc2Gradient = fc2.backward(gradient, learningRate);
            auto fc1Gradient = fc1.backward(fc2Gradient, learningRate);
            conv1.backward(fc1Gradient, learningRate);
        }

        return totalLoss / batch.size();
    }

    // 评估
    float evaluate(const std::vector<Sample> &samples)
    {
        int correct = 0;

        for (const auto &sample : samples)
        {
            auto output = forward(sample.features);

            // 找到概率最大的类别
            int predictedLabel = std::max_element(output.begin(), output.end()) - output.begin();

            if (predictedLabel == sample.label)
            {
                correct++;
            }
        }

        return static_cast<float>(correct) / samples.size();
    }

    // 保存模型
    void saveModel(const std::string &filename)
    {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open file for saving model: " + filename);
        }

        // 这里简单实现，实际应该序列化所有层的参数
        file.write(reinterpret_cast<const char *>(&inputDim), sizeof(inputDim));
        file.write(reinterpret_cast<const char *>(&numLabels), sizeof(numLabels));

        // 各层参数也需要保存
        // ...

        std::cout << "[INFO] Model saved to " << filename << std::endl;
    }

    // 加载模型
    static SimpleCNN loadModel(const std::string &filename)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open file for loading model: " + filename);
        }

        int inputDim, numLabels;
        file.read(reinterpret_cast<char *>(&inputDim), sizeof(inputDim));
        file.read(reinterpret_cast<char *>(&numLabels), sizeof(numLabels));

        SimpleCNN model(inputDim, numLabels);

        // 加载各层参数
        // ...

        std::cout << "[INFO] Model loaded from " << filename << std::endl;
        return model;
    }
};

#endif // _SIMPLE_CNN_HPP_