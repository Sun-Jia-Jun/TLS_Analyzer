#ifndef _TLS_DATA_PROCESSOR_HPP_
#define _TLS_DATA_PROCESSOR_HPP_

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <cmath>

// 定义样本结构
struct Sample
{
    int label;                   // 网站标签(0:bing, 1:baidu, 2:bilibili)
    std::vector<float> features; // 特征向量(包含数据包大小和方向)
};

class TLSDataProcessor
{
private:
    std::vector<Sample> samples;      // 所有样本
    std::vector<Sample> trainSamples; // 训练集
    std::vector<Sample> testSamples;  // 测试集

    int numLabels = 0;        // 标签数量
    int maxFeatureLength = 0; // 最长特征序列的长度
    int featureDim = 2;       // 特征维度(数据包大小、方向)
    float testRatio = 0.2;    // 测试集占比

public:
    TLSDataProcessor(const std::string &csvPath)
    {
        loadData(csvPath);
        findMaxFeatureLength();
        shuffleAndSplit();
    }

    // 获取特征维度(经过填充后)
    int getFeatureDim() const
    {
        return maxFeatureLength * featureDim;
    }

    // 获取标签数量
    int getNumLabels() const
    {
        return numLabels;
    }

    // 获取训练集
    const std::vector<Sample> &getTrainSamples() const
    {
        return trainSamples;
    }

    // 获取测试集
    const std::vector<Sample> &getTestSamples() const
    {
        return testSamples;
    }

private:
    // 加载CSV数据
    void loadData(const std::string &csvPath)
    {
        std::ifstream file(csvPath);
        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open file: " + csvPath);
        }

        std::string line;
        // 跳过标题行
        std::getline(file, line);

        // 逐行读取数据
        while (std::getline(file, line))
        {
            std::istringstream iss(line);
            std::string labelStr, featureStr;

            // 解析标签和特征字符串
            if (std::getline(iss, labelStr, ',') && std::getline(iss, featureStr))
            {
                Sample sample;
                sample.label = std::stoi(labelStr);
                numLabels = std::max(numLabels, sample.label + 1);

                // 解析特征字符串 (格式: 大小_方向;大小_方向;...)
                std::istringstream featureStream(featureStr);
                std::string pair;

                while (std::getline(featureStream, pair, ';'))
                {
                    size_t delimPos = pair.find('_');
                    if (delimPos != std::string::npos)
                    {
                        int size = std::stoi(pair.substr(0, delimPos));
                        int direction = std::stoi(pair.substr(delimPos + 1));

                        // 归一化数据包大小 (通常在1-1500字节之间)
                        float normalizedSize = static_cast<float>(size) / 1500.0f;

                        // 添加到特征向量
                        sample.features.push_back(normalizedSize);
                        sample.features.push_back(static_cast<float>(direction));
                    }
                }

                samples.push_back(sample);
            }
        }

        std::cout << "[INFO] Loaded " << samples.size() << " samples with "
                  << numLabels << " unique labels." << std::endl;
    }

    // 查找最长特征序列长度并填充
    void findMaxFeatureLength()
    {
        for (const auto &sample : samples)
        {
            maxFeatureLength = std::max(maxFeatureLength,
                                        static_cast<int>(sample.features.size() / featureDim));
        }

        std::cout << "[INFO] Max feature length: " << maxFeatureLength
                  << " pairs (= " << maxFeatureLength * featureDim
                  << " float values)" << std::endl;

        // 对所有样本进行填充
        for (auto &sample : samples)
        {
            int currentLength = sample.features.size() / featureDim;
            if (currentLength < maxFeatureLength)
            {
                // 填充0
                sample.features.resize(maxFeatureLength * featureDim, 0.0f);
            }
        }
    }

    // 随机打乱并分割训练集/测试集
    void shuffleAndSplit()
    {
        // 随机打乱样本
        auto rng = std::default_random_engine(std::random_device{}());
        std::shuffle(samples.begin(), samples.end(), rng);

        // 分割训练集和测试集
        size_t testSize = static_cast<size_t>(samples.size() * testRatio);
        size_t trainSize = samples.size() - testSize;

        trainSamples.assign(samples.begin(), samples.begin() + trainSize);
        testSamples.assign(samples.begin() + trainSize, samples.end());

        std::cout << "[INFO] Split data into " << trainSamples.size()
                  << " training samples and " << testSamples.size()
                  << " test samples." << std::endl;
    }
};

#endif // _TLS_DATA_PROCESSOR_HPP_