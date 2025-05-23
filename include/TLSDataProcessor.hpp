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
    int label;                   // 网站标签 (e.g. 0:bing, 1:baidu, 2:bilibili ...)
    std::vector<float> features; // 特征向量 包含数据包大小和方向)
};

class TLSDataProcessor
{
private:
    std::vector<Sample> samples;       // 所有样本
    std::vector<Sample> train_samples; // 训练集
    std::vector<Sample> test_samples;  // 测试集

    int num_labels = 0;         // 标签数量
    int max_feature_length = 0; // 最长特征序列的长度
    int feature_dim = 2;        // 特征维度(数据包大小、方向)
    float test_ratio = 0.2;     // 测试集占比

public:
    TLSDataProcessor(const std::string &csv_path)
    {
        load_data(csv_path);
        find_max_feature_length();
        shuffle_and_split();
    }

    // 获取特征维度(经过填充后)
    int get_feature_dim() const
    {
        return max_feature_length * feature_dim;
    }

    // 获取标签数量
    int get_num_labels() const
    {
        return num_labels;
    }

    // 获取训练集
    const std::vector<Sample> &get_train_samples() const
    {
        return train_samples;
    }

    // 获取测试集
    const std::vector<Sample> &get_test_samples() const
    {
        return test_samples;
    }

private:
    // 加载CSV数据
    void load_data(const std::string &csv_path)
    {
        std::ifstream ifs(csv_path);
        if (!ifs.is_open())
        {
            throw std::runtime_error("Failed to open file: " + csv_path);
        }

        std::string line;
        // 跳过第一行，是列名
        std::getline(ifs, line);

        // 逐行读取数据
        while (std::getline(ifs, line))
        {
            std::istringstream iss(line);
            std::string label_str, feature_str;

            // 解析标签和特征字符串 0,xxxxx
            if (std::getline(iss, label_str, ',') && std::getline(iss, feature_str))
            {
                Sample sample;
                sample.label = std::stoi(label_str);
                num_labels = std::max(num_labels, sample.label + 1);

                // 解析特征字符串 e.g. 387_0;1492_1;1000_1;198_0;298_1;233_0;1492_1;169_1
                std::istringstream feature_stream(feature_str);
                std::string pair;

                while (std::getline(feature_stream, pair, ';'))
                {
                    size_t delim_pos = pair.find('_');
                    if (delim_pos != std::string::npos)
                    {
                        int size = std::stoi(pair.substr(0, delim_pos));
                        int direction = std::stoi(pair.substr(delim_pos + 1));

                        // 归一化数据包大小 (通常在1-1500字节之间)
                        float normalized_size = static_cast<float>(size) / 1500.0f;

                        // 添加到特征向量
                        sample.features.push_back(normalized_size);
                        sample.features.push_back(static_cast<float>(direction));
                    }
                }

                samples.push_back(sample);
            }
        }

        std::cout << "[INFO] Loaded " << samples.size() << " samples with "
                  << num_labels << " unique labels." << std::endl;
    }

    // 查找最长特征序列长度并填充
    void find_max_feature_length()
    {
        for (const auto &sample : samples)
        {
            max_feature_length = std::max(max_feature_length,
                                          static_cast<int>(sample.features.size() / feature_dim));
        }

        std::cout << "[INFO] Max feature length: " << max_feature_length
                  << " pairs (= " << max_feature_length * feature_dim
                  << " float values)" << std::endl;

        // 对所有样本进行填充
        for (auto &sample : samples)
        {
            int current_length = sample.features.size() / feature_dim;
            if (current_length < max_feature_length)
            {
                // 填充0
                sample.features.resize(max_feature_length * feature_dim, 0.0f);
            }
        }
    }

    // 随机打乱并分割训练集/测试集
    void shuffle_and_split()
    {
        // 随机打乱样本
        auto rng = std::default_random_engine(std::random_device{}());
        std::shuffle(samples.begin(), samples.end(), rng);

        // 分割训练集和测试集
        size_t test_size = static_cast<size_t>(samples.size() * test_ratio);
        size_t train_size = samples.size() - test_size;

        train_samples.assign(samples.begin(), samples.begin() + train_size);
        test_samples.assign(samples.begin() + train_size, samples.end());

        std::cout << "[INFO] Split data into " << train_samples.size()
                  << " training samples and " << test_samples.size()
                  << " test samples." << std::endl;
    }
};

#endif // _TLS_DATA_PROCESSOR_HPP_