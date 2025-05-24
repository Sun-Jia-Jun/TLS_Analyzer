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
#include <numeric>
#include <iostream>

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
    // 改进的数据加载，添加数据增强和标准化
    void load_data(const std::string &csv_path)
    {
        std::ifstream ifs(csv_path);
        if (!ifs.is_open())
        {
            throw std::runtime_error("Failed to open file: " + csv_path);
        }

        std::string line;
        std::getline(ifs, line); // 跳过列名

        std::unordered_map<int, int> label_counts;

        while (std::getline(ifs, line))
        {
            std::istringstream iss(line);
            std::string label_str, feature_str;

            if (std::getline(iss, label_str, ',') && std::getline(iss, feature_str))
            {
                Sample sample;
                sample.label = std::stoi(label_str);
                label_counts[sample.label]++;
                num_labels = std::max(num_labels, sample.label + 1);

                // 解析特征并添加统计特征
                parse_features_with_stats(feature_str, sample);
                samples.push_back(sample);
            }
        }

        // 打印数据分布
        std::cout << "[INFO] Label distribution:" << std::endl;
        for (const auto &pair : label_counts)
        {
            std::cout << "  Label " << pair.first << ": " << pair.second << " samples" << std::endl;
        }

        // 数据增强以平衡类别
        balance_dataset();
    }

    // 解析特征并添加统计信息
    void parse_features_with_stats(const std::string &feature_str, Sample &sample)
    {
        std::istringstream feature_stream(feature_str);
        std::string pair;

        std::vector<float> sizes, directions;

        while (std::getline(feature_stream, pair, ';'))
        {
            size_t delim_pos = pair.find('_');
            if (delim_pos != std::string::npos)
            {
                int size = std::stoi(pair.substr(0, delim_pos));
                int direction = std::stoi(pair.substr(delim_pos + 1));

                // 改进的归一化：使用对数变换
                float normalized_size = std::log(static_cast<float>(size) + 1.0f) / std::log(1501.0f);

                sample.features.push_back(normalized_size);
                sample.features.push_back(static_cast<float>(direction));

                sizes.push_back(normalized_size);
                directions.push_back(static_cast<float>(direction));
            }
        }

        // 添加统计特征
        if (!sizes.empty())
        {
            // 包大小统计
            float avg_size = std::accumulate(sizes.begin(), sizes.end(), 0.0f) / sizes.size();
            float max_size = *std::max_element(sizes.begin(), sizes.end());
            float min_size = *std::min_element(sizes.begin(), sizes.end());

            // 方向统计
            float outgoing_ratio = std::count(directions.begin(), directions.end(), 1.0f) / static_cast<float>(directions.size());

            // 添加到特征末尾
            sample.features.insert(sample.features.end(), {avg_size, max_size, min_size, outgoing_ratio});
        }
    }

    // 平衡数据集
    void balance_dataset()
    {
        std::unordered_map<int, std::vector<Sample>> label_samples;
        for (const auto &sample : samples)
        {
            label_samples[sample.label].push_back(sample);
        }

        // 找到最大类别的样本数
        size_t max_samples = 0;
        for (const auto &pair : label_samples)
        {
            max_samples = std::max(max_samples, pair.second.size());
        }

        // 对小类别进行数据增强
        std::vector<Sample> balanced_samples;
        std::random_device rd;
        std::mt19937 gen(rd());

        for (const auto &pair : label_samples)
        {
            const auto &class_samples = pair.second;
            balanced_samples.insert(balanced_samples.end(), class_samples.begin(), class_samples.end());

            // 如果样本数不足，进行增强
            if (class_samples.size() < max_samples)
            {
                size_t needed = max_samples - class_samples.size();
                std::uniform_int_distribution<> dis(0, class_samples.size() - 1);

                for (size_t i = 0; i < needed; ++i)
                {
                    // 随机选择一个样本进行轻微变换
                    Sample augmented = class_samples[dis(gen)];
                    add_noise(augmented, gen);
                    balanced_samples.push_back(augmented);
                }
            }
        }

        samples = std::move(balanced_samples);
        std::cout << "[INFO] Balanced dataset to " << samples.size() << " samples" << std::endl;
    }

    // 添加噪声进行数据增强
    void add_noise(Sample &sample, std::mt19937 &gen)
    {
        std::normal_distribution<float> noise(0.0f, 0.02f); // 小幅噪声

        for (size_t i = 0; i < sample.features.size(); ++i)
        {
            if (i % 2 == 0) // 只对大小特征添加噪声，不对方向添加
            {
                sample.features[i] += noise(gen);
                sample.features[i] = std::max(0.0f, std::min(1.0f, sample.features[i])); // 限制范围
            }
        }
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