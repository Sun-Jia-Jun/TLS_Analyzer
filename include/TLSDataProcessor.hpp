/*
TLS数据处理器，用于加载、解析、预处理从csv读取的TLS数据，并将这些数据划分为训练集和测试集，以便后续的训练和评估。
*/
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

// 一个Sample为一次完整的TLS通信会话的特征化表示。
struct Sample
{
    int label;                   // 网站标签
    std::vector<float> features; // 特征向量
};

class TLSDataProcessor
{
private:
    std::vector<Sample> samples;
    std::vector<Sample> train_samples;
    std::vector<Sample> test_samples;

    int num_labels = 0;
    int max_sequence_length = 0;
    float test_ratio = 0.2f;

    // 统计特征维度：每个包(大小+方向) + 全局统计特征
    static const int PACKET_FEATURES = 2; // 大小 + 方向
    static const int STATS_FEATURES = 6;  // 平均大小、最大、最小、标准差、出包比例、总包数

public:
    TLSDataProcessor(const std::string &csv_path)
    {
        load_data(csv_path);
        normalize_features();
        shuffle_and_split();
    }

    // 获取特征维度
    int get_feature_dim() const
    {
        return max_sequence_length * PACKET_FEATURES + STATS_FEATURES;
    }

    int get_num_labels() const { return num_labels; }
    const std::vector<Sample> &get_train_samples() const { return train_samples; }
    const std::vector<Sample> &get_test_samples() const { return test_samples; }

private:
    void load_data(const std::string &csv_path)
    {
        std::ifstream ifs(csv_path);
        if (!ifs.is_open())
        {
            throw std::runtime_error("Failed to open file: " + csv_path);
        }

        std::string line;
        std::getline(ifs, line); // 跳过header

        std::unordered_map<int, int> label_counts; // label_counts记录每个网站的样本数

        while (std::getline(ifs, line))
        {
            if (line.empty())
                continue;

            std::istringstream iss(line);
            std::string label_str, feature_str; // feature_str为本次TLS通信的所有TLS数据包的特征字符串

            if (std::getline(iss, label_str, ',') && std::getline(iss, feature_str))
            {
                Sample sample;
                sample.label = std::stoi(label_str);
                label_counts[sample.label]++;
                num_labels = std::max(num_labels, sample.label + 1); //* 确保num_labels为当前最大的标签数，+1是因为sample.label从0开始

                parse_packet_features(feature_str, sample);
                samples.push_back(sample);
            }
        }

        // 打印数据分布
        std::cout << "[INFO] Data distribution:" << std::endl;
        for (const auto &pair : label_counts)
        {
            std::cout << "  Label " << pair.first << ": " << pair.second << " samples" << std::endl;
        }

        std::cout << "[INFO] Loaded " << samples.size() << " samples with "
                  << num_labels << " classes" << std::endl;
    }

    /*
    @brief 解析某一个样本的特征字符串，提取其中的参数(大小和方向)，归一化并添加到sample。同时更新最大序列长度。
    @param feature_str 特征字符串
    @param sample feature_str对应的sample
    */
    void parse_packet_features(const std::string &feature_str, Sample &sample)
    {
        std::vector<float> packet_sizes; // 一个样本中每个包大小的向量
        std::vector<float> directions;   // 一个样本中每个包方向的向量

        std::istringstream feature_stream(feature_str);
        std::string packet_info;

        // 解析每个包的信息：大小_方向
        while (std::getline(feature_stream, packet_info, ';'))
        {
            size_t delim_pos = packet_info.find('_');
            if (delim_pos != std::string::npos)
            {
                try
                {
                    int size = std::stoi(packet_info.substr(0, delim_pos));
                    int direction = std::stoi(packet_info.substr(delim_pos + 1));

                    // 对数归一化包大小，保持在[0,1]范围
                    float normalized_size = std::log(static_cast<float>(size) + 1.0f) / std::log(1501.0f);
                    normalized_size = std::min(1.0f, std::max(0.0f, normalized_size)); //* 正溢为1，负溢为0

                    packet_sizes.push_back(normalized_size);
                    directions.push_back(static_cast<float>(direction));

                    // 添加包特征到序列中
                    sample.features.push_back(normalized_size);
                    sample.features.push_back(static_cast<float>(direction));
                }
                catch (const std::exception &e)
                {
                    // 跳过无效的包信息
                    continue;
                }
            }
        }

        // 更新最大序列长度
        int current_length = sample.features.size() / PACKET_FEATURES;
        max_sequence_length = std::max(max_sequence_length, current_length);

        // 计算并添加统计特征
        add_statistical_features(sample, packet_sizes, directions);
    }

    /*
    @brief 计算统计特征，并添加到当前样本尾部
    @param sample 当前样本
    @param sizes 当前样本中每个包的大小的集合
    @param directions 当前样本中每个包的方向的集合
    */
    void add_statistical_features(Sample &sample,
                                  const std::vector<float> &sizes,
                                  const std::vector<float> &directions)
    {
        if (sizes.empty())
            return;

        // 包大小统计
        float avg_size = std::accumulate(sizes.begin(), sizes.end(), 0.0f) / sizes.size();
        float max_size = *std::max_element(sizes.begin(), sizes.end());
        float min_size = *std::min_element(sizes.begin(), sizes.end());

        // 计算标准差
        float variance = 0.0f;
        for (float size : sizes)
        {
            variance += (size - avg_size) * (size - avg_size);
        }
        float std_dev = std::sqrt(variance / sizes.size());

        // 方向统计
        float outgoing_ratio = std::count(directions.begin(), directions.end(), 1.0f) /
                               static_cast<float>(directions.size());

        // 总包数（归一化）
        float total_packets = std::log(static_cast<float>(sizes.size()) + 1.0f) / std::log(101.0f);

        // 将统计特征暂存，稍后添加
        sample.features.insert(sample.features.end(),
                               {avg_size, max_size, min_size, std_dev, outgoing_ratio, total_packets});
    }

    void normalize_features()
    {
        std::cout << "[INFO] Normalizing features. Max sequence length: " << max_sequence_length << std::endl;

        for (auto &sample : samples)
        {
            // 分离序列特征和统计特征
            std::vector<float> stats_features;
            if (sample.features.size() >= STATS_FEATURES)
            {
                stats_features.assign(sample.features.end() - STATS_FEATURES, sample.features.end());
                sample.features.erase(sample.features.end() - STATS_FEATURES, sample.features.end());
            }

            // 填充序列特征到固定长度
            int current_length = sample.features.size() / PACKET_FEATURES;
            if (current_length < max_sequence_length)
            {
                int padding_needed = (max_sequence_length - current_length) * PACKET_FEATURES;
                sample.features.resize(sample.features.size() + padding_needed, 0.0f);
            }

            // 重新添加统计特征
            sample.features.insert(sample.features.end(), stats_features.begin(), stats_features.end());
        }

        std::cout << "[INFO] Final feature dimension: " << get_feature_dim() << std::endl;
    }

    void shuffle_and_split()
    {
        // 随机打乱
        auto rng = std::default_random_engine(std::random_device{}());
        std::shuffle(samples.begin(), samples.end(), rng);

        // 分割数据集
        size_t test_size = static_cast<size_t>(samples.size() * test_ratio);
        size_t train_size = samples.size() - test_size;

        train_samples.assign(samples.begin(), samples.begin() + train_size);
        test_samples.assign(samples.begin() + train_size, samples.end());

        std::cout << "[INFO] Train samples: " << train_samples.size()
                  << ", Test samples: " << test_samples.size() << std::endl;
    }
};

#endif // _TLS_DATA_PROCESSOR_HPP_