#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <iomanip>

#include "SimpleCNN.hpp"
#include "TLSDataProcessor.hpp"

// 标签到网站名称的映射
const std::vector<std::string> SITE_NAMES = {"bing", "baidu", "bilibili"};
const std::string MODEL_PATH = "../data/tls_model.bin";

// 从单个PCAP文件加载TLS特征
std::vector<float> load_features_from_file(const std::string &file_path, int feature_dim)
{
    std::ifstream file(file_path);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + file_path);
    }

    std::string line;
    std::getline(file, line); // 读取第一行

    std::vector<float> features;
    std::istringstream iss(line);
    std::string pair;

    while (std::getline(iss, pair, ';'))
    {
        size_t delim_pos = pair.find('_');
        if (delim_pos != std::string::npos)
        {
            int size = std::stoi(pair.substr(0, delim_pos));
            int direction = std::stoi(pair.substr(delim_pos + 1));

            // 归一化数据包大小
            float normalized_size = static_cast<float>(size) / 1500.0f;

            features.push_back(normalized_size);
            features.push_back(static_cast<float>(direction));
        }
    }

    // 填充到固定长度
    if (features.size() < feature_dim)
    {
        features.resize(feature_dim, 0.0f);
    }

    return features;
}

int main(int argc, char **argv)
{
    try
    {
        if (argc < 2)
        {
            std::cerr << "Usage: " << argv[0] << " <tls_feature_file>" << std::endl;
            return 1;
        }

        // 必须指定特征维度和标签数量来加载模型
        // 这里假设为训练时的值，实际应从模型文件中读取
        int feature_dim = 100; // 示例值，需要与训练时一致
        int num_labels = 3;    // bing, baidu, bilibili

        // 加载模型
        std::cout << "[INFO] Loading model from " << MODEL_PATH << std::endl;
        SimpleCNN model(feature_dim, num_labels);
        // model = SimpleCNN::load_model(MODEL_PATH);  // 实际加载代码（需完善saveModel/loadModel功能）

        // 加载要预测的特征
        std::string feature_file = argv[1];
        std::cout << "[INFO] Loading features from " << feature_file << std::endl;

        std::vector<float> features = load_features_from_file(feature_file, feature_dim);

        // 进行预测
        std::vector<float> probabilities = model.forward(features);

        // 找到最可能的网站
        int predicted_label = std::max_element(probabilities.begin(), probabilities.end()) - probabilities.begin();

        // 输出结果
        std::cout << "\n===== Prediction Result =====" << std::endl;
        std::cout << "Predicted website: " << SITE_NAMES[predicted_label] << std::endl;
        std::cout << "Probabilities:" << std::endl;

        for (int i = 0; i < num_labels; ++i)
        {
            std::cout << "  " << std::setw(10) << std::left << SITE_NAMES[i] << ": "
                      << std::fixed << std::setprecision(2) << (probabilities[i] * 100) << "%"
                      << std::endl;
        }

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }
}