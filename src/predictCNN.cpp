#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <iomanip>

#include "SimpleCNN.hpp"
#include "TLSDataProcessor.hpp"
#include "DomainManager.hpp"

// 标签到网站名称的映射
const std::vector<std::string> SITE_NAMES = DomainManager::instance()->get_domains();
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

        // 首先从训练数据确定特征维度
        TLSDataProcessor processor("../output/tls_features.csv");
        int feature_dim = processor.get_feature_dim();
        int num_labels = processor.get_num_labels();

        std::cout << "[INFO] Feature dimension: " << feature_dim << std::endl;
        std::cout << "[INFO] Number of labels: " << num_labels << std::endl;

        // 加载模型
        std::cout << "[INFO] Loading model from " << MODEL_PATH << std::endl;
        SimpleCNN model = SimpleCNN::load_model(MODEL_PATH, feature_dim, num_labels);

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
        if (predicted_label < SITE_NAMES.size())
        {
            std::cout << "Predicted website: " << SITE_NAMES[predicted_label] << std::endl;
        }
        else
        {
            std::cout << "Predicted label: " << predicted_label << std::endl;
        }

        std::cout << "Probabilities:" << std::endl;

        for (int i = 0; i < num_labels; ++i)
        {
            std::string site_name = (i < SITE_NAMES.size()) ? SITE_NAMES[i] : ("Label_" + std::to_string(i));
            std::cout << "  " << std::setw(10) << std::left << site_name << ": "
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