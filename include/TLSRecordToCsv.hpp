#ifndef _TLSRECORD_TO_CSV_HPP_
#define _TLSRECORD_TO_CSV_HPP_

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <sys/stat.h>

#include "Parser.hpp"
#include "FileLoader.hpp"
#include "DomainManager.hpp"

class TLSRecordToCsv
{
private:
    std::unordered_map<std::string, int> site_labels; // 所有网站名称及其对应的索引
    Parser &parser;                                   // 指向Parser对象的引用，用于获取TLS记录

    std::string output_csv_path = "../data/tls_features.csv"; // 输出CSV文件路径
    std::string label_map_path = "../data/site_labels.csv";   // 输出标签映射文件路径
    int sample_count = 0;                                     // 样本数量计数器
    int max_records_in_sample = -1;                           // 每个样本最多TLS记录数量，用于构建CNN输入向量

public:
    TLSRecordToCsv(Parser &parser_ref, const std::string &output_dir = "../output")
        : parser(parser_ref)
    {
        ensure_output_directory(output_dir);

        output_csv_path = output_dir + "/tls_features.csv";
        label_map_path = output_dir + "/site_labels.csv";

        initialize_site_labels();
    }

    // 生成CSV文件，将TLS记录转换为CNN训练所需的特征
    bool generate_csv()
    {
        std::cout << "[INFO] Generating CSV file for CNN training..." << std::endl;

        // 打开输出文件
        std::ofstream ofs(output_csv_path);
        if (!ofs.is_open())
        {
            std::cerr << "[ERROR] Failed to open output CSV file: " << output_csv_path << std::endl;
            return false;
        }

        // 写入列标签
        ofs << "site_label,packet_features" << std::endl;

        // 从Parser获取所有TLS记录
        auto &records_map = parser.get_tls_records_map();

        // 遍历每个站点
        for (const auto &site_pair : records_map)
        {
            std::string site_name = site_pair.first;
            int site_label = site_labels[site_name];

            const auto &site_files = site_pair.second;

            // 遍历该站点的每个pcap文件
            for (const auto &file_pair : site_files)
            {
                const auto &tls_records = file_pair.second;

                // 将单个pcap文件的所有TLS记录转换为一个特征向量
                std::string feature_str = convert_tls_records_to_feature_string(tls_records);

                // 写入CSV行: site_label,feature_string
                if (!feature_str.empty())
                {
                    ofs << site_label << "," << feature_str << std::endl;
                    sample_count++;
                }
            }
        }

        ofs.close();

        // 生成标签映射文件
        generate_label_map();

        std::cout << "[INFO] CSV generation completed. Total samples: " << sample_count << std::endl;
        std::cout << "[INFO] CSV file saved to: " << output_csv_path << std::endl;
        std::cout << "[INFO] Label map saved to: " << label_map_path << std::endl;

        return true;
    }

private:
    // 初始化站点标签映射
    void initialize_site_labels()
    {
        auto &records_map = parser.get_tls_records_map();

        // 从DomainManager加载所有域名，并分配标签
        std::unordered_set<std::string> unique_sites;
        std::vector<std::string> domains = DomainManager::instance()->get_domains();
        for (const auto &domain : domains)
        {
            std::string site_name = extract_site_name_from_url(domain);
            unique_sites.insert(site_name);
        }

        int label = 0;
        for (const auto &site : unique_sites)
        {
            site_labels[site] = label++;
        }

        std::cout << "[INFO] Initialized " << site_labels.size() << " site labels: ";
        for (const auto &pair : site_labels)
        {
            std::cout << pair.first << "(" << pair.second << ") ";
        }
        std::cout << std::endl;
    }

    // 生成标签映射文件
    void generate_label_map()
    {
        std::ofstream ofs(label_map_path);
        if (!ofs.is_open())
        {
            std::cerr << "[ERROR] Failed to open label map file: " << label_map_path << std::endl;
            return;
        }

        ofs << "label,site_name" << std::endl;

        // 创建一个临时向量来排序标签
        std::vector<std::pair<int, std::string>> sorted_labels;
        for (const auto &pair : site_labels)
        {
            sorted_labels.push_back({pair.second, pair.first});
        }

        // 按标签值排序
        std::sort(sorted_labels.begin(), sorted_labels.end());

        // 写入排序后的标签映射
        for (const auto &pair : sorted_labels)
        {
            ofs << pair.first << "," << pair.second << std::endl;
        }

        ofs.close();
    }

    // 将TLS记录转换为特征字符串
    std::string convert_tls_records_to_feature_string(const std::vector<TLSRecord> &records)
    {
        // 一个pcap中的所有TLSRecord -> 387_0;1492_1;1000_1;198_0;298_1;233_0;1492_1;169_1 ...
        if (records.empty())
        {
            return "";
        }

        std::stringstream ss;
        bool first = true;

        // 提取每个TLS记录的大小和方向，用分号连接
        for (const auto &record : records)
        {
            // 只考虑有效的记录（有长度和方向信息）
            if (record.frame_length > 0 && record.tls_direction >= 0)
            {
                if (!first)
                {
                    ss << ";";
                }
                // 格式: 长度_方向
                ss << record.frame_length << "_" << record.tls_direction;
                first = false;
            }
        }

        return ss.str();
    }

    // 确保输出目录存在
    bool ensure_output_directory(const std::string &dir_path)
    {
        struct stat st;
        if (stat(dir_path.c_str(), &st) != 0)
        {
            if (mkdir(dir_path.c_str(), 0755) != 0)
            {
                std::cerr << "[ERROR] Failed to create output directory: "
                          << dir_path << ": " << strerror(errno) << std::endl;
                return false;
            }
            std::cout << "[INFO] Created output directory: " << dir_path << std::endl;
        }
        else if (!S_ISDIR(st.st_mode))
        {
            // 路径存在但不是目录
            std::cerr << "[ERROR] Output path exists but is not a directory: "
                      << dir_path << std::endl;
            return false;
        }

        return true;
    }

    static std::string extract_site_name_from_url(const std::string &url)
    {
        // www.baidu.com -> baidu
        std::vector<std::string> parts;
        std::string part;
        std::istringstream iss(url);
        std::string site_name;

        while (std::getline(iss, part, '.'))
        {
            parts.emplace_back(part);
        }
        if (parts.size() >= 2)
            return parts[parts.size() - 2];
        else
        {
            std::cerr << "[WARN] Invalid URL format:" << url << std::endl;
            return url;
        }
    }

    void set_max_records_in_sample()
    {
        // 每个样本中的tls记录数为std::vector<TLSRecord>.size()
        auto &records_map = parser.get_tls_records_map();
        for (const auto &site_pair : records_map)
        {
            for (const auto &file_pair : site_pair.second)
            {
                const auto &tls_records = file_pair.second;
                if (tls_records.size() > max_records_in_sample)
                {
                    max_records_in_sample = tls_records.size();
                }
            }
        }
        std::cout << "[INFO] Max records in sample is : " << max_records_in_sample << std::endl;
    }

public:
    // Getter
    const int get_max_records_in_sample() const { return max_records_in_sample; }
};
#endif // _TLSRECORD_TO_CSV_HPP_