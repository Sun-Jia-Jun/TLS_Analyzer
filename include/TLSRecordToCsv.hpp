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
#include <sstream>

#include "Parser.hpp"
#include "FileLoader.hpp"
#include "DomainManager.hpp"

class TLSRecordToCsv
{
private:
    std::unordered_map<std::string, int> site_labels; // 网站名称到标签的映射
    Parser &parser;

    std::string output_csv_path;
    std::string label_map_path;
    int sample_count = 0;

public:
    TLSRecordToCsv(Parser &parser_ref, const std::string &output_dir = "../output")
        : parser(parser_ref)
    {
        ensure_output_directory(output_dir);
        output_csv_path = output_dir + "/tls_features.csv";
        label_map_path = output_dir + "/site_labels.csv";
        initialize_site_labels();
    }

    // 生成CSV文件：将每个pcap文件的TLS记录序列转换为一行特征数据
    bool generate_csv()
    {
        std::cout << "[INFO] Generating CSV file for CNN training..." << std::endl;

        std::ofstream ofs(output_csv_path);
        if (!ofs.is_open())
        {
            std::cerr << "[ERROR] Failed to open output CSV file: " << output_csv_path << std::endl;
            return false;
        }

        // CSV格式：site_label,packet_features
        // 其中packet_features格式：387_0;1492_1;1000_1;198_0 (大小_方向;大小_方向;...)
        ofs << "site_label,packet_features" << std::endl;

        auto &records_map = parser.get_tls_records_map();

        // 遍历每个网站
        for (const auto &site_pair : records_map)
        {
            std::string site_name = site_pair.first;

            // 检查站点是否在标签映射中
            if (site_labels.find(site_name) == site_labels.end())
            {
                std::cerr << "[WARN] Site not found in labels: " << site_name << std::endl;
                continue;
            }

            int site_label = site_labels[site_name];
            const auto &site_files = site_pair.second;

            // 遍历该网站的每个pcap文件
            for (const auto &file_pair : site_files)
            {
                const std::string &filename = file_pair.first;
                const auto &tls_records = file_pair.second;

                // 将一个pcap文件的所有TLS记录转换为特征字符串
                std::string feature_str = convert_records_to_features(tls_records);

                if (!feature_str.empty())
                {
                    ofs << site_label << "," << feature_str << std::endl;
                    sample_count++;

                    if (sample_count % 100 == 0)
                    {
                        std::cout << "[INFO] Processed " << sample_count << " samples..." << std::endl;
                    }
                }
            }
        }

        ofs.close();
        generate_label_map();

        std::cout << "[INFO] CSV generation completed." << std::endl;
        std::cout << "[INFO] Total samples: " << sample_count << std::endl;
        std::cout << "[INFO] CSV file: " << output_csv_path << std::endl;
        std::cout << "[INFO] Label map: " << label_map_path << std::endl;

        return true;
    }

private:
    // 初始化站点标签映射：为每个网站分配一个唯一的数字标签
    void initialize_site_labels()
    {
        // 从domain_list.txt获取所有域名，确保标签分配的一致性
        std::vector<std::string> domains = DomainManager::instance()->get_domains();

        for (size_t i = 0; i < domains.size(); ++i)
        {
            std::string site_name = extract_site_name_from_domain(domains[i]);
            site_labels[site_name] = static_cast<int>(i);
        }

        std::cout << "[INFO] Initialized " << site_labels.size() << " site labels:" << std::endl;
        for (const auto &pair : site_labels)
        {
            std::cout << "  " << pair.first << " -> " << pair.second << std::endl;
        }
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

        // 按标签值排序输出
        std::vector<std::pair<int, std::string>> sorted_labels;
        for (const auto &pair : site_labels)
        {
            sorted_labels.push_back({pair.second, pair.first});
        }
        std::sort(sorted_labels.begin(), sorted_labels.end());

        for (const auto &pair : sorted_labels)
        {
            ofs << pair.first << "," << pair.second << std::endl;
        }

        ofs.close();
    }

    // 将TLS记录序列转换为特征字符串
    std::string convert_records_to_features(const std::vector<TLSRecord> &records)
    {
        if (records.empty())
        {
            return "";
        }

        std::stringstream ss;
        bool first = true;

        // 格式：包大小_方向;包大小_方向;...
        // 例如：387_0;1492_1;1000_1;198_0
        for (const auto &record : records)
        {
            if (record.frame_length > 0 && record.tls_direction >= 0)
            {
                if (!first)
                {
                    ss << ";";
                }
                ss << record.frame_length << "_" << record.tls_direction;
                first = false;
            }
        }

        return ss.str();
    }

    // 从域名提取网站名称 (www.baidu.com -> baidu)
    static std::string extract_site_name_from_domain(const std::string &domain)
    {
        std::vector<std::string> parts;
        std::stringstream ss(domain);
        std::string part;

        while (std::getline(ss, part, '.'))
        {
            parts.push_back(part);
        }

        if (parts.size() >= 2)
        {
            return parts[parts.size() - 2]; // 倒数第二部分
        }
        else
        {
            std::cerr << "[WARN] Invalid domain format: " << domain << std::endl;
            return domain;
        }
    }

    // 确保输出目录存在
    bool ensure_output_directory(const std::string &dir_path)
    {
        struct stat st;
        if (stat(dir_path.c_str(), &st) != 0)
        {
            if (mkdir(dir_path.c_str(), 0755) != 0)
            {
                std::cerr << "[ERROR] Failed to create directory: " << dir_path << std::endl;
                return false;
            }
            std::cout << "[INFO] Created directory: " << dir_path << std::endl;
        }
        return true;
    }
};

#endif // _TLSRECORD_TO_CSV_HPP_