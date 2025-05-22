// Pcap文件加载器，用于从./data/目录下加载各域名对应的所有pcap文件到一个map中(std::unordered_map<std::string, std::vector<std::string>>)
// 同时提供单例全局接口

#ifndef FILELOADER_HPP
#define FILELOADER_HPP

#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <sys/stat.h>
#include <sstream>
#include <dirent.h>
#include <algorithm>

#include "DomainManager.hpp"
// #include "Capture.hpp"

class FileLoader
{
private:
    std::unordered_map<std::string, std::vector<std::string>> file_map;
    std::string data_base_dir; // 数据根目录，即./data
    DomainManager *domain_manager;

public:
    void start(const std::string &input_data_base_dir = "../data")
    {
        data_base_dir = input_data_base_dir;
        load_files();
    }

    bool load_files()
    {
        struct stat data_base_st;
        if (stat(data_base_dir.c_str(), &data_base_st) != 0)
        {
            std::cerr << "[ERROR] Data base directory not found: " << data_base_dir << std::endl;
            return false;
        }

        std::cout << "[INFO] loading pcap files from " << data_base_dir << std::endl;

        // 为了update，需先清空原有map
        file_map.clear();

        std::cout << "[INFO] Found " << DomainManager::instance()->get_domains().size() << " domains." << std::endl;
        // 遍历所有域名，提取其对应目录下的所有文件
        for (const auto &domain : DomainManager::instance()->get_domains())
        {
            std::string site_name = extract_site_name_from_url(domain);
            std::string domain_dir = data_base_dir + "/" + site_name;

            struct stat domain_st;
            if (stat(domain_dir.c_str(), &domain_st) != 0 || !S_ISDIR(domain_st.st_mode)) // 域名子目录不存在或不是目录
            {
                std::cerr << "[WARN] Directory for domain " << domain << " not found or is not a directory." << std::endl;
                file_map[site_name] = std::vector<std::string>(); // 填入一个空向量
                continue;
            }

            std::cout << "[INFO] Loading pcap files for domain " << domain << " from " << domain_dir << std::endl;
            std::vector<std::string> domain_pcap_vector = load_pcaps_from_domain_dir(domain_dir);
            file_map[site_name] = domain_pcap_vector;
            std::cout << "[INFO] Loaded " << domain_pcap_vector.size() << " pcap files for domain " << domain << std::endl;
        }

        return true;
    }

    void list_all_files() const
    {
        for (const auto &domain : file_map)
        {
            std::cout << "Domain: " << domain.first << std::endl;
            for (const auto &file : domain.second)
            {
                std::cout << "   ---" << file << std::endl;
            }
        }
    }

    // Getter方法
    auto get_file_map() const { return file_map; }

private:
    static std::string extract_site_name_from_url(const std::string &url) // 和Capture中的函数重复，可优化。
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

    std::vector<std::string> load_pcaps_from_domain_dir(const std::string &domain_dir) // 从域名子目录中加载所有pcap文件
    {
        std::vector<std::string> pcap_vector;

        DIR *dir = opendir(domain_dir.c_str());
        if (!dir)
        {
            std::cerr << "[ERROR] Failed to open directory: " << domain_dir << "when loading pcap files." << std::endl;
            return pcap_vector;
        }

        struct dirent *entry;
        while ((entry = readdir(dir)) != NULL)
        {
            std::string file_name = entry->d_name;
            if (file_name.find(".pcap") && file_name.size() > 5)
                pcap_vector.emplace_back(domain_dir + "/" + file_name);
        }
        closedir(dir);
        std::sort(pcap_vector.begin(), pcap_vector.end()); // 升序排序
        return pcap_vector;
    }

private:
    static std::unique_ptr<FileLoader> file_loader;
    FileLoader() = default;
    FileLoader(const FileLoader &other) = delete;
    FileLoader &operator=(const FileLoader &other) = delete;

public:
    static FileLoader *instance()
    {
        if (!file_loader)
            file_loader = std::unique_ptr<FileLoader>(new FileLoader());
        return file_loader.get();
    }

    ~FileLoader() = default;
};

inline std::unique_ptr<FileLoader> FileLoader::file_loader = nullptr;

#endif // FILELOADER_HPP
