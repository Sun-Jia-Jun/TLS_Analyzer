// 域名管理器，提供域名的增删改查以及从文件中加载域名的功能。同时单例暴露唯一实例。

#ifndef _DOMAIN_MANAGER_HPP_
#define _DOMAIN_MANAGER_HPP_

#include <iostream>
#include <fstream>
#include <unordered_set>
#include <string>
#include <vector>
#include <memory>

#include <openssl/ssl.h>
#include <openssl/err.h>

class DomainManager
{
private:
    std::unordered_set<std::string> domains;

public:
    void add_domain(const std::string &domain)
    {
        bool inserted = domains.insert(domain).second;
        if (inserted)
            std::cout << "[INFO] Domain Added: " << domain << std::endl;
        else
            std::cout << "[INFO] Domain Already Exists: " << domain << std::endl;
    }

    void remove_domain(const std::string &domain)
    {
        size_t erased = domains.erase(domain);
        if (erased > 0)
            std::cout << "[INFO] Domain " << domain << " Removed!" << std::endl;
        else
            std::cout << "[INFO] Domain Not Found: " << domain << std::endl;
    }

    void load_domains_from_file(const std ::string domain_list_file)
    {
        std::ifstream ifs(domain_list_file);
        if (!ifs.is_open())
        {
            std::cerr << "[ERROR] Failed to open domain list file: " << domain_list_file << std::endl;
            return;
        }
        std::string line;
        while (std::getline(ifs, line))
        {
            add_domain(line);
        }
        ifs.close();
    }

    void list_domains() const
    {
        std::cout << "[INFO] All Current Domains (" << domains.size() << "):" << std::endl;
        int cnt = 1;
        for (const auto &domain : domains)
        {
            std::cout << "   No.[" << cnt++ << "] : " << domain << std::endl;
        }
    }

    std::vector<std::string> get_domains() const
    {
        return std::vector<std::string>(domains.begin(), domains.end());
    }

    bool is_empty() const
    {
        return domains.empty();
    }

    size_t size() const
    {
        return domains.size();
    }

private:
    static std::unique_ptr<DomainManager> domain_manager;

    DomainManager() = default;

    DomainManager(const DomainManager &other) = delete;
    DomainManager &operator=(const DomainManager &other) = delete;

public:
    static DomainManager *instance()
    {
        if (!domain_manager)
            domain_manager = std::unique_ptr<DomainManager>(new DomainManager());
        return domain_manager.get();
    }

    ~DomainManager() = default;
};

std::unique_ptr<DomainManager> DomainManager::domain_manager = nullptr;

#endif