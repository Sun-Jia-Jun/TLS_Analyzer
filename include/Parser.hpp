/*  Parser解析器用于从pcap文件中提取出TLS数据包的大小、方向等特征值，并将其保存到csv文件中。
    我们规定csv的列为：site_name, timestamp, ip_src, ip_dst, tls_record_type, frame_length, tls_handshake_type, tls_direction
    对于tls_direction，我们规定0:client->server, 1:server->client
*/

/*
tls.handshake.type:(用于判断数据包方向)
0: Hello Request
1: Client Hello
2: Server Hello
...
所以说，如果是1，说明是从客户端发起握手的包；2则代表服务端响应握手的包
*/
#ifndef _PARSER_HPP_
#define _PARSER_HPP_

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <unistd.h>
#include <array>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cstdlib>

#include "FileLoader.hpp"
#include "DomainManager.hpp"

struct TLSRecord
{
    std::string site_name;
    std::string ip_src;
    std::string ip_dst;
    int tls_record_type = -1;
    int frame_length = -1;
    int tls_handshake_type = -1;
    int tls_direction = -1;
};

class Parser
{
private:
    std::string server_ip;
    std::string client_ip;

    std::vector<TLSRecord> single_pcap_tls_records; // 一个pcap文件中的所有TLS特征，整体作为一个样本。

    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<TLSRecord>>> tls_records_map; // 所有域名下所有pcap文件中的所有TLS特征。作为一个全局映射。
    // all_domain -> baidu -> 50*xxx.pcap -> 8*TLS
    // 一个哈希表中储存多个子哈希表，一个子哈希表代表一个域名下所有的pcap文件，子哈希表的key为pcap文件名，value为该pcap文件中的所有TLS特征。

public:
    Parser()
    {
        if (!is_tshark_available())
        {
            std::cerr << "[ERROR] tshark is not available! Try to install it first." << std::endl;
            exit(1);
        }
        parse_all_files();
    }

private:
    bool is_tshark_available()
    {
        FILE *fp = popen("which tshark", "r");
        if (!fp)
            return false;
        char buf[256];
        bool found = (fgets(buf, sizeof(buf), fp) != NULL);
        pclose(fp);
        return found;
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

    void parse_single_file(const std::string &file_path)
    {
        if (file_path.empty() || access(file_path.c_str(), R_OK) != 0)
        {
            std::cerr << "[ERROR] Cannot access pcap file: " << file_path << std::endl;
            return;
        }

        std::cout << "[INFO] Parsing TLSRecord from file: " << file_path << std::endl;

        // 清空上一次解析的结果
        single_pcap_tls_records.clear();
        server_ip.clear();
        client_ip.clear();

        std::stringstream tshark_cmd;
        tshark_cmd << "tshark -r \"" << file_path << "\""
                   << " -Y \"tls\""
                   << " -T fields"
                   << " -e frame.time_epoch" // 0
                   << " -e ip.src"           // 1
                   << " -e ip.dst"           // 2
                   //    << " -e tls.record.content_type"
                   << " -e frame.len"          // 3 tls.record.length -> frame.len
                   << " -e tls.handshake.type" // 4
                   << " -E header=n"
                   << " -E separator=,"
                   // << " -E quote=d" // 使用引号隔开
                   << " -E occurrence=f";

        std::cout << "[INFO] Running tshark command: " << tshark_cmd.str() << std::endl;

        FILE *fp = popen(tshark_cmd.str().c_str(), "r");
        if (!fp)
        {
            std::cerr << "[ERROR] Failed to run tshark command: " << strerror(errno) << std::endl;
            return;
        }

        std::array<char, 4096> buf;
        size_t record_count = 0;
        std::string site_name = extract_site_name_from_url(file_path);

        while (fgets(buf.data(), buf.size(), fp) != nullptr) // 逐行读取输出
        {
            std::string line(buf.data());
            if (line.empty())
                continue;

            // 去掉行尾的换行符
            size_t pos = line.find_last_not_of("\r\n");
            if (pos != std::string::npos)
            {
                line.erase(pos + 1);
            }
            else
            {
                line.clear(); // 空行
                continue;
            }

            std::vector<std::string> tokens;
            std::stringstream iss(line);
            std::string token;

            while (std::getline(iss, token, ','))
            {
                if (!token.empty() && token.front() == '"' && token.back() == '"') // 处理带引号的字段
                    token = token.substr(1, token.size() - 2);
                tokens.push_back(token);
            }
            // std::cout << "[DEBUG] tokens: " << tokens.size() << std::endl;

            TLSRecord tls_record;
            tls_record.site_name = site_name;
            tls_record.ip_src = tokens[1];
            tls_record.ip_dst = tokens[2];

            // 安全地转换数值字段
            try
            {
                if (!tokens[3].empty()) // length
                    tls_record.frame_length = std::stoi(tokens[3]);
                if (tokens.size() > 4)
                    if (!tokens[4].empty()) // type
                        tls_record.tls_handshake_type = std::stoi(tokens[4]);
            }
            catch (const std::exception &e)
            {
                std::cerr << "[WARN] Failed to parse numeric fields: " << e.what() << " in line: " << line << std::endl;
                continue;
            }

            // DEBUG (token4可能越界)
            // std::cout << "[DEBUG]" << tokens[0] << " " << tokens[1] << " " << tokens[2] << " " << tokens[3] << " " << tokens[4] << " " << std::endl;

            // 根据握手包的类型确定该pcap文件中所有数据包的client_ip和server_ip，从而确定所有数据包的方向
            if (tls_record.tls_handshake_type == 1)
            {
                tls_record.tls_direction = 0; // client->server
                client_ip = tls_record.ip_src;
                server_ip = tls_record.ip_dst;
            }
            else if (tls_record.tls_handshake_type == 2)
            {
                tls_record.tls_direction = 1; // server->client
                client_ip = tls_record.ip_dst;
                server_ip = tls_record.ip_src;
            }
            else if (!client_ip.empty() || !server_ip.empty())
            {
                auto tls_direction_temp1 = (client_ip == tls_record.ip_src) ? 0 : 1;
                auto tls_direction_temp2 = (server_ip == tls_record.ip_src) ? 1 : 0;
                if (tls_direction_temp1 == tls_direction_temp2)
                {
                    tls_record.tls_direction = tls_direction_temp1;
                }
                else
                {
                    std::cerr << "[WARN] Failed to determine tls_direction for tls_record: " << tls_record.ip_src << "->" << tls_record.ip_dst << std::endl;
                    continue;
                }
            }
            else
            {
                std::cerr << "[WARN] Failed to determine tls_direction for tls_record: " << tls_record.ip_src << "->" << tls_record.ip_dst << std::endl;
            }

            single_pcap_tls_records.push_back(tls_record);
            record_count++;
        }

        int status = pclose(fp);
        if (status != 0)
        {
            if (WIFEXITED(status))
            {
                std::cerr << "[WARN] tshark command exited with status: " << WEXITSTATUS(status) << std::endl;
            }
            else if (WIFSIGNALED(status))
            {
                std::cerr << "[WARN] tshark command killed by signal: " << WTERMSIG(status) << std::endl;
            }
            else
            {
                std::cerr << "[WARN] tshark command terminated abnormally" << std::endl;
            }
        }

        std::cout << "[INFO] Parsed " << record_count << " TLS records from " << file_path << std::endl;

        // 存储到全局映射中
        if (!single_pcap_tls_records.empty() && !site_name.empty())
        {
            std::string filename = file_path.substr(file_path.find_last_of('/') + 1);
            tls_records_map[site_name][filename] = single_pcap_tls_records;
        }
    }

    void parse_all_files()
    {
        for (auto &domain : FileLoader::instance()->get_file_map())
        {
            for (auto &file : domain.second)
            {
                parse_single_file(file);
            }
        }
    }

public:
    const auto &get_tls_records_map() const { return tls_records_map; }
};

#endif