#include <iostream>

#include "HttpsClient.hpp"
#include "DomainManager.hpp"
#include "Capture.hpp"
#include "FileLoader.hpp"
#include "Parser.hpp"
#include "TLSRecordToCsv.hpp"

const int MAX_CAPTURE_COUNT = 50;

int main(int argc, char **argv)
{
    // 加载并列出所有目标域名
    DomainManager::instance()->load_domains_from_file("../domain_list.txt");
    if (DomainManager::instance()->is_empty())
    {
        std::cerr << "[ERROR] Domain list is empty: " << "../domain_list.txt" << std::endl;
        return 1;
    }
    std::cout << "PRESS ANY KEY TO CONTINUE..." << std::endl;
    std::cout << "Press 1 to skip capture" << std::endl;
    char ch = getchar();
    if (ch == '1')
        goto skip_capture;

    DomainManager::instance()->list_domains();
    // 遍历所有域名，进行抓包和HTTPS请求
    for (const auto &domain : DomainManager::instance()->get_domains())
    {
        for (int i = 0; i < MAX_CAPTURE_COUNT; i++)
        {
            std::cout << std::endl
                      << "[INFO] Processing domain: " << domain << std::endl;
            std::cout << "-----------------------------------------------------------------------" << std::endl;

            Capture capture("any", "host " + domain);

            try
            {
                std::cout << "[INFO] Starting capture packets ..." << std::endl;
                capture.start(domain);
                sleep(1);

                std::cout << "[INFO] Sending HTTPS request to " << domain << "..." << std::endl;
                HttpsClient client(domain);
                sleep(1.5);

                std::cout << "[INFO] Stopping packet capture ..." << std::endl;
                capture.stop();
            }
            catch (const std::exception &e)
            {
                std::cerr << "[ERROR] EXCEPTION:" << e.what() << std::endl;
                if (capture.is_capturing())
                {
                    capture.stop();
                }

                std::cout << "--------------------------------------------------------------------" << std::endl;
            }
        }
    }
    std::cout << "[INFO] ALl domains processed." << std::endl;

    // 解析pcap文件，提取特征值并保存到csv中
skip_capture:

    std::cout << "Press to continue CSV CONVERSION..." << std::endl;
    getchar();

    FileLoader::instance()->start("../data");
    FileLoader::instance()->list_all_files();
    Parser parser;

    // 添加以下代码用于生成CSV文件
    std::cout << "Press to continue CSV generation..." << std::endl;
    getchar();

    TLSRecordToCsv csv_converter(parser);
    csv_converter.generate_csv();

    return 0;
}