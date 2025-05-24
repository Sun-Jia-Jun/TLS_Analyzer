/*
Capture用于调用tcpdump进行抓包，并将抓到的数据包保存到pcap文件中
*/
#ifndef CAPTURE_HPP
#define CAPTURE_HPP

#include <stdio.h>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <mutex>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <chrono>
#include <thread>
#include <fstream>
#include <cerrno>
#include <cstring>
#include <sys/wait.h>

class Capture
{
private:
    std::string output_file;
    std::string filter;
    std::string interface = "any"; // 抓包接口，默认为any。可根据需要改为ethx
    bool capturing = false;
    pid_t tcpdump_pid;
    std::mutex mtx;

public:
    Capture(const std::string &interface, const std::string &filter)
        : interface(interface), filter(filter), tcpdump_pid(-1)
    {
        if (!is_tcpdump_available())
        {
            std::cerr << "[ERROR] tcpdump is not available. Try to install it first." << std::endl;
        }
    }
    ~Capture()
    {
        stop();
    }

    bool start(const std::string &host, int port = 443)
    {
        std::lock_guard<std::mutex> lock(mtx);

        if (capturing)
        {
            std::cerr << "[WARN] tcpdump is already running. Unable to start capturing again." << std::endl;
            return false;
        }

        // 构建过滤规则：host xxx and port 443
        std::string filter_cmd = filter;
        if (filter_cmd.empty())
        {
            filter_cmd = "host " + host;
            if (port > 0 && port != 443)
                filter_cmd += " and port " + std::to_string(port);
        }

        // 构建pcap储存目录
        std::string site_name = extract_site_name_from_url(host);

        std::string all_data_dir = "../data";
        std::string pcap_dir = all_data_dir + "/" + site_name;
        if (!ensure_dir_exists(pcap_dir) || !ensure_dir_exists(all_data_dir))
        {
            std::cerr << "[ERROR] Failed to create data directories:" << pcap_dir << "or " << all_data_dir << std::endl;
            return false;
        }

        std::string timestamp = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
        std::string pcap_file = pcap_dir + "/" + timestamp + ".pcap";
        output_file = pcap_file; //* 更新output_file，供stop调用
        std::cout << "[INFO] Setting up capture to output file: " << pcap_file << std::endl;

        // 构建tcpdump命令并exec
        std::stringstream tcpdump_cmd; // tcpdump -i any -w ../data/baidu/123.pcap host www.baidu.com and port 443
        tcpdump_cmd << "tcpdump -i" << interface << " -w " << pcap_file << " " << filter_cmd;
        std::cout << "[INFO] Running tcpdump with command: " << tcpdump_cmd.str() << std::endl
                  << std::flush;

        tcpdump_pid = fork();
        if (tcpdump_pid < 0)
        {
            std::cerr << "[ERROR] Failed to fork child process for tcpdump." << std::endl;
            return false;
        }
        else if (tcpdump_pid == 0) // 子进程入口，执行tcpdump
        {
            //* 使用popen可能没有正确退出  -->  艹改完回头又瞅了一眼，发现我忘记pclose(fp)了，(悲)
            // std::string cmd = tcpdump_cmd.str();
            // popen(cmd.c_str(), "r");

            // 使用 execl 而不是 popen 来执行 tcpdump
            std::string cmd = tcpdump_cmd.str();
            std::vector<std::string> args;
            std::istringstream iss(cmd);
            std::string token;
            while (iss >> token)
            {
                args.push_back(token);
            }

            // 构建参数数组
            char *argv[args.size() + 1];
            for (size_t i = 0; i < args.size(); i++)
            {
                argv[i] = const_cast<char *>(args[i].c_str());
            }
            argv[args.size()] = nullptr;

            execvp(argv[0], argv);

            // 如果 execvp 失败，记录错误并退出
            int err_code = errno;
            FILE *err_log = fopen("../data/tcpdump_exec_error.log", "a");
            if (err_log)
            {
                fprintf(err_log, "[ERROR] Failed to execute tcpdump: %s\n", strerror(err_code));
                fclose(err_log);
            }

            exit(1); // 确保子进程在 execvp 失败时退出
        }

        // 父进程继续执行
        capturing = true;
        std::this_thread::sleep_for(std::chrono::milliseconds(500)); // 等待tcpdump启动
        int status;
        pid_t result = waitpid(tcpdump_pid, &status, WNOHANG);

        if (result > 0) // 子进程结束
        {
            capturing = false;
            tcpdump_pid = -1;

            if (WIFEXITED(status))
            {
                std::cerr << "[ERROR] tcpdump exited prematurely with status: "
                          << WEXITSTATUS(status) << std::endl;
            }
            else if (WIFSIGNALED(status))
            {
                std::cerr << "[ERROR] tcpdump terminated by signal: "
                          << WTERMSIG(status) << std::endl;
            }
            return false;
        }
        else if (result < 0)
        {
            std::cerr << "[ERROR] Error occurred when waiting for tcpdump process: " << strerror(errno) << std::endl;
            capturing = false;
            return false;
        }

        std::cout << "[INFO] Packet capture started. Output file: " << pcap_file << std::endl;
        return true;
    }

    bool stop()
    {
        std::lock_guard<std::mutex> lock(mtx);

        if (!capturing || tcpdump_pid < 0)
        {
            return true;
        }

        std::cout << "[INFO] Stopping packet capture..." << std::endl;

        if (kill(tcpdump_pid, SIGTERM) < 0)
        {
            std::cerr << "[WARN] Failed to send SIGTERM to tcpdump process: " << strerror(errno) << std::endl;

            if (kill(tcpdump_pid, SIGKILL) < 0)
            {
                std::cerr << "[ERROR] Failed to send SIGKILL to tcpdump process: " << strerror(errno) << std::endl;
                return false;
            }
        }

        tcpdump_pid = -1;
        capturing = false;
        std::cout << "[INFO] Packet capture stopped." << std::endl;

        struct stat output_file_st;
        if (stat(output_file.c_str(), &output_file_st) == 0)
            std::cerr << "[INFO] Output file size: " << output_file_st.st_size << " bytes." << std::endl;
        else
            std::cerr << "[WARN] Failed to get output file size: " << strerror(errno) << std::endl;

        return true;
    }

public:
    bool is_capturing() const { return capturing; }

private:
    static bool is_tcpdump_available()
    {
        FILE *fp = popen("which tcpdump", "r");
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

    static bool ensure_dir_exists(const std::string &dir)
    {
        struct stat dir_st;

        // 检查目录是否已存在
        if (stat(dir.c_str(), &dir_st) == 0)
        {
            if (S_ISDIR(dir_st.st_mode))
            {
                return true; // 目录已存在
            }
            else
            {
                std::cerr << "[ERROR] Path exists but is not a directory: " << dir << std::endl;
                return false;
            }
        }

        // 目录不存在，尝试创建
        if (mkdir(dir.c_str(), 0755) != 0)
        {
            std::cerr << "[ERROR] Failed to create directory: " << dir
                      << " - " << strerror(errno) << std::endl;
            return false;
        }

        // 验证创建成功
        if (stat(dir.c_str(), &dir_st) == 0 && S_ISDIR(dir_st.st_mode))
        {
            std::cout << "[INFO] Directory " << dir << " created successfully." << std::endl;
            return true;
        }
        else
        {
            std::cerr << "[ERROR] Directory creation verification failed: " << dir << std::endl;
            return false;
        }
    }
};

#endif // CAPTURE_HPP