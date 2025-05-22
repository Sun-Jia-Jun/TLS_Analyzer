// 基于OpenSSL的模拟https客户端

#ifndef HTTPSCLIENT_HPP
#define HTTPSCLIENT_HPP

#include <iostream>
#include <fstream>
#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/stat.h>
#include <string>
#include <stdexcept>

#include <openssl/ssl.h>
#include <openssl/err.h>

#include "DomainManager.hpp"

class HttpsClient
{
private:
    DomainManager *domain_manager;

    SSL_CTX *ssl_ctx = nullptr;
    SSL *ssl = nullptr;
    int socket_fd = -1;
    int port;

    std::string hostname;
    std::string request;

public:
    HttpsClient(const std::string &hostname_, int port_ = 443)
        : hostname(hostname_), port(port_)
    {
        // 初始化请求字符串（在hostname初始化后）
        request = "GET / HTTP/1.1\r\n"
                  "Host: " +
                  hostname + "\r\n"
                             "Connection: close\r\n"
                             "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36\r\n"
                             "\r\n";

        try
        {
            ensure_output_directory();
            init_openssl();
            create_ssl_context();
            create_socket_and_connect();
            establish_ssl_connection();
            send_request(request);
            receive_response("../data/" + hostname + "_output.html");
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << std::endl;
            cleanup();
            throw;
        }
    }

    ~HttpsClient()
    {
        cleanup();
    }

    void send_request(const std::string &request)
    {
        std::cout << "[INFO] Sending request to " << hostname << "..." << std::endl;
        if (SSL_write(ssl, request.c_str(), request.size()) <= 0)
        {
            ERR_print_errors_fp(stderr);
            throw std::runtime_error("[ERROR] Failed to send HTTP request!");
        }
    }

    void receive_response(const std::string &output_file)
    {
        std::cout << "[INFO] Receiving response from " << hostname << "..." << std::endl;
        std::ofstream ofs(output_file, std::ios::binary); // binary mode
        if (!ofs)
            throw std::runtime_error("[ERROR] Failed to open output file: " + output_file);

        char buffer[4096];
        int bytes_read;
        size_t total_bytes = 0;

        while ((bytes_read = SSL_read(ssl, buffer, sizeof(buffer))) > 0)
        {
            ofs.write(buffer, bytes_read);
            total_bytes += bytes_read;
        }

        if (bytes_read < 0)
        {
            ERR_print_errors_fp(stderr);
            ofs.close();
            throw std::runtime_error("[ERROR] Error while reading response");
        }

        ofs.close();
        std::cout << "[INFO] Response saved to " << output_file
                  << " (" << total_bytes << " bytes)" << std::endl;
    }

private:
    void init_openssl()
    {
        domain_manager = DomainManager::instance();
        SSL_library_init();
        SSL_load_error_strings();
        OpenSSL_add_all_algorithms();
    }

    void create_ssl_context()
    {
        const SSL_METHOD *method = TLS_client_method();
        ssl_ctx = SSL_CTX_new(method);
        if (!ssl_ctx)
        {
            ERR_print_errors_fp(stderr);
            throw std::runtime_error("[ERROR] Failed to create SSL context!");
        }
    }

    void create_socket_and_connect()
    {
        socket_fd = socket(AF_INET, SOCK_STREAM, 0); // IPv4, tcp
        if (socket_fd < 0)
        {
            throw std::runtime_error("[ERROR] Failed to create socket: " + std::string(strerror(errno)));
        }

        hostent *host = gethostbyname(hostname.c_str());
        if (!host)
        {
            throw std::runtime_error("[ERROR] Failed to resolve hostname: " + hostname);
        }

        sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        addr.sin_addr = *((in_addr *)host->h_addr);

        if (connect(socket_fd, (sockaddr *)&addr, sizeof(addr)) < 0)
        {
            throw std::runtime_error("[ERROR] Failed to connect to server: " +
                                     std::string(strerror(errno)));
        }
        std::cout << "[INFO] Connected to " << hostname << ":" << port << std::endl;
    }

    void establish_ssl_connection()
    {
        ssl = SSL_new(ssl_ctx);
        if (!ssl)
        {
            ERR_print_errors_fp(stderr);
            throw std::runtime_error("[ERROR] Failed to create SSL object!");
        }

        SSL_set_fd(ssl, socket_fd);
        SSL_set_tlsext_host_name(ssl, hostname.c_str()); // 设置SNI

        if (SSL_connect(ssl) <= 0)
        {
            ERR_print_errors_fp(stderr);
            throw std::runtime_error("[ERROR] Failed to establish SSL connection!");
        }

        std::cout << "[INFO] SSL connection established using " << SSL_get_cipher(ssl) << std::endl;

        // 输出证书信息
        X509 *cert = SSL_get_peer_certificate(ssl);
        if (cert)
        {
            std::cout << "[INFO] Server certificate verified" << std::endl;
            X509_free(cert);
        }
        else
        {
            std::cout << "[WARN] No server certificate presented" << std::endl;
        }
    }

    void ensure_output_directory()
    {
        struct stat st;
        if (stat("../data", &st) != 0)
        {
            if (mkdir("../data", 0755) != 0)
            {
                throw std::runtime_error("Failed to create data directory: " +
                                         std::string(strerror(errno)));
            }
            std::cout << "[INFO] Created output directory: ../data" << std::endl;
        }
    }

    void cleanup()
    {
        if (ssl)
        {
            SSL_shutdown(ssl);
            SSL_free(ssl);
            ssl = nullptr;
        }
        if (socket_fd >= 0)
        {
            close(socket_fd);
            socket_fd = -1;
        }
        if (ssl_ctx)
        {
            SSL_CTX_free(ssl_ctx);
            ssl_ctx = nullptr;
        }
    }
};

#endif // HTTPSCLIENT_HPP