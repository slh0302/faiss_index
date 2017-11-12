#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/select.h>

#define    FEATURE_FINISH_FLAG      "FEATURE_TRANSPORT_FINISH"
#define    TRANSPORT_FINISH_FLAG    "SYSTEM_TRANSPORT_FINISH"
#define    MAXLINE        1024

void usage(char *command)
{
    printf("usage :%s portnum filename\n", command);
    exit(0);
}
int main(int argc, char **argv)
{
    struct sockaddr_in     serv_addr;
    struct sockaddr_in     clie_addr;
    char                   buf[MAXLINE];
    int                    sock_id;
    int                    recv_len;
    socklen_t              clie_addr_len;
    FILE                   *fp;
    char* tmp = new char[1024*8*10];
    int totalen = 0 ;
    if (argc != 3) {
        usage(argv[0]);
    }
    if ((fp = fopen(argv[2], "w")) == NULL) {
        perror("Creat file failed");
        exit(0);
    }
    if ((sock_id = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("Create socket failed\n");
        exit(0);
    }
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(atoi(argv[1]));
    serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    if (bind(sock_id, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        perror("Bind socket faild\n");
        exit(0);
    }

    clie_addr_len = sizeof(clie_addr);
    bzero(buf, MAXLINE);
    int feature_len = 0;

    while (recv_len = recvfrom(sock_id, buf, MAXLINE, 0, (struct sockaddr *)&clie_addr, &clie_addr_len)) {
        if (recv_len < 0) {
            printf("Recieve data from client failed!\n");
            break;
        }
        if (strstr(buf, TRANSPORT_FINISH_FLAG) != NULL) {
            memset(tmp,0,sizeof(char)* 1024*8*10);
            printf("\nTRANSPORT_FINISH_FLAG\n");
            // stop connect
            continue;
        }
        if (strstr(buf, FEATURE_FINISH_FLAG) != NULL) {
            float* p = new float[1024];
            memcpy(p, tmp, sizeof(float) * 1024);
            for(int i=0;i<1024;i++){
                printf("%f ", p[i]);
            }
            printf("\n");
            delete p;
            //end of feature frame,we will get new feature next time
            feature_len = 0;
            printf("\nFinish receive transport_finish_flag\n");
        }
        else {
            //write received feature to file
            memcpy(tmp + feature_len, buf, recv_len);
            // TODO
            feature_len += recv_len;
        }
        bzero(buf, MAXLINE);
    }
    printf("Finish receive\n");
    fclose(fp);
    close(sock_id);
    return 0;
}
