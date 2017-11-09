#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define    FEATURE_FINISH_FLAG      "FEATURE_TRANSPORT_FINISH"
#define    TRANSPORT_FINISH_FLAG    "SYSTEM_TRANSPORT_FINISH"
#define    MAXLINE        1024

void usage(char *command)
{
    printf("usage :%s ipaddr portnum filename\n", command);
    exit(0);
}
int main(int argc, char **argv)
{
    FILE                   *fp;
    struct sockaddr_in     serv_addr;
    char                   buf[MAXLINE];
    int                    sock_id;
    int                    read_len;
    int                    send_len;
    int                    serv_addr_len;
    int                    i_ret;
    int                    i;

    if (argc != 4) {
        usage(argv[0]);
    }
    if ((fp = fopen(argv[3], "r")) == NULL) {
        perror("Open file failed\n");
        exit(0);
    }
    if ((sock_id = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("Create socket failed");
        exit(0);
    }
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(atoi(argv[2]));
    inet_pton(AF_INET, argv[1], &serv_addr.sin_addr);
    serv_addr_len = sizeof(serv_addr);
    i_ret = connect(sock_id, (struct sockaddr *)&serv_addr, sizeof(struct sockaddr));
    if (-1 == i_ret) {
        perror("Connect socket failed!\n");
        exit(0);
    }
    bzero(buf, MAXLINE);
    while ((read_len = fread(buf, sizeof(char), MAXLINE, fp)) > 0) {
        send_len = send(sock_id, buf, read_len, 0);
        if (send_len < 0) {
            perror("Send data failed\n");
            exit(0);
        }
        bzero(buf, MAXLINE);
    }
    fclose(fp);
    bzero(buf, MAXLINE);
    strcpy(buf, FEATURE_FINISH_FLAG);
    buf[strlen(buf)] = '\0';
    for (i = 1000; i>0; i--) {
        send_len = send(sock_id, buf, strlen(buf) + 1, 0);
        if (send_len < 0) {
            printf("Finish send the feature end string\n");
            break;
        }
    }

    bzero(buf, MAXLINE);
    strcpy(buf, TRANSPORT_FINISH_FLAG);
    buf[strlen(buf)] = '\0';
    for (i = 1000; i>0; i--) {
        send_len = send(sock_id, buf, strlen(buf) + 1, 0);
        if (send_len < 0) {
            printf("Finish send the transport end string\n");
            break;
        }
    }

    close(sock_id);
    printf("Send finish\n");
    return 0;
}
