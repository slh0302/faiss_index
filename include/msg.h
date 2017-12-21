# ifndef   _MSG_INFO_HPP_
# define   _MSG_INFO_HPP_

# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# define MAX_IP_SIZE			16
# define MAX_PATH_SIZE			32
# define MAX_FILE_NAME_SIZE		32
# define MAX_PKG_SIZE			40000

//  network message format
//	| datalen | data |
//	message foramt
//	| msg_flag | datalen | data |
//	feature message format
//	| FILE_FEATURE  | datalen | ServerIP | RelativePath | FileName | FrameNum | FrameType | BodingBoxNum | BoundingBoxData |ComressedFeatureData |
//	image message format
//	| FILE_PICTURE  | datalen | | ServerIP | RelativePath | FileName | FrameNum | FrameType | BodingBoxNum | BoundingBoxData | JpegData |
//	search result message format
//	| SEARCH_RESULT | datalen | ServerIP | RelativePath | FileName | FrameNum | FrameType | BodingBoxNum | BoundingBoxData |
typedef unsigned char uchar;
typedef unsigned int uint;
enum MSG_Flag //used for message type
{
	FORBIDDEN = 0,
	SEARCH_RESULT = 1, 
	FILE_FEATURE = 2,
	FILE_PICTURE = 3,
	FILE_VIDEO = 4,

	SEARCH_RESULT_OK = 5,
	FILE_FEATURE_OK = 6,
	FILE_PICTURE_OK = 7,
	FILE_VIDEO_OK = 8,

	MAX_MSG_FLAG = 9
};

typedef struct
{
	char ServerIP[MAX_IP_SIZE];//000.000.000.000
	char RelativePath[MAX_PATH_SIZE];
	char FileName[MAX_FILE_NAME_SIZE];
	unsigned int FrameNum;
	unsigned int FrameType;
	unsigned int BoundingBoxNum;
}FeatureMsgInfo;

typedef struct
{
	unsigned int	msg_flag;
	unsigned int	datalen;
}MSGPackageInfo;

class MSGPackage
{
public:
	unsigned int datalen;
	char data[MAX_PKG_SIZE];
};

int MSG_Package_Feature( const FeatureMsgInfo * fmi, const int ftr_length,const char * ftr, char * ret);

int MSG_Package_Retrival( const FeatureMsgInfo * fmi,int topLeftX,int topLeftY, int bottomRightX,int bottomRightY,char * ret );

int MSG_Package_Image   ( const FeatureMsgInfo * fmi,int topLeftX,int topLeftY, int bottomRightX,int bottomRightY, char * ret);

int MSG_Package_Feature( const FeatureMsgInfo * fmi, const int ftr_length, const char * ftr, char *ret )
{  
	
	MSGPackageInfo * pkginfo = (MSGPackageInfo *)ret;
	pkginfo->msg_flag = FILE_FEATURE;
	pkginfo->datalen = sizeof(FeatureMsgInfo) + ftr_length;

	FeatureMsgInfo * ftrinfo = (FeatureMsgInfo *)(ret + sizeof(MSGPackageInfo));
	memcpy(ftrinfo,fmi,sizeof(FeatureMsgInfo));

	memcpy( ret + sizeof(MSGPackageInfo) + sizeof(FeatureMsgInfo), ftr, ftr_length );

	return pkginfo->datalen + sizeof(MSGPackageInfo);
}

int MSG_Package_Retrival( const FeatureMsgInfo * fmi,int topLeftX,int topLeftY, int bottomRightX,int bottomRightY,char * ret )
{
	MSGPackageInfo * pkginfo = (MSGPackageInfo *)ret;
	pkginfo->msg_flag = SEARCH_RESULT;
	pkginfo->datalen = sizeof(FeatureMsgInfo)+fmi->BoundingBoxNum*4*sizeof(int);

	FeatureMsgInfo * ftrinfo = (FeatureMsgInfo *)(ret + sizeof(MSGPackageInfo));
	memcpy(ftrinfo,fmi,sizeof(FeatureMsgInfo));
	
	if(fmi->BoundingBoxNum == 1){
		char* featureData = ret + sizeof(MSGPackageInfo) + sizeof(FeatureMsgInfo);
		memcpy(featureData+0*sizeof(int),&topLeftX,sizeof(int));
		memcpy(featureData+1*sizeof(int),&topLeftY,sizeof(int));
		memcpy(featureData+2*sizeof(int),&bottomRightX,sizeof(int));
		memcpy(featureData+3*sizeof(int),&bottomRightY,sizeof(int));
	}

	return pkginfo->datalen + sizeof(MSGPackageInfo);
}

int MSG_Package_Image ( const FeatureMsgInfo * fmi, const char* prefix,int topLeftX,int topLeftY, int bottomRightX,int bottomRightY,char * ret )
{  
	MSGPackageInfo * pkginfo = (MSGPackageInfo *)ret;
	pkginfo->msg_flag = FILE_PICTURE;

	FeatureMsgInfo * ftrinfo = (FeatureMsgInfo *)(ret + sizeof(MSGPackageInfo));
	memcpy(ftrinfo,fmi,sizeof(FeatureMsgInfo));

	if(fmi->BoundingBoxNum == 1){
		char* featureData = ret + sizeof(MSGPackageInfo) + sizeof(FeatureMsgInfo);
		memcpy(featureData+0*sizeof(int),&topLeftX,sizeof(int));
		memcpy(featureData+1*sizeof(int),&topLeftY,sizeof(int));
		memcpy(featureData+2*sizeof(int),&bottomRightX,sizeof(int));
		memcpy(featureData+3*sizeof(int),&bottomRightY,sizeof(int));
	}

	int bytesread = 0;
	char * bin = ret + sizeof(MSGPackageInfo)+sizeof(FeatureMsgInfo)+fmi->BoundingBoxNum*4*sizeof(int);
	char imgname[256]={'\0'};
	//sprintf(imgname,"%s/%s/%s_%d_%d.jpg",prefix,fmi->RelativePath,fmi->FileName,fmi->FrameNum,fmi->FrameType);
	sprintf(imgname,"%s/%s/%s_%d.jpg",prefix,fmi->RelativePath,fmi->FileName,fmi->FrameNum);
	//sprintf(imgname,"%s/%s/%d.jpg",prefix,fmi->RelativePath,fmi->FrameNum);
	printf("filename=%s\n",imgname);
	FILE * img = fopen( imgname, "rb" );
	if(img == NULL)
	{
		pkginfo->datalen = 0;
		return sizeof(MSGPackageInfo);
	}
	while(!feof(img))
	{
		bytesread += fread( bin + bytesread,1,4096,img );
		//std::cout<<"read bytes="<<bytesread<<std::endl;
	}
	fclose(img);
	if( bytesread == 0 ){
		return -1;
	}
	pkginfo->datalen = bytesread + sizeof(FeatureMsgInfo) +fmi->BoundingBoxNum*4*sizeof(int);

	return pkginfo->datalen + sizeof(MSGPackageInfo) ;
}

#endif
