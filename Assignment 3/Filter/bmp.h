
#pragma pack(2)
struct bmp_header {
        uint16_t magic;
        uint32_t file_size;
        uint16_t reserved0;
        uint16_t reserved1;
        uint32_t offset;
};

#define BMP_MAGIC       0x424D

struct dib_header {
        uint32_t header_size;
        int32_t width;          /* Pixels */
        int32_t height; /* Pixels */
        uint16_t color_planes;
        uint16_t bpp;
        uint32_t compression;
        uint32_t image_size;
        int32_t horizontal;     /* Pixels per meter */
        int32_t vertical;               /* Pixels per meter */
        uint32_t colors;
        uint32_t important_colors;
};
#pragma pack()

#define BI_RGB          0x00
#define BI_RLE8 0x01
#define BI_RLE4 0x02
#define BI_BITFIELDS    0x03
#define BI_JPEG 0x04
#define BI_PNG          0x05

#define FATAL(msg)\
do {\
fprintf(stderr,"FATAL [%s:%d]:%s:%s\n", __FILE__, __LINE__, msg, strerror(errno)); \
assert(0); \
} while(0)

#define SRC 1
#define DST 2

#define BMP_SIZE 14
#define DIB_SIZE 40

#define COEFS_SIZE sizeof(float)*9

