#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/poll.h>
#include <time.h>
#include <unistd.h>
#include <vector>
#include <dirent.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "rtsp_demo.h"
#include "luckfox_mpi.h"
#include "retinaface_facenet.h"
#include "alignment.h"

#define DISP_WIDTH                  1280    // 720
#define DISP_HEIGHT                 1080    // 480
#define TEST_FUNCTION               1

#define FACENET_INPUT_WIDTH         160
#define FACENET_INPUT_HEIGHT        160

#define MAX_FACE_COUNT              3

MPP_CHN_S stSrcChn, stSrcChn1, stvpssChn, stvencChn;
VENC_RECV_PIC_PARAM_S stRecvParam;

rknn_app_context_t app_retinaface_ctx;
rknn_app_context_t app_facenet_ctx; 
object_detect_result_list od_results;

rtsp_demo_handle g_rtsplive = NULL;
rtsp_session_handle g_rtsp_session;

float reference_out_fp32[MAX_FACE_COUNT][128] = {0};
char* reference_name[MAX_FACE_COUNT][256] = {0};
int reference_count;
float fps;

void LoadReferenceFace()
{
    printf("LoadReferenceFace\n");

    const char* face_path = "./faces";
    DIR *dir = opendir(face_path);
    if (dir == NULL) {
        return;
    }

    struct dirent *entry;
    reference_count = 0;
    
    char path[256] = {0};

    int ret;
    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(".", entry->d_name) == 0 || strcmp("..", entry->d_name) == 0) {
            continue;
        }

        sprintf(path, "%s/%s", face_path, entry->d_name);
        printf("face feature name: %s\n", path);

        FILE* fp = fopen(path, "rb+");
        if (fp != NULL) {
            fread(reference_out_fp32[reference_count], sizeof(float), 128, fp);

            printf("%s, features: %f,%f,%f,%f,%f,%f,%f,%f\n",
                reference_name[reference_count], 
                reference_out_fp32[reference_count][0], 
                reference_out_fp32[reference_count][1], 
                reference_out_fp32[reference_count][2], 
                reference_out_fp32[reference_count][3], 
                reference_out_fp32[reference_count][4],
                reference_out_fp32[reference_count][5],
                reference_out_fp32[reference_count][6],
                reference_out_fp32[reference_count][7]
            );

            fclose(fp);
            memcpy(reference_name[reference_count], entry->d_name, strlen(entry->d_name) - 4);
        } else {
            printf("Read face feature failed.\n");
        }

        reference_count++;
        if (reference_count >= MAX_FACE_COUNT) {
            break;
        }
    }

    closedir(dir);

    printf("LoadReferenceFace Succ.\n");
}

static void* GetMediaBuffer(void *arg) {
    (void)arg;
    printf("========%s========\n", __func__);
    void *pData = RK_NULL;

    int s32Ret;

    VENC_STREAM_S stFrame;
    stFrame.pstPack = (VENC_PACK_S *) malloc(sizeof(VENC_PACK_S));

    while (1) {
        s32Ret = RK_MPI_VENC_GetStream(0, &stFrame, -1);
        if (s32Ret == RK_SUCCESS) {
            if (g_rtsplive && g_rtsp_session) {
                pData = RK_MPI_MB_Handle2VirAddr(stFrame.pstPack->pMbBlk);
                rtsp_tx_video(g_rtsp_session, (uint8_t *)pData, stFrame.pstPack->u32Len, stFrame.pstPack->u64PTS);

                rtsp_do_event(g_rtsplive);
            }

            s32Ret = RK_MPI_VENC_ReleaseStream(0, &stFrame);
            if (s32Ret != RK_SUCCESS) {
                RK_LOGE("RK_MPI_VENC_ReleaseStream fail %x", s32Ret);
            }
        }
        usleep(10 * 1000);
    }
    printf("\n======exit %s=======\n", __func__);
    free(stFrame.pstPack);

    return NULL;
}

static void *RetinaProcessBuffer(void *arg) {
    (void)arg;
    printf("========%s========\n", __func__);

    int disp_width  = DISP_WIDTH;
	int disp_height = DISP_HEIGHT;
	int model_width = 640;
	int model_height = 640;
	
	char text[16];	
	float scale_x = (float)disp_width / (float)model_width;  
	float scale_y = (float)disp_height / (float)model_height;   
	int sX,sY,eX,eY;
	int s32Ret;
	int group_count = 0;
	VIDEO_FRAME_INFO_S stViFrame;

    cv::Mat facenet_input(FACENET_INPUT_HEIGHT, FACENET_INPUT_WIDTH, CV_8UC3, app_facenet_ctx.input_mems[0]->virt_addr);
    float out_fp32[128] = {0.};
	
    while(1)
	{
		s32Ret = RK_MPI_VI_GetChnFrame(0, 1, &stViFrame, -1);
		if(s32Ret == RK_SUCCESS)
		{
			void *vi_data = RK_MPI_MB_Handle2VirAddr(stViFrame.stVFrame.pMbBlk);
			if(vi_data != RK_NULL)
			{
				cv::Mat yuv420sp(disp_height + disp_height / 2, disp_width, CV_8UC1, vi_data);
				cv::Mat bgr(disp_height, disp_width, CV_8UC3);			
				cv::Mat model_bgr(model_height, model_width, CV_8UC3);			

				cv::cvtColor(yuv420sp, bgr, cv::COLOR_YUV420sp2BGR);
				cv::resize(bgr, model_bgr, cv::Size(model_width ,model_height), 0, 0, cv::INTER_LINEAR);

				memcpy(app_retinaface_ctx.input_mems[0]->virt_addr, model_bgr.data, model_width * model_height * 3);
				inference_retinaface_model(&app_retinaface_ctx, &od_results);

                for(int i = 0; i < od_results.count; i++)
				{
                    if (i >= MAX_FACE_COUNT) {
                        break;
                    }

					object_detect_result *det_result = &(od_results.results[i]);
                    printf("Face index: %d, prop: %f\n", i, det_result->prop);                    

#ifdef TEST_FUNCTION
                    cv::imwrite("face_bgr.jpg",  bgr);
                    cv::imwrite("face_model.jpg",  model_bgr);

                    FILE* fp = fopen("face_points.txt", "w+");
                    if (fp != NULL) 
                    {
                        char str[500] = {0};
                        sprintf(str, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
                            det_result->point[0].x, det_result->point[0].y,
                            det_result->point[1].x, det_result->point[1].y,
                            det_result->point[2].x, det_result->point[2].y,
                            det_result->point[3].x, det_result->point[3].y,
                            det_result->point[4].x, det_result->point[4].y
                        );
                        fwrite(str, strlen(str), 1, fp);

                        sprintf(str, "%d,%d,%d,%d\n",
                            det_result->box.left, det_result->box.top,
                            det_result->box.right, det_result->box.bottom);
                        
                        fwrite(str, strlen(str), 1, fp);
                        fclose(fp);
                    };
#endif
                    // Face capture
                    cv::Rect roi(det_result->box.left, det_result->box.top, 
                                 (det_result->box.right - det_result->box.left),
                                 (det_result->box.bottom - det_result->box.top)); 

                    cv::Mat face_img = model_bgr(roi);
                    
#ifdef ALIGN_FACE
                    cv::Mat trans = getTransformMatrixSafas(det_result->box.left,
                                                            det_result->box.top,
                                                            &det_result->point[0]);
                    toTransform(face_img, 
                                facenet_input, 
                                trans, 
                                cv::Size(112, 112));
                    cv::resize(facenet_input, facenet_input, cv::Size(FACENET_INPUT_WIDTH, FACENET_INPUT_HEIGHT));
#else 
                    letterbox(face_img, facenet_input);
#endif

#ifdef TEST_FUNCTION
                    cv::imwrite("face_face.jpg", face_img);
                    cv::imwrite("face_input.jpg", facenet_input);
#endif
                    mapCoordinates(bgr, model_bgr, &det_result->box.left , &det_result->box.top);
                    mapCoordinates(bgr, model_bgr, &det_result->box.right, &det_result->box.bottom);

                    s32Ret = rknn_run(app_facenet_ctx.rknn_ctx, nullptr);
                    if (s32Ret < 0) {
                        printf("rknn_run fail! ret=%d\n", s32Ret);
                        continue;
                    }

                    uint8_t *output = (uint8_t *)(app_facenet_ctx.output_mems[0]->virt_addr);
                    output_normalization(&app_facenet_ctx, output, out_fp32);

#ifdef TEST_FUNCTION
                    FILE* fp2 = fopen("feature.dat", "wb+");
                    if (fp2 != NULL) {
                        fwrite(out_fp32, sizeof(float), 128, fp2);    
                        fclose(fp2);
                    }
#endif
                    float min_distance = 9999.0;
                    int min_idx = -1;
                    
                    for (int k = 0; k < reference_count; k++ )
                    {
                        // float norm = cosine_similarity(reference_out_fp32[k], out_fp32);
                        float norm = get_duclidean_distance(reference_out_fp32[k], out_fp32); 
                        printf("%s -> %.3f\n", reference_name[k], norm);
                        
                        if (norm < min_distance) {
                            min_idx = k;
                            min_distance = norm;
                        }
                    }

                    if (min_idx >= 0 && min_distance < 1.0) {
                        char text[15] = {0};
                        sX = (int)((float)det_result->box.left);	
                        sY = (int)((float)det_result->box.top);	
                        eX = (int)((float)det_result->box.right);	
                        eY = (int)((float)det_result->box.bottom);

                        sX = sX - (sX % 2);
                        sY = sY - (sY % 2);
                        eX = eX	- (eX % 2);
                        eY = eY	- (eY % 2);

                        test_rgn_overlay_line_process(sX, sY, 0, group_count);
                        test_rgn_overlay_line_process(eX, sY, 1, group_count);
                        test_rgn_overlay_line_process(eX, eY, 2, group_count);
                        test_rgn_overlay_line_process(sX, eY, 3, group_count);

                        sprintf(text, "%s %.2f", reference_name[min_idx], min_distance);
                        test_rgn_overlay_text_process(sX + 4, sY + 4, text, group_count);

                        // printf("Index: %d (%d %d %d %d) %.3f, %s\n",
                        //         i,
                        //         det_result->box.left ,det_result->box.top,
                        //         det_result->box.right,det_result->box.bottom,
                        //         min_distance, reference_name[min_idx]); 

                        group_count++;
                    }
				}
			}
			s32Ret = RK_MPI_VI_ReleaseChnFrame(0, 1, &stViFrame);
			if (s32Ret != RK_SUCCESS) {
				RK_LOGE("RK_MPI_VI_ReleaseChnFrame fail %x", s32Ret);
			}
		}
		else{
			printf("Get viframe error %d !\n", s32Ret);
			continue;
		}

		usleep(500000);
        for(int i = 0;i < group_count; i++) {
            rgn_overlay_release(i);
        }
        group_count = 0;
	}			
	
    return NULL;
}

int main(int argc, char* argv[])
{
    system("RkLunch-stop.sh");
    RK_S32 s32Ret = 0;

    int width       = DISP_WIDTH;
    int height      = DISP_HEIGHT;

    const char *model_path = "./model/retinaface.rknn";
    const char *model_path2 = "./model/mobilefacenet.rknn";
    
    memset(&app_retinaface_ctx, 0, sizeof(rknn_app_context_t));
    memset(&app_facenet_ctx, 0, sizeof(rknn_app_context_t));

    // init model    
    if (init_retinaface_facenet_model(model_path, model_path2, &app_retinaface_ctx, &app_facenet_ctx) != RK_SUCCESS) {
        RK_LOGE("rknn model init fail!");
        return -1;
    }

    LoadReferenceFace();

    // rkaiq init 
	RK_BOOL multi_sensor = RK_FALSE;	
	const char *iq_dir = "/etc/iqfiles";
	rk_aiq_working_mode_t hdr_mode = RK_AIQ_WORKING_MODE_NORMAL;
	//hdr_mode = RK_AIQ_WORKING_MODE_ISP_HDR2;
	SAMPLE_COMM_ISP_Init(0, hdr_mode, multi_sensor, iq_dir);
	SAMPLE_COMM_ISP_Run(0);

	// rkmpi init
	if (RK_MPI_SYS_Init() != RK_SUCCESS) {
		RK_LOGE("rk mpi sys init fail!");
		return -1;
	}

    // rtsp init	
	g_rtsplive = create_rtsp_demo(554);
	g_rtsp_session = rtsp_new_session(g_rtsplive, "/live/0");
	rtsp_set_video(g_rtsp_session, RTSP_CODEC_ID_VIDEO_H264, NULL, 0);
	rtsp_sync_video_ts(g_rtsp_session, rtsp_get_reltime(), rtsp_get_ntptime());

	// vi init
	vi_dev_init();
	vi_chn_init(0, width, height);
	vi_chn_init(1, width, height);
	
	// venc init
	RK_CODEC_ID_E enCodecType = RK_VIDEO_ID_AVC;
	venc_init(0, width, height, enCodecType);

	// bind vi to venc	
	stSrcChn.enModId = RK_ID_VI;
	stSrcChn.s32DevId = 0;
	stSrcChn.s32ChnId = 0;
		
	stvencChn.enModId = RK_ID_VENC;
	stvencChn.s32DevId = 0;
	stvencChn.s32ChnId = 0;
	printf("====RK_MPI_SYS_Bind vi0 to venc0====\n");
	s32Ret = RK_MPI_SYS_Bind(&stSrcChn, &stvencChn);
	if (s32Ret != RK_SUCCESS) {
		RK_LOGE("bind 1 ch venc failed");
		return -1;
	}
			
	printf("init success\n");	
	
	pthread_t main_thread;
	pthread_create(&main_thread, NULL, GetMediaBuffer, NULL);
	pthread_t retina_thread;
	pthread_create(&retina_thread, NULL, RetinaProcessBuffer, NULL);
	
	while (1) {		
		usleep(50000);
	}

	pthread_join(main_thread, NULL);
	pthread_join(retina_thread, NULL);

	RK_MPI_SYS_UnBind(&stSrcChn, &stvencChn);
	RK_MPI_VI_DisableChn(0, 0);
	RK_MPI_VI_DisableChn(0, 1);
	
	RK_MPI_VENC_StopRecvFrame(0);
	RK_MPI_VENC_DestroyChn(0);
	
	RK_MPI_VI_DisableDev(0);

	RK_MPI_SYS_Exit();

	// Stop RKAIQ
	SAMPLE_COMM_ISP_Stop(0);

	// Release rknn model
    release_facenet_model(&app_facenet_ctx);
    release_retinaface_model(&app_retinaface_ctx);

	return 0;
}