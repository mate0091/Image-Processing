// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <math.h>
#include <queue>

#define MAX_SIZE 20


void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

const Mat negative(const Mat src)
{
	int height = src.rows;
	int width = src.cols;
	Mat dst(height, width, CV_8UC1);
	
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			uchar val = src.at<uchar>(i, j);
			uchar neg = 255 - val;
			dst.at<uchar>(i,j) = neg;
		}
	}

	return dst;
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}

void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_COLOR);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

void changeGrayLevels()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst(height, width, CV_8UC1);

		int brightness = 0;

		printf("+Brightness?\n");
		scanf("%d", &brightness);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar newPixel;

				if (val + brightness < 0) newPixel = 0;
				else if (val + brightness > 255)
				{
					newPixel = 255;
				}

				else
				{
					newPixel = val + brightness;
				}

				dst.at<uchar>(i, j) = newPixel;
			}
		}

		imshow("Original", src);
		imshow("Changed brightness", dst);
		waitKey();
	}
}

void four_square_color_image()
{
	double t = (double)getTickCount(); // Get the current time [s]
	int height = 500;
	int width = 700;
	Mat dst(height, width, CV_8UC3);
	// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
	// Varianta ineficienta (lenta)
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (i < height / 2)
			{
				if (j < width / 2)
				{
					//upperLeft
					dst.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
				}

				else
				{
					//upperRight
					dst.at<Vec3b>(i, j) = Vec3b(0, 0, 255);
				}
			}

			else
			{
				if (j < width / 2)
				{
					dst.at<Vec3b>(i, j) = Vec3b(0, 255, 0);
				}

				else
				{
					dst.at<Vec3b>(i, j) = Vec3b(0, 255, 255);
				}
			}
		}
	}

	// Get the current time again and compute the time difference [s]
	t = ((double)getTickCount() - t) / getTickFrequency();
	// Print (in the console window) the processing time in [ms] 
	printf("Time = %.3f [ms]\n", t * 1000);

	imshow("constructed image", dst);
	waitKey();
}

void inverse_matrix()
{
	float vals[9] = { 1.0f, 2.0f, 3.0f, 0.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };
	Mat matrix(3, 3, CV_32FC1, vals);

	Mat inv = matrix.inv();

	std::cout << inv << std::endl;
	getchar();
	getchar();
}

void rgb24_splitchannels()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;

		Mat iRed(height, width, CV_8UC3);
		Mat iGreen(height, width, CV_8UC3);
		Mat iBlue(height, width, CV_8UC3);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b p = src.at<Vec3b>(i, j);
				uchar B = p[0];
				uchar G = p[1];
				uchar R = p[2];

				iBlue.at<Vec3b>(i, j) = Vec3b(B, 0, 0);
				iGreen.at<Vec3b>(i, j) = Vec3b(0, G, 0);
				iRed.at<Vec3b>(i, j) = Vec3b(0, 0, R);
			}
		}

		imshow("source", src);
		imshow("red", iRed);
		imshow("blue", iBlue);
		imshow("green", iGreen);
		waitKey();
	}
}

void color_2_grayscale()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;

		Mat dst(height, width, CV_8UC1);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b p = src.at<Vec3b>(i, j);
				uchar B = p[0];
				uchar G = p[1];
				uchar R = p[2];

				dst.at<uchar>(i, j) = (uchar)((B + G + R) / 3);
			}
		}

		imshow("source", src);
		imshow("dest", dst);
		waitKey();
	}
}

void grayscale_2_blacknwhite()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		Mat dst(height, width, CV_8UC1);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar p = src.at<uchar>(i, j);

				dst.at<uchar>(i, j) = (p > (uchar)120)? (uchar) 255 : (uchar) 0;
			}
		}

		imshow("source", src);
		imshow("dest", dst);
		waitKey();
	}
}

void rgb_2_hsv()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;

		Mat iH(height, width, CV_8UC1);
		Mat iS(height, width, CV_8UC1);
		Mat iV(height, width, CV_8UC1);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b p = src.at<Vec3b>(i, j);
				uchar B = p[0];
				uchar G = p[1];
				uchar R = p[2];

				float r = (float)R / 255;
				float g = (float)G / 255;
				float b = (float)B / 255;

				float M = max(max(r, g), b);
				float m = min(min(r, g), b);

				float C = M - m;

				//value
				float V = M;

				//saturation
				float S;

				if (V != 0.0f)
				{
					S = C / V;
				}

				else S = 0;

				//hue
				float H;

				if (C != 0)
				{
					if (M == r) H = 60 * (g - b) / C;
					if (M == g) H = 120 + 60 * (b - r) / C;
					if (M == b) H = 240 + 60 * (r - g) / C;
				}

				else
				{
					H = 0;
				}

				if (H < 0) H += 360;

				uchar H_norm = H * 255 / 360;
				uchar S_norm = S * 255;
				uchar V_norm = V * 255;

				iH.at<uchar>(i, j) = H_norm;
				iS.at<uchar>(i, j) = S_norm;
				iV.at<uchar>(i, j) = V_norm;
			}
		}

		imshow("source", src);
		imshow("Hue", iH);
		imshow("Saturation", iS);
		imshow("Value", iV);
		waitKey();
	}
}

void histogram(Mat src, int* h, float* p)
{
	int height = src.rows;
	int width = src.cols;
	int M = height * width;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			uchar current = src.at<uchar>(i, j);
			h[current]++;

			if (p != NULL)
			{
				p[current] = (float)h[current] / M;
			}
		}
	}
}

void histogram_show()
{
	//load grayscale
	//calculate h(g)
	//show histogram

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		int h[256] = { 0 };
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		
		histogram(src, h, NULL);

		showHistogram("Gray levels", h, 256, 200);

		waitKey();
	}
}

int nearest_max(int val, int max[], int n)
{
	int pos = 1;
	while (val > max[pos]) pos++;

	if (abs(max[pos] - val) < abs(max[pos - 1] - val)) return max[pos];
	return max[pos - 1];
}

void calc_maxima(Mat src, int* index, int* maxima, int WH, float TH)
{
	int h[256] = { 0 };
	float p[256];
	int ind = 0;
	
	maxima[ind++] = 0;

	histogram(src, h, p);

	for (int k = WH; k < 255 - WH; k++)
	{
		float avg = 0.0f;
		float currentMax = -1;

		for (int i = k - WH; i <= k + WH; i++)
		{
			avg += p[i];

			if (p[i] > currentMax) currentMax = p[i];
		}

		avg /= (float)2 * WH + 1;

		if (p[k] > avg + TH && currentMax == p[k])
		{
			maxima[ind++] = k;
		}
	}

	maxima[ind++] = 255;
	*index = ind;
}

void multilevel_thresholding(int WH, float TH)
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		//find nearest max for each pixel
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst(height, width, CV_8UC1);

		int maxima[256] = { 0 };
		int length;

		calc_maxima(src, &length, maxima, WH, TH);

		for (int i = 0; i < length; i++)
		{
			std::cout << maxima[i] << " ";
		}

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar current = src.at<uchar>(i, j);
				
				dst.at<uchar>(i, j) = (uchar)nearest_max(current, maxima, length);
			}
		}

		imshow("original", src);
		imshow("thresholded", dst);

		waitKey();
	}
}

void floyd_steinberg_dithering(int WH, float TH)
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst;
		src.copyTo(dst);

		int maxima[256] = { 0 };
		int length;

		calc_maxima(src, &length, maxima, WH, TH);

		for (int i = 0; i < length; i++)
		{
			std::cout << maxima[i] << " ";
		}

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar current = dst.at<uchar>(i, j);
				uchar newPixel = (uchar)nearest_max(current, maxima, length);

				dst.at<uchar>(i, j) = newPixel;
				float error = current - newPixel;

				if (j + 1 < width)
				{
					if ((int)dst.at<uchar>(i, j + 1) + 7 * error / 16 > 255)
					{
						dst.at<uchar>(i, j + 1) = (uchar) 255;
					}
					else if((int)dst.at<uchar>(i, j + 1) + 7 * error / 16 < 0)
					{
						dst.at<uchar>(i, j + 1) = (uchar) 0;
					}

					else
					{
						dst.at<uchar>(i, j + 1) += 7 * error / 16;
					}
				}

				if (i + 1 < height && j - 1 >= 0)
				{
					if ((int)dst.at<uchar>(i + 1, j - 1) + 3 * error / 16 > 255)
					{
						dst.at<uchar>(i + 1, j - 1) = (uchar)255;
					}
					else if ((int)dst.at<uchar>(i + 1, j - 1) + 3 * error / 16 < 0)
					{
						dst.at<uchar>(i + 1, j - 1) = (uchar)0;
					}

					else
					{
						dst.at<uchar>(i + 1, j - 1) += 3 * error / 16;
					}
				}

				if (i + 1 < height)
				{
					if ((int)dst.at<uchar>(i + 1, j) + 5 * error / 16 > 255)
					{
						dst.at<uchar>(i + 1, j) = (uchar)255;
					}
					else if ((int)dst.at<uchar>(i + 1, j) + 5 * error / 16 < 0)
					{
						dst.at<uchar>(i + 1, j) = (uchar)0;
					}

					else
					{
						dst.at<uchar>(i + 1, j) += 5 * error / 16;
					}
				}

				if (i + 1 < height && j + 1 < width)
				{
					if ((int)dst.at<uchar>(i + 1, j + 1) + error / 16 > 255)
					{
						dst.at<uchar>(i + 1, j + 1) = (uchar)255;
					}

					else if ((int)dst.at<uchar>(i + 1, j + 1) + error / 16 < 0)
					{
						dst.at<uchar>(i + 1, j + 1) = (uchar)0;
					}

					else
					{
						dst.at<uchar>(i + 1, j + 1) += error / 16;
					}
				}
			}
		}

		imshow("source", src);
		imshow("Floyd-Steinberg dithered", dst);

		waitKey();
	}
}

void geometrical_features(int event, int x, int y, int flags, void* param)
{
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDBLCLK)
	{
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);

		//double for + area(i, j) is equal to the color clicked

		Vec3b pix((*src).at<Vec3b>(y, x)[0], (*src).at<Vec3b>(y, x)[1], (*src).at<Vec3b>(y, x)[2]);

		int height = (*src).rows;
		int width = (*src).cols;
		int area = 0;
		int r = 0, c = 0;

		//Area
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if ((*src).at<Vec3b>(i, j) == pix)
				{
					area++;
					r += i;
					c += j;
				}
			}
		}
		
		printf("Area = %d\n", area);

		int center_r = r / area;
		int center_c = c / area;

		printf("Center of mass: (%d, %d)\n", center_r, center_c);

		//Axis of elongation
		float X = 0.0f;
		float Y = 0.0f;

		float s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if ((*src).at<Vec3b>(i, j) == pix)
				{
					s1 += (i - center_r) * (j - center_c);
					s2 += (j - center_c) * (j - center_c);
					s3 += (i - center_r) * (i - center_r);
				}
			}
		}

		Y = 2 * s1;
		X = s2 - s3;

		float axis_of_elong_rad = 0.5f * atan2(Y, X);

		if (axis_of_elong_rad < 0) axis_of_elong_rad += CV_PI;

		float axis_of_elong_deg = axis_of_elong_rad * (180 / CV_PI);

		printf("Axis of elongation: %.2f\n", axis_of_elong_deg);

		// Perimeter
		int P = 0;
		Mat dst;
		(*src).copyTo(dst);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if ((*src).at<Vec3b>(i, j) == pix)
				{
					bool flag = false;
					//scan neighbors
					if (i - 1 >= 0)
					{
						if ((*src).at<Vec3b>(i - 1, j) != pix)
						{
							flag = true;
						}

						if (j - 1 >= 0)
						{
							if ((*src).at<Vec3b>(i - 1, j - 1) != pix)
							{
								flag = true;
							}
						}

						if (j + 1 < width)
						{
							if ((*src).at<Vec3b>(i - 1, j + 1) != pix)
							{
								flag = true;
							}
						}
					}

					if (i + 1 < height)
					{
						if ((*src).at<Vec3b>(i + 1, j) != pix)
						{
							flag = true;
						}

						if (j - 1 >= 0)
						{
							if ((*src).at<Vec3b>(i + 1, j - 1) != pix)
							{
								flag = true;
							}
						}

						if (j + 1 < width)
						{
							if ((*src).at<Vec3b>(i + 1, j + 1) != pix)
							{
								flag = true;
							}
						}
					}

					if (j - 1 >= 0)
					{
						if ((*src).at<Vec3b>(i, j - 1) != pix)
						{
							flag = true;
						}
					}

					if (j + 1 < width)
					{
						if ((*src).at<Vec3b>(i, j + 1) != pix)
						{
							flag = true;
						}
					}

					if (flag)
					{
						P += 1;
						//draw pixel in dst
						dst.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
					}
				}
			}
		}

		imshow("Perimeter", dst);

		float NP = P * CV_PI / 4.0f;

		printf("Perimeter = %.2f\n", NP);

		//thinness ratio
		float T = 4 * CV_PI * (area / (NP * NP));
		printf("Thinness ratio = %.2f\n", T);

		//aspect ratio
		int r_min = 99999, c_min = 99999, r_max = -1, c_max = -1;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if ((*src).at<Vec3b>(i, j) == pix)
				{
					if (i < r_min) r_min = i;
					if (i > r_max) r_max = i;
					if (j < c_min) c_min = j;
					if (j > c_max) c_max = j;
				}
			}
		}

		float R = (float)(c_max - c_min + 1) / (r_max - r_min + 1);
		printf("Aspect ratio = %.2f\n", R);

		//Drawing axis of elongation
		int c_a = c_min;
		int r_a = center_r + tan(axis_of_elong_rad) * (c_min - center_c);
		int c_b = c_max;
		int r_b = center_r + tan(axis_of_elong_rad) * (c_max - center_c);

		Point A(c_a, r_a);
		Point B(c_b, r_b);

		Mat axis_of_elong_mat;
		(*src).copyTo(axis_of_elong_mat);

		line(axis_of_elong_mat, A, B, Scalar(0, 0, 0), 2);

		imshow("Axis of elongation", axis_of_elong_mat);

		// Horizontal projection
		Mat h_proj(height, width, CV_8UC3);
		int pos = 0;
		for (int i = 0; i < height; i++)
		{
			pos = 0;
			for (int j = 0; j < width; j++)
			{
				if ((*src).at<Vec3b>(i, j) == pix)
				{
					h_proj.at<Vec3b>(i, pos) = pix;
					pos++;
				}
			}
		}

		imshow("Horizontal projection", h_proj);

		// Vertical projection
		Mat v_proj(height, width, CV_8UC3);
		pos = 0;
		for (int j = 0; j < width; j++)
		{
			pos = 0;
			for (int i = 0; i < height; i++)
			{
				if ((*src).at<Vec3b>(i, j) == pix)
				{
					v_proj.at<Vec3b>(pos, j) = pix;
					pos++;
				}
			}
		}

		imshow("Vertical projection", v_proj);
	}
}

void l4_geometrical_features()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_COLOR);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", geometrical_features, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

void colorImage(char* windowName, Mat label, int nr)
{
	uchar* red = (uchar*) malloc(sizeof(uchar) * nr);
	uchar* green = (uchar*) malloc(sizeof(uchar) * nr);
	uchar* blue = (uchar*) malloc(sizeof(uchar) * nr);

	Mat dst(label.rows, label.cols, CV_8UC3, Scalar(255, 255, 255));
	srand(time(NULL));

	for (int i = 0; i < nr; i++)
	{
		red[i] = rand() % 255;
		green[i] = rand() % 255;
		blue[i] = rand() % 255;
	}

	for (int i = 0; i < label.rows; i++)
	{
		for (int j = 0; j < label.cols; j++)
		{
			int val = label.at<int>(i, j);
			if(val) dst.at<Vec3b>(i, j) = Vec3b(blue[val], green[val], red[val]);
		}
	}

	imshow(windowName, dst);
}

bool pointInside(int i, int j, int rows, int cols)
{
	if (i <= 0 || i >= rows) return false;
	if (j <= 0 || j >= cols) return false;
	return true;
}

void BFSTraversal()
{
	Mat src;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);

		int label = 0;
		int height = src.rows;
		int width = src.cols;
		Mat labels = Mat::zeros(height, width, CV_32SC1);

		int dy[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
		int dx[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };

		for (int i = 0; i < height - 1; i++)
		{
			for (int j = 0; j < width - 1; j++)
			{
				if (src.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0)
				{
					label++;
					std::queue<Point> Q;

					labels.at<int>(i, j) = label;

					Q.push(Point(j, i));

					while (!Q.empty())
					{
						Point q = Q.front();
						Q.pop();

						for (int k = 0; k < 8; k++)
						{
							int y = q.y + dx[k];
							int x = q.x + dy[k];

							if (pointInside(x, y, width, height) && src.at<uchar>(y, x) == 0 && labels.at<int>(y, x) == 0)
							{
								labels.at<int>(y, x) = label;
								Point neighbor(x, y);
								Q.push(neighbor);
							}
						}
					}
				}
			}
		}

		colorImage("BFS", labels, label);
	}
}

void twoPass()
{
	Mat src;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);

		int label = 0;
		int height = src.rows;
		int width = src.cols;
		Mat labels = Mat::zeros(height, width, CV_32SC1);

		int dy[4] = { -1, -1, -1, 0 };
		int dx[4] = { -1, 0, 1, -1 };

		std::vector<std::vector<int>> edges;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (src.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0)
				{
					std::vector<int> L;

					for (int k = 0; k < 4; k++)
					{
						int y = j + dx[k];
						int x = i + dy[k];

						if (pointInside(x, y, height, width) && labels.at<int>(x, y) > 0)
						{
							L.push_back(labels.at<int>(x, y));
						}
					}

					if (L.size() == 0)
					{
						label++;
						edges.resize(label + 1);
						labels.at<int>(i, j) = label;
					}

					else
					{
						int x = *std::min_element(L.begin(), L.end());
						labels.at<int>(i, j) = x;

						for (int n = 0; n < L.size(); n++)
						{
							int y = L[n];

							if (y != x)
							{
								edges[x].push_back(y);
								edges[y].push_back(x);
							}
						}
					}
				}
			}
		}

		int newLabel = 0;
		int* newLabels = (int*)malloc(sizeof(int) * (label + 1));

		for (int i = 0; i < label + 1; i++)
		{
			newLabels[i] = 0;
		}

		for (int i = 1; i < label + 1; i++)
		{
			if (newLabels[i] == 0)
			{
				newLabel++;
				
				std::queue<int> Q;
				newLabels[i] = newLabel;
				Q.push(i);

				while (!Q.empty())
				{
					int x = Q.front();
					Q.pop();

					for (int k = 0; k < edges[x].size(); k++)
					{
						int y = edges[x].at(k);

						if (newLabels[y] == 0)
						{
							newLabels[y] = newLabel;
							Q.push(y);
						}
					}
				}
			}
		}

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				labels.at<int>(i, j) = newLabels[labels.at<int>(i, j)];
			}
		}

		colorImage("Two pass", labels, newLabel);
	}
}

void border_tracing()
{
	Mat src;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);

		int label = 0;
		int height = src.rows;
		int width = src.cols;

		Mat dst(height, width, CV_8UC1, 255);

		int dy[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
		int dx[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };

		bool flag = false;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (src.at<uchar>(i, j) == 0)
				{
					std::vector<int> AC;
					std::vector<int> DC;

					std::vector<Point> P;
					P.push_back(Point(j, i));
					int dir = 7;
					int n = 0;

					do
					{
						if (dir % 2 == 0)
						{
							dir = (dir + 7) % 8;
						}

						else dir = (dir + 6) % 8;

						int x = P.at(n).y + dy[dir];
						int y = P.at(n).x + dx[dir];

						while (src.at<uchar>(x, y) != 0)
						{
							dir = (dir + 1) % 8;

							x = P.at(n).y + dy[dir];
							y = P.at(n).x + dx[dir];
						}

						//store pixel
						n++;
						P.push_back(Point(y, x));
						
						dst.at<uchar>(P.at(n).y, P.at(n).x) = 0;
						dst.at<uchar>(P.at(n - 1).y, P.at(n - 1).x) = 0;
						
						//chain codes
						AC.push_back(dir);

					} while ((n < 2) || (P.at(n) != P.at(1)) || (P.at(n - 1) != P.at(0)));

					printf("Start point: (%d, %d)\n", P.at(0).x, P.at(0).y);

					printf("AC:\n");

					for (int k = 0; k < AC.size() - 1; k++)
					{
						printf("%d ", AC.at(k));
					}

					for (int k = 0; k < AC.size() - 1; k++)
					{
						DC.push_back((AC.at(k + 1) - AC.at(k) + 8) % 8);
					}

					printf("\nDC:\n");

					for (int k = 0; k < DC.size() - 1; k++)
					{
						printf("%d ", DC.at(k));
					}

					flag = true;
				}

				if (flag) break;
			}

			if (flag) break;
		}

		imshow("Contour", dst);
	}
}

void reconstruct_AC()
{
	FILE *file = fopen("Images/reconstruct.txt", "r");
	//parse to points
	int n;
	int x, y;

	fscanf(file, "%d %d\n", &x, &y);
	fscanf(file, "%d\n", &n);
	
	int dy[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dx[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };

	Mat dst(400, 600, CV_8UC1, 255);

	for (int i = 0; i < n; i++)
	{
		int dir;
		fscanf(file, "%d ", &dir);

		x += dx[dir];
		y += dy[dir];

		dst.at<uchar>(y, x) = 0;
	}

	imshow("Reconstructed", dst);

	waitKey();
}

const Mat dilate(const Mat src)
{
	Mat dst;
	src.copyTo(dst);

	int height = dst.rows;
	int width = dst.cols;

	int str_x[5] = { 0, 1, 0, -1, 0};
	int str_y[5] = { 0, 0, 1, 0, -1 };

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (src.at<uchar>(i, j) == 0)
			{
				Point p(0, 0);

				for (int k = 0; k < 5; k++)
				{
					p.x = j + str_x[k];
					p.y = i + str_y[k];

					if (pointInside(p.y, p.x, height, width))
					{
						dst.at<uchar>(p.y, p.x) = 0;
					}
				}
			}
		}
	}

	return dst;
}

const Mat erode(const Mat src)
{
	int height = src.rows;
	int width = src.cols;

	Mat dst(height, width, CV_8UC1, 255);

	int str_x[5] = { 0, 1, 0, -1, 0 };
	int str_y[5] = { 0, 0, 1, 0, -1 };

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (src.at<uchar>(i, j) == 0)
			{
				Point p(0, 0);

				bool flag = true;

				for (int k = 0; k < 5; k++)
				{
					p.x = j + str_x[k];
					p.y = i + str_y[k];

					if (pointInside(p.y, p.x, height, width))
					{
						if (src.at<uchar>(p.y, p.x) != 0) //any background point
						{
							flag = false;
							break;
						}
					}
				}

				if (flag)
				{
					dst.at<uchar>(i, j) = 0;
				}
			}
		}
	}

	return dst;
}

const Mat opened(const Mat src)
{
	return dilate(erode(src));
}

const Mat closed(const Mat src)
{
	return erode(dilate(src));
}

void dilate()
{
	Mat src;
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);

		Mat temp = src.clone();
		Mat dst;

		int n = 0;

		printf("How many times?\n");
		scanf("%d", &n);
		waitKey();

		while (n > 0)
		{
			dst = dilate(temp);
			dst.copyTo(temp);

			n--;
		}

		imshow("Original", src);
		imshow("Dilation", dst);
	}
}

void erode()
{
	Mat src;
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);

		Mat temp = src.clone();
		Mat dst;

		int n = 0;

		printf("How many times?\n");
		scanf("%d", &n);
		waitKey();

		while (n > 0)
		{
			dst = erode(temp);
			dst.copyTo(temp);

			n--;
		}

		imshow("Original", src);
		imshow("Erosion", dst);
	}
}

void opening()
{
	Mat src;
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);

		Mat dst = opened(src);

		imshow("Original", src);
		imshow("Opened", dst);
	}
}

void closing()
{
	Mat src;
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);

		Mat dst = closed(src);

		imshow("Original", src);
		imshow("Closed", dst);
	}
}

const Mat diff(const Mat m1, const Mat m2)
{
	int height = m1.rows;
	int width = m1.cols;

	Mat dst(height, width, CV_8UC1, 255);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			uchar p1 = m1.at<uchar>(i, j);
			uchar p2 = m2.at<uchar>(i, j);

			if (p1 == 0 && p2 != 0)
			{
				dst.at<uchar>(i, j) = 0;
			}
		}
	}

	return dst;
}

const Mat intersect(const Mat m1, const Mat m2)
{
	int height = m1.rows;
	int width = m1.cols;

	Mat dst(height, width, CV_8UC1, 255);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			uchar p1 = m1.at<uchar>(i, j);
			uchar p2 = m2.at<uchar>(i, j);

			if (p1 == 0 && p2 == 0)
			{
				dst.at<uchar>(i, j) = 0;
			}
		}
	}

	return dst;
}

const Mat unite(const Mat m1, const Mat m2)
{
	int height = m1.rows;
	int width = m1.cols;

	Mat dst(height, width, CV_8UC1, 255);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			uchar p1 = m1.at<uchar>(i, j);
			uchar p2 = m2.at<uchar>(i, j);

			if (p1 == 0 || p2 == 0)
			{
				dst.at<uchar>(i, j) = 0;
			}
		}
	}

	return dst;
}

void boundary_extraction()
{
	Mat src;
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);

		Mat temp = erode(src);
		Mat dst = diff(src, temp);

		imshow("Original", src);
		imshow("Boundary", dst);
	}
}

bool content_equals(const Mat m1, const Mat m2)
{
	int height = m1.rows;
	int width = m1.cols;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			uchar p1 = m1.at<uchar>(i, j);
			uchar p2 = m2.at<uchar>(i, j);

			if (p1 != p2)
			{
				return false;
			}
		}
	}

	return true;
}

void region_filling()
{
	Mat src;
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;

		Mat neg = negative(src);
		Mat prev(height, width, CV_8UC1, 255);
		Mat current(height, width, CV_8UC1, 255);

		current.at<uchar>(height / 2, width / 2) = 0;

		while (!content_equals(prev, current))
		{
			prev = current;
			current = intersect(dilate(prev), neg);
		}

		Mat dst = unite(src, current);

		imshow("Original", src);
		imshow("Region Filling", dst);
	}
}

const Mat zoom_nearest_neighbor(const Mat src, const float zoomX, const float zoomY)
{
	int height = src.rows;
	int width = src.cols;

	int newH = height * zoomX;
	int newW = width * zoomY;

	Mat dst(newH, newW, CV_8UC1, 255);

	for (int i = 0; i < newH; i++)
	{
		for (int j = 0; j < newW; j++)
		{
			dst.at<uchar>(i, j) = src.at<uchar>(i / zoomX, j / zoomY);
		}
	}

	return dst;
}

const Mat zoom_bilinear(const Mat src, const float zoomX, const float zoomY)
{
	int height = src.rows;
	int width = src.cols;

	int newH = floor(height * zoomX);
	int newW = floor(width * zoomY);

	Mat dst(height * zoomX, width * zoomY, CV_8UC1, 255);
	Mat src2;

	copyMakeBorder(src, src2, 0, zoomX, 0, zoomY, BorderTypes::BORDER_REPLICATE);

	for (int i = 0; i < newH; i++)
	{
		int x1 = floor(i / zoomX);
		int x2 = x1 + 1;

		float x = (i / zoomX) - floor(i / zoomX);

		for (int j = 0; j < newW; j++)
		{
			int y1 = floor(j / zoomY);
			int y2 = y1 + 1;

			float y = (j / zoomY) - floor(j / zoomY);

			if (pointInside(x2, y2, src2.rows, src2.cols))
			{
				uchar topLeft = src2.at<uchar>(x1, y1);
				uchar topRight = src2.at<uchar>(x1, y2);
				uchar bottomLeft = src2.at<uchar>(x2, y1);
				uchar bottomRight = src2.at<uchar>(x2, y2);

				uchar top = (topRight * y) + (topLeft * (1 - y));
				uchar bottom = (bottomRight * y) + (bottomLeft * (1 - y));

				dst.at<uchar>(i, j) = (bottom * x) + (top * (1 - x));
			}
		}
	}	

	return dst;
}

float cubicInterpolate(float p[4], float x)
{
	return p[1] + 0.5 * x*(p[2] - p[0] + x * (2.0*p[0] - 5.0*p[1] + 4.0*p[2] - p[3] + x * (3.0*(p[1] - p[2]) + p[3] - p[0])));
}

float bicubicInterpolate(float p[4][4], float x, float y)
{
	float arr[4];
	arr[0] = cubicInterpolate(p[0], y);
	arr[1] = cubicInterpolate(p[1], y);
	arr[2] = cubicInterpolate(p[2], y);
	arr[3] = cubicInterpolate(p[3], y);

	return cubicInterpolate(arr, x);
}

const Mat zoom_bicubic(const Mat src, const float zoomX, const float zoomY)
{
	int height = src.rows;
	int width = src.cols;

	int newH = floor(height * zoomX);
	int newW = floor(width * zoomY);

	Mat dst(height * zoomX, width * zoomY, CV_8UC1, 255);
	Mat src2;

	copyMakeBorder(src, src2, 0, 3 * zoomX, 0, 3 * zoomY, BorderTypes::BORDER_REPLICATE);

	for (int i = 0; i < newH; i++)
	{
		for (int j = 0; j < newW; j++)
		{
			float y = (j / zoomY) - floor(j / zoomY);

			int factory = floor(j / zoomY);

			int y_coord[4];

			y_coord[0] = factory;
			y_coord[1] = factory + 1;
			y_coord[2] = factory + 2;
			y_coord[3] = factory + 3;

			int factor = floor(i / zoomX);

			int x_coord[4];

			x_coord[0] = factor;
			x_coord[1] = factor + 1;
			x_coord[2] = factor + 2;
			x_coord[3] = factor + 3;

			float x = (i / zoomX) - floor(i / zoomX);
			
			float poly[4][4];

			for (int m = 0; m < 4; m++)
			{
				for (int n = 0; n < 4; n++)
				{
					poly[m][n] = src2.at<uchar>(x_coord[m], y_coord[n]);
				}
			}

			float val = bicubicInterpolate(poly, x, y);

			if (val < 0) val = 0;
			else if (val > 255) val = 255;

			dst.at<uchar>(i, j) = (uchar)val;
		}
	}

	return dst;
}

void zoom()
{
	float zoomX = 1.3f;
	float zoomY = 2.7f;

	Mat src;
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);

		printf("Zoom_x, Zoom_y\n");
		scanf("%f", &zoomX);
		scanf("%f", &zoomY);

		Mat bilinear = zoom_bilinear(src, zoomX, zoomY);
		Mat nearest_neighbor = zoom_nearest_neighbor(src, zoomX, zoomY);
		Mat bicubic = zoom_bicubic(src, zoomX, zoomY);

		imshow("Original", src);
		imshow("With Bilinear interpolation", bilinear);
		imshow("Nearest neighbor", nearest_neighbor);
		imshow("With Bicubic", bicubic);
	}
}

void stats()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		int h[256] = { 0 };
		float p[256] = { 0 };

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		histogram(src, h, p);

		showHistogram("Gray levels", h, 256, 200);

		float mean = 0.0f;

		for (int i = 0; i < 256; i++)
		{
			mean += i * p[i];
		}

		printf("Mean: %.3f\n", mean);

		float std_dev = 0.0f;

		for (int i = 0; i < 256; i++)
		{
			std_dev += (i - mean) * (i - mean) * p[i];
		}

		std_dev = sqrt(std_dev);
		printf("Standard deviation: %.3f\n", std_dev);

		waitKey();
	}
}

void global_tresholding()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		int h[256] = { 0 };
		float p[256] = { 0 };

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		histogram(src, h, p);

		showHistogram("Gray levels", h, 256, 200);

		int min_intensity;
		int max_intensity;

		for (int i = 0; i < 256; i++)
		{
			if (p[i] > 0)
			{
				min_intensity = i;
				break;
			}
		}

		for (int i = 255; i > 0; i--)
		{
			if (p[i] > 0)
			{
				max_intensity = i;
				break;
			}
		}

		printf("min intensity = %d, max_intensity = %d\n", min_intensity, max_intensity);

		float Tprev = 0.0f;
		float Tcurrent;

		Tcurrent = (min_intensity + max_intensity) / 2.0f;

		while (abs(Tprev - Tcurrent) > 0.1f)
		{
			int N1, N2;
			float mean1, mean2;

			N1 = 0;
			for (int i = min_intensity; i <= floor(Tcurrent); i++)
			{
				N1 += h[i];
			}

			N2 = 0;
			for (int i = floor(Tcurrent) + 1; i <= max_intensity; i++)
			{
				N2 += h[i];
			}

			mean1 = 0.0f;
			mean2 = 0.0f;

			for (int i = min_intensity; i <= floor(Tcurrent); i++)
			{
				mean1 += i * h[i];
			}

			mean1 /= (float)N1;

			for (int i = floor(Tcurrent) + 1; i <= max_intensity; i++)
			{
				mean2 += i * h[i];
			}

			mean2 /= (float)N2;

			Tprev = Tcurrent;
			Tcurrent = (mean1 + mean2) / 2.0f;
		}

		printf("T = %.3f\n", Tcurrent);

		Mat dst(height, width, CV_8UC1, 255);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (src.at<uchar>(i, j) < (int)Tcurrent)
				{
					dst.at<uchar>(i, j) = 0;
				}

				else
				{
					dst.at<uchar>(i, j) = 255;
				}
			}
		}

		imshow("Original", src);
		imshow("Thresholded", dst);
	}
}

void contrast_change()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		int h[256] = { 0 };
		float p[256] = { 0 };
		
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int width = src.cols;
		int height = src.rows;
		histogram(src, h, p);

		Mat dst(height, width, CV_8UC1, 255);

		int gOutMin = 0;
		int gOutMax = 0;
		int gInMin = 0;
		int gInMax = 0;

		printf("Min, max values?\n");
		scanf("%d, %d", &gOutMin, &gOutMax);

		for (int i = 0; i < 256; i++)
		{
			if (p[i] > 0)
			{
				gInMin = i;
				break;
			}
		}

		for (int i = 255; i > 0; i--)
		{
			if (p[i] > 0)
			{
				gInMax = i;
				break;
			}
		}

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar gIn = src.at<uchar>(i, j);

				int temp = gOutMin + (gIn - gInMin) * (float)((gOutMax - gOutMin) / (gInMax - gInMin));

				if (temp > 255) temp = 255;
				else if (temp < 0) temp = 0;

				dst.at<uchar>(i, j) = temp;
			}
		}

		imshow("Original", src);
		imshow("Constrast changed", dst);
	}
}

void gamma_correction()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int width = src.cols;
		int height = src.rows;

		Mat dst(height, width, CV_8UC1, 255);

		float gamma;

		printf("Gamma?\n");
		scanf("%f", &gamma);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar gIn = src.at<uchar>(i, j);

				float frac = (float)gIn / 255.0f;

				int temp = 255 * pow(frac, gamma);

				if (temp > 255) temp = 255;
				else if (temp < 0) temp = 0;

				dst.at<uchar>(i, j) = temp;
			}
		}

		imshow("Original", src);
		imshow("Gamma corrected", dst);
	}
}

void histogram_eq()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		int h[256] = { 0 };
		float p[256] = { 0 };

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int width = src.cols;
		int height = src.rows;
		histogram(src, h, p);

		Mat dst(height, width, CV_8UC1, 255);

		float pc[256] = { 0 };

		for (int i = 0; i < 256; i++)
		{
			float val = 0.0f;

			for (int j = 0; j <= i; j++)
			{
				val += p[j];
			}

			pc[i] = val;
		}

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar gIn = src.at<uchar>(i, j);

				int temp = 255 * pc[gIn];

				if (temp > 255) temp = 255;
				else if (temp < 0) temp = 0;

				dst.at<uchar>(i, j) = temp;
			}
		}

		imshow("Original", src);
		imshow("Histogram Equalized", dst);
	}
}

Mat apply_filter(int filter[][MAX_SIZE], int filter_size, Mat src, bool lowpass)
{
	Mat dst;
	src.copyTo(dst);

	int width = dst.cols;
	int height = dst.rows;

	int k = (filter_size - 1) / 2;

	printf("k = %d\n", k);

	float sum = 0.0f;

	float S_minus = 0.0f;
	float S_plus = 0.0f;

	for (int i = 0; i < filter_size; i++)
	{
		for (int j = 0; j < filter_size; j++)
		{
			sum += filter[i][j];

			if (filter[i][j] > 0) S_plus += filter[i][j];
			else S_minus += -filter[i][j];
		}
	}

	float S = 1.0f / (2.0f * max(S_plus, S_minus));

	for (int i = k; i < height - k; i++)
	{
		for (int j = k; j < width - k; j++)
		{
			float dst_pix = 0;

			for (int u = 0; u < filter_size; u++)
			{
				for (int v = 0; v < filter_size; v++)
				{
					dst_pix += filter[u][v] * src.at<uchar>(i + u - k, j + v - k);
				}
			}

			if (lowpass) dst_pix /= sum;

			else
			{
				dst_pix = S * dst_pix + floor(255.0f / 2.0f);
			}

			//constrain
			if (dst_pix > 255.0f) dst_pix = 255.0f;
			else if (dst_pix < 0.0f) dst_pix = 0.0f;

			dst.at<uchar>(i, j) = (int)dst_pix;
		}
	}

	return dst;
}

void convolution_filters()
{
	int mean_3x3_filter[3][MAX_SIZE] = {
		{1, 1, 1},
		{1, 1, 1},
		{1, 1, 1} };

	int mean_5x5_filter[5][MAX_SIZE] = {
		{1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1}, 
		{1, 1, 1, 1, 1} };

	int gaussian_filter[3][MAX_SIZE] = {
		{1, 2, 1},
		{2, 4, 2},
		{1, 2, 1}
	};

	int laplace_filter[3][MAX_SIZE] = {
		{-1, -1, -1},
		{-1, 8, -1},
		{-1, -1, -1}
	};

	int high_pass_filter[3][MAX_SIZE] = {
		{-1, -1, -1},
		{-1, 9, -1},
		{-1, -1, -1}
	};

	Mat src;
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);
		Mat mean_3x3 = apply_filter(mean_3x3_filter, 3, src, true);
		Mat mean_5x5 = apply_filter(mean_5x5_filter, 5, src, true);
		Mat gaussian = apply_filter(gaussian_filter, 3, src, true);
		Mat laplace = apply_filter(laplace_filter, 3, src, false);
		Mat high_pass = apply_filter(high_pass_filter, 3, src, false);

		imshow("Original", src);
		imshow("Mean 3x3", mean_3x3);
		imshow("Mean 5x5", mean_5x5);
		imshow("Gaussian", gaussian);
		imshow("Laplace", laplace);
		imshow("High-pass", high_pass);
	}
}

void centering_transform(Mat img) 
{
	//expects floating point image
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img.at<float>(i, j) = ((i + j) & 1) ? -img.at<float>(i, j) : img.at<float>(i, j);
		}
	}
}

Mat freq_lowpass(Mat src)
{
	Mat dst;
	src.copyTo(dst);

	int height = dst.rows;
	int width = dst.cols;

	float R = 20.0f;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			float val = (height / 2.0 - i) * (height / 2.0 - i) + (width / 2.0 - j) * (width / 2.0 - j);

			
			if (val <= R * R)
			{
				dst.at<float>(i, j) = src.at<float>(i, j);
			}

			else
			{
				dst.at<float>(i, j) = 0.0f;
			}
		}
	}

	return dst;
}

Mat freq_highpass(Mat src)
{
	Mat dst;
	src.copyTo(dst);

	int height = dst.rows;
	int width = dst.cols;

	float R = 20.0f;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			float val = (height / 2.0 - i) * (height / 2.0 - i) + (width / 2.0 - j) * (width / 2.0 - j);
			
			if (val > R * R)
			{
				dst.at<float>(i, j) = src.at<float>(i, j);
			}

			else
			{
				dst.at<float>(i, j) = 0.0f;
			}
		}
	}

	return dst;
}

Mat freq_gaussian_lowpass(Mat src)
{
	Mat dst;
	src.copyTo(dst);

	int height = dst.rows;
	int width = dst.cols;

	float A = 20.0f;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			float val = (height / 2.0 - i) * (height / 2.0 - i) + (width / 2.0 - j) * (width / 2.0 - j);
			dst.at<float>(i, j) = src.at<float>(i, j) * exp(-(val / (A * A)));
		}
	}

	return dst;
}

Mat freq_gaussian_highpass(Mat src)
{
	Mat dst;
	src.copyTo(dst);

	int height = dst.rows;
	int width = dst.cols;

	float A = 20.0f;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			float val = (height / 2.0 - i) * (height / 2.0 - i) + (width / 2.0 - j) * (width / 2.0 - j);
			dst.at<float>(i, j) = src.at<float>(i, j) * (1.0f - exp(-(val / (A * A))));
		}
	}

	return dst;
}

Mat generic_frequency_domain_filter(Mat src, Mat (*filter)(Mat), bool logarithm_of_magnitude) {
	//convert input image to float image
	Mat srcf;
	src.convertTo(srcf, CV_32FC1);
	//centering transformation
	centering_transform(srcf);

	//perform forward transform with complex image output
	Mat fourier;
	dft(srcf, fourier, DFT_COMPLEX_OUTPUT);

	//split into real and imaginary channels
	Mat channels[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
	split(fourier, channels); // channels[0] = Re(DFT(I)), channels[1] = Im(DFT(I))

	//calculate magnitude and phase in floating point images mag and phi
	Mat mag, phi;
	magnitude(channels[0], channels[1], mag);
	phase(channels[0], channels[1], phi);

	//display the phase and magnitude images here
	if (logarithm_of_magnitude) 
	{
		Mat log_of_mag;
		src.copyTo(log_of_mag);

		float maxLog = -1.0f;

		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				float val = log(mag.at<float>(i, j));
				if (val > maxLog) maxLog = val;
			}
		}

		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				float val = log(mag.at<float>(i, j) + 1) / maxLog;
				log_of_mag.at<uchar>(i, j) = val * 255;
			}
		}

		return log_of_mag;
	}

	//insert filtering operations on Fourier coefficients here
	
	channels[0] = filter(channels[0]);
	channels[1] = filter(channels[1]);

	//store in real part in channels[0] and imaginary part in channels[1]
	// ......
	//perform inverse transform and put results in dstf
	Mat dst, dstf;
	merge(channels, 2, fourier);
	dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
	//inverse centering transformation
	centering_transform(dstf);
	//normalize the result and put in the destination image
	normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);
	//Note: normalizing distorts the resut while enhancing the image display in the range [0,255].
	//For exact results (see Practical work 3) the normalization should be replaced with convertion:
	//dstf.convertTo(dst, CV_8UC1);
	return dst;
}

void freq_filters()
{
	Mat src;
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);

		Mat log_of_mag = generic_frequency_domain_filter(src, nullptr, true);
		Mat ideal_lowpass = generic_frequency_domain_filter(src, freq_lowpass, false);
		Mat ideal_highpass = generic_frequency_domain_filter(src, freq_highpass, false);
		Mat gaussian_lowpass = generic_frequency_domain_filter(src, freq_gaussian_lowpass, false);
		Mat gaussian_highpass = generic_frequency_domain_filter(src, freq_gaussian_highpass, false);

		imshow("Original", src);
		imshow("Logarithm of magnitude", log_of_mag);
		imshow("Ideal low-pass", ideal_lowpass);
		imshow("Ideal high-pass", ideal_highpass);
		imshow("Gaussian low-pass", gaussian_lowpass);
		imshow("Gaussian high-pass", gaussian_highpass);
	}
}

Mat apply_median_filter(Mat src, int w)
{
	Mat dst;
	src.copyTo(dst);

	int width = dst.cols;
	int height = dst.rows;

	int k = (w - 1) / 2;

	for (int i = k; i < height - k; i++)
	{
		for (int j = k; j < width - k; j++)
		{
			std::vector<uchar> arr;

			for (int u = 0; u < w; u++)
			{
				for (int v = 0; v < w; v++)
				{
					arr.push_back(src.at<uchar>(i + u - k, j + v - k));
				}
			}

			std::sort(arr.begin(), arr.end());

			dst.at<uchar>(i, j) = arr.at(w * w / 2);
		}
	}

	return dst;
}

void median_filter()
{
	Mat src;
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);

		printf("w?\n");

		int w;
		scanf("%d", &w);

		double t = (double)getTickCount();
		Mat dst = apply_median_filter(src, w);
		t = ((double)getTickCount() - t) / getTickFrequency();
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("Original", src);
		imshow("Median filtered", dst);
	}
}

Mat apply_gauss2D_filter(Mat src, float gamma)
{
	Mat dst;
	src.copyTo(dst);

	int width = dst.cols;
	int height = dst.rows;

	//generate gaussian filter
	int w = gamma * 6;
	if (w % 2 == 0) w += 1;
	float G[MAX_SIZE][MAX_SIZE] = { 0 };

	printf("w=%d\n", w);

	float S = 0.0f;

	for (int i = 0; i < w; i++)
	{
		for (int j = 0; j < w; j++)
		{
			G[i][j] = (1.0f / (2.0f * PI * gamma * gamma)) * exp(-(((i - w / 2) * (i - w / 2) + (j - w / 2) * (j - w / 2)) / (2 * gamma * gamma)));
			S += G[i][j];
		}
	}

	printf("Gaussian filter:\n");

	for (int i = 0; i < w; i++)
	{
		for (int j = 0; j < w; j++)
		{
			printf("%.5f ", G[i][j]);
		}

		printf("\n");
	}

	printf("\nSum = %.3f\n", S);

	int k = (w - 1) / 2;

	for (int i = k; i < height - k; i++)
	{
		for (int j = k; j < width - k; j++)
		{
			float dst_pix = 0;

			for (int u = 0; u < w; u++)
			{
				for (int v = 0; v < w; v++)
				{
					dst_pix += G[u][v] * src.at<uchar>(i + u - k, j + v - k);
				}
			}

			dst_pix /= S;

			//constrain
			if (dst_pix > 255.0f) dst_pix = 255.0f;
			else if (dst_pix < 0.0f) dst_pix = 0.0f;

			dst.at<uchar>(i, j) = (int)dst_pix;
		}
	}

	return dst;
}

void gauss2D_filter()
{
	Mat src;
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);

		printf("gamma?\n");

		float w;
		scanf("%f", &w);

		double t = (double)getTickCount();
		Mat dst = apply_gauss2D_filter(src, w);
		t = ((double)getTickCount() - t) / getTickFrequency();
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("Original", src);
		imshow("Gauss 1x2D filtered", dst);
	}
}

Mat apply_gauss2x1D_filter(Mat src, float gamma)
{
	Mat dst, aux;
	src.copyTo(dst);
	src.copyTo(aux);

	int width = dst.cols;
	int height = dst.rows;

	//generate gaussian filter
	int w = gamma * 6;
	if (w % 2 == 0) w += 1;
	float G[MAX_SIZE] = { 0 };

	float S = 0.0f;

	for (int i = 0; i < w; i++)
	{
		G[i] = (1.0f / (sqrt(2.0f * PI) * gamma)) * exp(-(((i - w / 2) * (i - w / 2)) / (2 * gamma * gamma)));
		S += G[i];
	}

	int k = (w - 1) / 2;

	for (int i = k; i < height - k; i++)
	{
		for (int j = 0; j < width; j++)
		{
			float dst_pix = 0;

			for (int u = 0; u < w; u++)
			{
				dst_pix += G[u] * src.at<uchar>(i + u - k, j);
			}

			dst_pix /= S;

			//constrain
			if (dst_pix > 255.0f) dst_pix = 255.0f;
			else if (dst_pix < 0.0f) dst_pix = 0.0f;

			aux.at<uchar>(i, j) = (int)dst_pix;
		}
	}

	for (int i = 0; i < height; i++)
	{
		for (int j = k; j < width - k; j++)
		{
			float dst_pix = 0;

			for (int u = 0; u < w; u++)
			{
				dst_pix += G[u] * aux.at<uchar>(i, j + u - k);
			}

			dst_pix /= S;

			//constrain
			if (dst_pix > 255.0f) dst_pix = 255.0f;
			else if (dst_pix < 0.0f) dst_pix = 0.0f;

			dst.at<uchar>(i, j) = (int)dst_pix;
		}
	}

	return dst;
}

void gauss1Dx2_filter()
{
	Mat src;
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);

		printf("gamma?\n");

		float w;
		scanf("%f", &w);

		double t = (double)getTickCount();
		Mat dst = apply_gauss2x1D_filter(src, w);
		t = ((double)getTickCount() - t) / getTickFrequency();
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("Original", src);
		imshow("Gauss 2x1D filtered", dst);
	}
}

Mat apply_filter(int filter[][MAX_SIZE], int filter_size, Mat src)
{
	Mat dst;
	src.convertTo(dst, CV_32FC1);

	int width = dst.cols;
	int height = dst.rows;

	int k = (filter_size - 1) / 2;

	for (int i = k; i < height - k; i++)
	{
		for (int j = k; j < width - k; j++)
		{
			float dst_pix = 0;

			for (int u = 0; u < filter_size; u++)
			{
				for (int v = 0; v < filter_size; v++)
				{
					dst_pix += filter[u][v] * src.at<uchar>(i + u - k, j + v - k);
				}
			}

			dst.at<float>(i, j) = dst_pix;
		}
	}

	return dst;
}

int get_zone(float deg)
{
	//zone 0
	if ((deg >= -CV_PI / 8.0f && deg <= CV_PI / 8.0f) || (deg >= 7.0 * CV_PI / 2.0f) || (deg <= -7.0f * CV_PI / 2.0f)) return 0;
	//zone 1
	else if ((deg >= CV_PI / 8.0f && deg <= 3.0 * CV_PI / 8.0f) || (deg >= -7.0 * CV_PI / 8.0f && deg <= -5.0 * CV_PI / 8.0f)) return 1;
	//zone 2
	else if ((deg >= 3.0f * CV_PI / 8.0f && deg <= 5.0 * CV_PI / 8.0f) || (deg >= -5.0 * CV_PI / 8.0f && deg <= -3.0 * CV_PI / 8.0f)) return 2;
	//zone 3
	else if ((deg >= 5.0f * CV_PI / 8.0f && deg <= 7.0f * CV_PI / 8.0f) || (deg >= -3.0 * CV_PI / 8.0f && deg <= CV_PI / 8.0f)) return 3;

	return -1;
}

void histogram_b(Mat src, int* h)
{
	int height = src.rows;
	int width = src.cols;
	
	for (int i = 1; i < height - 1; i++)
	{
		for (int j = 1; j < width - 1; j++)
		{
			uchar current = src.at<uchar>(i, j);
			h[current]++;
		}
	}
}

void edge_detection()
{
	Mat src;
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);

		int Sx[3][MAX_SIZE] = {
			{-1, 0, 1},
			{-2, 0, 2},
			{-1, 0, 1}
		};

		int Sy[3][MAX_SIZE] = {
			{1, 2, 1},
			{0, 0, 0},
			{-1, -2, -1}
		};

		Mat blurred = apply_gauss2x1D_filter(src, 0.6);

		int width = src.cols;
		int height = src.rows;

		Mat Gx = apply_filter(Sx, 3, blurred);
		Mat Gy = apply_filter(Sy, 3, blurred);

		Mat G(height, width, CV_32FC1, 0.0f);
		Mat f(height, width, CV_32FC1, 0.0f);
		Mat Gs(height, width, CV_32FC1, 0.0f);

		Mat G_display(height, width, CV_8UC1);

		Mat Gs_uchar(height, width, CV_8UC1);
		Mat th(height, width, CV_8UC1);

		Mat thresholded(height, width, CV_8UC1);
		
		float div = 4.0f * sqrt(2.0);

		for (int i = 1; i < height - 1; i++)
		{
			for (int j = 1; j < width - 1; j++)
			{
				float gx = Gx.at<float>(i, j);
				float gy = Gy.at<float>(i, j);

				f.at<float>(i, j) = atan2(gy, gx);

				float val = sqrt(gx * gx + gy * gy) / div;

				G.at<float>(i, j) = val;
			}
		}

		for (int i = 1; i < height - 1; i++)
		{
			for (int j = 1; j < width - 1; j++)
			{
				float degree = f.at<float>(i, j);

				int zone = get_zone(degree);
				float N1 = -1.0, N2 = -1.0;

				if (zone == 0)
				{
					N1 = G.at<float>(i, j - 1);
					N2 = G.at<float>(i, j + 1);
				}

				else if (zone == 1)
				{
					N1 = G.at<float>(i - 1, j + 1);
					N2 = G.at<float>(i + 1, j - 1);
				}

				else if (zone == 2)
				{
					N1 = G.at<float>(i - 1, j);
					N2 = G.at<float>(i + 1, j);
				}

				else if (zone == 3)
				{
					N1 = G.at<float>(i - 1, j - 1);
					N2 = G.at<float>(i + 1, j + 1);
				}

				float val = G.at<float>(i, j);

				if (val >= N1 && val >= N2)
				{
					Gs.at<float>(i, j) = val;
				}

				else
				{
					Gs.at<float>(i, j) = 0.0f;
				}
			}
		}
		
		Gs.convertTo(Gs_uchar, CV_8UC1);
		
		//get histogram
		int h[256] = { 0 };
		
		histogram_b(Gs_uchar, h);

		float p = 0.1, k = 0.4;

		float NoEdgePixels = p * (((height - 2) * (width - 2)) - h[0]);

		printf("noEdgePixels = %.3f", NoEdgePixels);
		
		//threshold the img
		int sum = 0;
		int Thigh = 0;

		for (int i = 255; i > 1; i--)
		{
			sum += h[i];

			if (sum >= NoEdgePixels)
			{
				Thigh = i;
				break;
			}
		}

		int Tlow = k * Thigh;

		printf("t_low = %d, t_high = %d", Tlow, Thigh);

		for (int i = 1; i < height - 1; i++)
		{
			for (int j = 1; j < width - 1; j++)
			{
				uchar pix = Gs_uchar.at<uchar>(i, j);

				if (pix >= Thigh)
				{
					th.at<uchar>(i, j) = 255;
				}

				else if (pix <= Tlow)
				{
					th.at<uchar>(i, j) = 0;
				}

				else
				{
					th.at<uchar>(i, j) = 127;
				}
			}
		}

		th.copyTo(thresholded);

		//BFS
		//suppress weak edges
		int label = 0;
		
		int dy[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
		int dx[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };

		for (int i = 1; i < height - 1; i++)
		{
			for (int j = 1; j < width - 1; j++)
			{
				if (th.at<uchar>(i, j) == 255)
				{
					label++;
					std::queue<Point> Q;

					Q.push(Point(j, i));

					while (!Q.empty())
					{
						Point q = Q.front();
						Q.pop();

						for (int k = 0; k < 8; k++)
						{
							int y = q.y + dx[k];
							int x = q.x + dy[k];

							if (x > 1 && x < width - 1 && y > 1 && y < height - 1 && th.at<uchar>(y, x) == 127)
							{
								th.at<uchar>(y, x) = 255;
								Point neighbor(x, y);
								Q.push(neighbor);
							}
						}
					}
				}
			}
		}

		for (int i = 1; i < height - 1; i++)
		{
			for (int j = 1; j < width - 1; j++)
			{
				if (th.at<uchar>(i, j) == 127)
				{
					th.at<uchar>(i, j) = 0;
				}
			}
		}

		imshow("Original", src);
		imshow("Non-maxima suppression", Gs_uchar);

		imshow("Thresholded", thresholded);
		imshow("Hysteresis", th);

		G.convertTo(G_display, CV_8UC1);
		imshow("Gradient magnitude", G_display);
	}
}

void gaussian_test()
{
	//generate gaussian filter
	int gamma = 2;
	int w = gamma * 6;
	if (w % 2 == 0) w += 1;
	float G[MAX_SIZE][MAX_SIZE] = { 0 };

	printf("w=%d\n", w);

	float S = 0.0f;

	for (int i = 0; i < w; i++)
	{
		for (int j = 0; j < w; j++)
		{
			G[i][j] = (1.0f / (2.0f * PI * gamma * gamma)) * exp(-(((i - w / 2) * (i - w / 2) + (j - w / 2) * (j - w / 2)) / (2 * gamma * gamma)));
			S += G[i][j];
		}
	}

	printf("Gaussian filter:\n");

	for (int i = 0; i < w; i++)
	{
		for (int j = 0; j < w; j++)
		{
			printf("%.5f ", G[i][j]);
		}

		printf("\n");
	}

	float mid = G[7][7];

	float result = G[7][3] / mid;

	printf("%.4f\n", log(result));

	getchar();
	getchar();
}

void create_img()
{
	int height = 200;
	int width = 100;
	Mat dst(height, width, CV_8UC1, 127);
	// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
	// Varianta ineficienta (lenta)
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (i == 50)
			{
				dst.at<uchar>(i, j) = 255;
			}
		}
	}

	imshow("constructed image", dst);
	waitKey();
}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - Change gray levels by 50\n");
		printf(" 11 - Construct White Red Green Yellow image\n");
		printf(" 12 - Inverse Matrix\n");
		printf(" 13 - RGB24-split channels\n");
		printf(" 14 - Color2Grayscale\n");
		printf(" 15 - Grayscale2BlacknWhite\n");
		printf(" 16 - RGB2HSV\n");
		printf(" 17 - Gray level histogram\n");
		printf(" 18 - Multilevel thresholding\n");
		printf(" 19 - Floyd-Steinberg dithering\n");
		printf(" 20 - Geometrical features\n");
		printf(" 21 - BFS labeling\n");
		printf(" 22 - Two pass labeling\n");
		printf(" 23 - Border tracing\n");
		printf(" 24 - Reconstruct from AC\n");
		printf(" 25 - Dilation\n");
		printf(" 26 - Erosion\n");
		printf(" 27 - Opened\n");
		printf(" 28 - Closed\n");
		printf(" 29 - Boundary extraction\n");
		printf(" 30 - Region filling\n");
		printf(" 31 - Stats\n");
		printf(" 32 - Global tresholding\n");
		printf(" 33 - Brightness change\n");
		printf(" 34 - Contrast change\n");
		printf(" 35 - Gamma correction\n");
		printf(" 36 - Histogram equalization\n");
		printf(" 37 - Convolution filters\n");
		printf(" 38 - Frequency domain filters\n");
		printf(" 39 - Median filter\n");
		printf(" 40 - Gauss 1x2D filter\n");
		printf(" 41 - Gauss 2x1D filter\n");
		printf(" 42 - Edge Detection\n");
		printf(" 43 - Gauss_test\n");
		printf(" 44 - line in the middle");
		printf(" 999 - Zoom (Nearest Neighbor)\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 10:
				changeGrayLevels();
				break;
			case 11:
				four_square_color_image();
				break;
			case 12:
				inverse_matrix();
				break;
			case 13:
				rgb24_splitchannels();
				break;
			case 14:
				color_2_grayscale();
				break;
			case 15:
				grayscale_2_blacknwhite();
				break;
			case 16:
				rgb_2_hsv();
				break;
			case 17:
				histogram_show();
				break;
			case 18:
				multilevel_thresholding(5, 0.0003f);
				break;
			case 19:
				floyd_steinberg_dithering(5, 0.0003f);
				break;
			case 20:
				l4_geometrical_features();
				break;
			case 21:
				BFSTraversal();
				break;
			case 22:
				twoPass();
				break;
			case 23:
				border_tracing();
				break;
			case 24:
				reconstruct_AC();
				break;
			case 25:
				dilate();
				break;
			case 26:
				erode();
				break;
			case 27:
				opening();
				break;
			case 28:
				closing();
				break;
			case 29:
				boundary_extraction();
				break;
			case 30:
				region_filling();
				break;
			case 31:
				stats();
				break;
			case 32:
				global_tresholding();
				break;
			case 33:
				changeGrayLevels();
				break;
			case 34:
				contrast_change();
				break;
			case 35:
				gamma_correction();
				break;
			case 36:
				histogram_eq();
				break;
			case 37:
				convolution_filters();
				break;
			case 38:
				freq_filters();
				break;
			case 39:
				median_filter();
				break;
			case 40:
				gauss2D_filter();
				break;
			case 41:
				gauss1Dx2_filter();
				break;
			case 42:
				edge_detection();
				break;
			case 43:
				gaussian_test();
				break;
			case 44:
				create_img();
				break;
			case 999:
				zoom();
				break;
			default:
				break;
		}
	}
	while (op!=0);
	return 0;
}