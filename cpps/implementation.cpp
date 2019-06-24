#include <string>
#include <iostream>
#include <vector>
#include <io.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

//孔洞填充算法
void fillHole(const cv::Mat &srcBw, cv::Mat &dstBw)
{
	cv::Size m_Size = srcBw.size();
	cv::Mat Temp = cv::Mat::zeros(m_Size.height + 2, m_Size.width + 2, srcBw.type());

	//Range(int start, int end), 范围包含start, 但不包含end
	srcBw.copyTo(Temp(cv::Range(1, m_Size.height + 1), cv::Range(1, m_Size.width + 1)));

	cv::floodFill(Temp, cv::Point(0, 0), cv::Scalar(255));//缺陷: 边缘封闭图形无法填充
	cv::Mat cutImg;//裁剪延展的图像
	Temp(cv::Range(1, m_Size.height + 1), cv::Range(1, m_Size.width + 1)).copyTo(cutImg);

	dstBw = srcBw | (~cutImg);//按位或(按位取反)
}

// 根据输入轮廓, 找二值图中元件的标准外接矩形,
// 并按矩形面积筛选出大矩形, 最后在图上绘制
void binBoundRectPick(const cv::Mat &src, cv::Mat& dst, size_t areaThre,
	std::vector<std::vector<cv::Point>> boundContours, 
	std::vector<cv::Mat> &boundMat, 
	std::vector<cv::Rect> &boundRect)
{
	Mat cellMat;
	std::vector<cv::Rect>::iterator myIter = boundRect.begin();
	dst = src.clone();
	if (src.channels() != 3)
	{
		cv::cvtColor(src, dst, CV_GRAY2BGR);
	}
	for (int i = 0; i<boundContours.size(); i++)
	{
		boundRect[i] = cv::boundingRect(cv::Mat(boundContours[i])); //计算每个轮廓外接矩形
	}

	while (myIter != boundRect.end())
	{
		size_t rectNum = myIter - boundRect.begin();
		size_t bArea = boundRect[rectNum].area();
		float w_h_ratio = float(boundRect[rectNum].width) / float(boundRect[rectNum].height);
		if (bArea < areaThre || (w_h_ratio > 0.8 && w_h_ratio < 1.25) || w_h_ratio < 0.2 || w_h_ratio > 5.0)
		{
			myIter = boundRect.erase(myIter);
		}
		else
		{
			myIter++;
		}
	}
	for (int i = 0; i < boundRect.size(); i++)
	{
		cv::rectangle(dst, boundRect[i], cv::Scalar(0, 255, 255), 2);
		/*
		char boundRectText[256];
		sprintf_s(boundRectText, 256, "Rect%u:Rows(%u)Cols(%u)", i, boundRect[i].height, boundRect[i].width);
		cv::putText(dst, boundRectText, boundRect[i].br() + cv::Point(-10, 10), 1, 0.9, cv::Scalar(255, 255, 255));
		*/
		cellMat = src(boundRect[i]);
		boundMat.push_back(cellMat);
	}
}

// 找特定面积以上的轮廓, 并画出
std::vector<std::vector<cv::Point>> myContour(cv::Mat src, double areaThreshold)
{
	std::vector<std::vector<cv::Point>> myContours;
	cv::findContours(src, myContours, CV_RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	std::vector<std::vector<cv::Point>>::iterator myIt = myContours.begin();

	while (myIt != myContours.end())
	{
		double myArea = cv::contourArea(myContours[myIt - myContours.begin()]);
		if (myArea < areaThreshold)
		{
			myIt = myContours.erase(myIt);
		}
		else
		{
			++myIt;
		}
	}
	//cv::drawContours(src, myContours, -1, cv::Scalar(255, 255, 255), 1);
	return myContours;
}

void getFiles(string path, vector<string>& files)
{
	long long hFile = 0;
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("/*.jpg").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append("/").append(fileinfo.name), files);
			}
			else
			{
				files.push_back(p.assign(path).append("/").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);

		_findclose(hFile);
	}
}

void get_0(Mat& trainingImages, vector<int>& trainingLabels)
{
	char * filePath = "C:/testimg/UNITS/0-电阻";
	vector<string> files;
	getFiles(filePath, files);
	int number = files.size();
	for (int i = 0; i < number; i++)
	{
		Mat SrcImage = imread(files[i].c_str(), 0);
		resize(SrcImage, SrcImage, Size(130, 320), (0, 0), (0, 0), INTER_LINEAR);
		SrcImage = SrcImage.reshape(1, 1);
		trainingImages.push_back(SrcImage);
		trainingLabels.push_back(0);
	}
}

void get_1(Mat& trainingImages, vector<int>& trainingLabels)
{
	char * filePath = "C:/testimg/UNITS/1-焊锡";
	vector<string> files;
	getFiles(filePath, files);
	int number = files.size();
	for (int i = 0; i < number; i++)
	{
		Mat SrcImage = imread(files[i].c_str(), 0);
		resize(SrcImage, SrcImage, Size(130, 320), (0, 0), (0, 0), INTER_LINEAR);
		SrcImage = SrcImage.reshape(1, 1);
		trainingImages.push_back(SrcImage);
		trainingLabels.push_back(1);
	}
}

void get_2(Mat& trainingImages, vector<int>& trainingLabels)
{
	char * filePath = "C:/testimg/UNITS/2-电容";
	vector<string> files;
	getFiles(filePath, files);
	int number = files.size();
	for (int i = 0; i < number; i++)
	{
		Mat SrcImage = imread(files[i].c_str(), 0);
		resize(SrcImage, SrcImage, Size(130, 320), (0, 0), (0, 0), INTER_LINEAR);
		SrcImage = SrcImage.reshape(1, 1);
		trainingImages.push_back(SrcImage);
		trainingLabels.push_back(2);
	}
}

int SVM_TRAIN()
{
	//将所有图片大小统一转化
	const int imageRows = 320;
	const int imageCols = 130;

	//获取训练数据
	Mat classes;
	Mat trainingData;
	Mat trainingImages;
	vector<int> trainingLabels;
	get_0(trainingImages, trainingLabels);
	get_1(trainingImages, trainingLabels);
	get_2(trainingImages, trainingLabels);

	Mat(trainingImages).copyTo(trainingData);
	trainingData.convertTo(trainingData, CV_32FC1);
	Mat(trainingLabels).copyTo(classes);

	//配置SVM训练器参数
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 10000, 0.0001));

	cout << "开始训练！！！" << endl;
	//训练
	Ptr<TrainData> tData = TrainData::create(trainingData, cv::ml::ROW_SAMPLE, classes);
	svm->train(tData);
	//svm->trainAuto(tData);
	//保存模型
	svm->save("svm_UNITS.xml");
	cout << "训练好了！！！" << endl;

	return 0;
}

auto SVM_PREDICT(vector<Mat> &pics, vector<Rect> &rects)
{
	vector<string> labels;

	//将所有图片大小统一转化
	const int imageRows = 320;
	const int imageCols = 130;

	////==========================预测部分==============================////

	//使用训练好的model预测测试图像
	Ptr<SVM> svm_predict = SVM::load("svm_UNITS.xml");
	Mat test;
	for (int i = 0; i < pics.size(); ++i)
	{
		test = pics[i];
		if (test.channels() != 1)
		{
			cvtColor(test, test, CV_BGR2GRAY);
		}
		if (test.rows < test.cols)
		{
			transpose(test, test);
		}
		resize(test, test, Size(imageCols, imageRows), (0, 0), (0, 0), INTER_LINEAR);
		//threshold(test, test, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

		//将测试图像转化为行向量
		Mat_<float> testMat(1, imageRows*imageCols);
		for (int j = 0; j < imageRows*imageCols; j++)
		{
			testMat.at<float>(0, j) = (float)test.at<uchar>(j / imageCols, j % imageCols);
		}

		float digit = svm_predict->predict(testMat);

		if (digit == 0)
		{
			labels.push_back("R");
		}

		if (digit == 1)
		{
			labels.push_back("S");
		}

		if (digit == 2)
		{
			labels.push_back("C");
		}
	}
	return labels;
}


void main()
{
	SVM_TRAIN();

	Mat grayImg = imread("inputGray2.bmp", IMREAD_GRAYSCALE);
	Mat colorImg = imread("inputColor2.bmp", IMREAD_COLOR);
	Mat colorImgLabel;
	adaptiveThreshold(grayImg, grayImg, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 0);
	fillHole(grayImg, grayImg);
	medianBlur(grayImg, grayImg, 7);
	dilate(grayImg, grayImg, getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

	vector<vector<Point>> contours = myContour(grayImg, 400);
	//drawContours(colorImg, contours, -1, cv::Scalar(0, 255, 255), 2);
	vector<Rect> unitRects(contours.size());
	vector<Mat> unitPics;
	binBoundRectPick(colorImg, colorImgLabel, 700, contours, unitPics, unitRects);
	vector<string> unitLabels;
	unitLabels = SVM_PREDICT(unitPics, unitRects);
	for (int i = 0; i < unitLabels.size(); ++i)
	{
		char numText[16];
		sprintf_s(numText, 16, "(%u)", i);
		string labelText = numText + unitLabels[i];
		cv::putText(colorImgLabel, labelText, unitRects[i].br() + cv::Point(-70, 30), 4, 1, cv::Scalar(0, 255, 255));
	}
	//imshow("识别结果", colorImgLabel);
	imwrite("识别结果.bmp", colorImgLabel);
	waitKey(0);
}
