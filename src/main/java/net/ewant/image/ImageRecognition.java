package net.ewant.image;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.KNearest;
import org.opencv.utils.Converters;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.net.URISyntaxException;
import java.util.*;

import static org.opencv.ml.Ml.ROW_SAMPLE;

public class ImageRecognition {
    private static String TRAIN_PATH = ImageRecognition.class.getClassLoader().getResource("trainsimple/").getPath().substring(1);
    private static String NUM_NAME[] = { "1_", "2_", "3_", "4_", "5_", "6_", "7_", "8_", "9_", "10_", "j_", "q_", "k_" };
    private static String FLAG_NAME[] = { "hongtao_", "fangkuai_", "meihua_", "heitao_" };

    static{
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }


    /**
     * 训练模型
     */
    private static void trainPixel() {
        Mat numData = new Mat(), flagData = new Mat();
        List<Integer> numLabels = new ArrayList<>(), flagLabels = new ArrayList<>();
        int trainNum = 40;// 每项样本数
        for (int i = 0; i < trainNum * NUM_NAME.length; i++){
            Mat tmp = new Mat();
            Mat img = Imgcodecs.imread(TRAIN_PATH + NUM_NAME[i/trainNum] + (i%trainNum) + ".jpg", 0);
            Imgproc.resize(img, tmp, new Size(30, 40));
            numData.push_back(tmp.reshape(0, 1));  //序列化后放入特征矩阵
            numLabels.add(i / trainNum + 1);
        }
        numData.convertTo(numData, CvType.CV_32F); //uchar型转换为cv_32f
        //使用KNN算法
        int K = 5;
        KNearest numModel = KNearest.create();
        numModel.setDefaultK(K);
        numModel.setIsClassifier(true);
        numModel.train(numData, ROW_SAMPLE, Converters.vector_int_to_Mat(numLabels));
        numModel.save(TRAIN_PATH + "../num_knn_pixel.yml");

        // train flag
        for (int i = 0; i < trainNum * FLAG_NAME.length; i++){
            Mat tmp = new Mat();
            Mat img = Imgcodecs.imread(TRAIN_PATH + FLAG_NAME[i/trainNum] + (i%trainNum) + ".jpg", 0);
            Imgproc.resize(img, tmp, new Size(30, 30));
            flagData.push_back(tmp.reshape(0, 1));  //序列化后放入特征矩阵
            flagLabels.add(i / trainNum + 1);
        }
        flagData.convertTo(flagData, CvType.CV_32F); //uchar型转换为cv_32f
        //使用KNN算法
        int L = 5;
        KNearest flagModel = KNearest.create();
        flagModel.setDefaultK(L);
        flagModel.setIsClassifier(true);
        flagModel.train(flagData, ROW_SAMPLE, Converters.vector_int_to_Mat(flagLabels));
        flagModel.save(TRAIN_PATH + "../flag_knn_pixel.yml");
    }

    public static void main(String[] args) throws Exception{
        trainPixel();
        /**
         * 1. 读取原始图像转换为OpenCV的Mat数据格式
         */
        Mat srcMat = Imgcodecs.imread("F:/cardtest/cards.png");  //原始图像
        System.out.println(srcMat.height());
        /**
         * 2. 强原始图像转化为灰度图像
         */
        Mat grayMat = new Mat(); //灰度图像
        Imgproc.cvtColor(srcMat, grayMat, Imgproc.COLOR_RGB2GRAY);

        BufferedImage grayImage =  toBufferedImage(grayMat);

        saveJpgImage(grayImage,"F:/cardtest/grayImage.jpg");

        System.out.println("保存灰度图像！");

        Mat binMat = new Mat(); //二值化图像
        Imgproc.threshold(grayMat, binMat, 80d, 255d, Imgproc.THRESH_BINARY);// 对光照和环境要求较高，阈值设置合适值

        //查找轮廓
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(binMat, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);

        System.out.println("轮廓数量："+ contours.size() + "===" + hierarchy.cols());
        System.out.println("hierarchy类型："+ hierarchy);
        List<Rect> cards = new ArrayList<>();
        for(int k=0;k<contours.size();k++) {
            MatOfPoint point = contours.get(k);
            Rect rect = Imgproc.boundingRect(point);
            Point tl = rect.tl();
            Point br = rect.br();
            double width = Math.abs(br.x - tl.x);
            double height = Math.abs(tl.y - br.y);
            //System.out.println("轮廓下标："+k +" { width:"+width+"; height:"+height+" } ");
            if(width < 65 && width > 20 && height < 65 && height > 20){// 20-65区间为数字或花色
                cards.add(rect);
            }
        }
        Collections.sort(cards, new Comparator<Rect>(){
            @Override
            public int compare(Rect o1, Rect o2) {
                double diff = o1.tl().x - o2.tl().x;
                if(diff > 0){
                    return 1;
                }else if(diff < 0){
                    return -1;
                }
                return 0;
            }
        });
        int i = 0;
        for(Rect rect : cards){
            System.out.println("TL x:"+rect.tl().x+",y:"+rect.tl().y + "; BR x:"+rect.br().x+",y:"+rect.br().y);
            saveJpgImage(toBufferedImage(new Mat(binMat, rect)), "F:/cardtest/rect"+(i++)+".jpg");
        }
    }

    /**
     * 将Mat图像格式转化为 BufferedImage
     * @param matrix  mat数据图像
     * @return BufferedImage
     */
    private static BufferedImage toBufferedImage(Mat matrix) {
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if (matrix.channels() > 1) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        int bufferSize = matrix.channels() * matrix.cols() * matrix.rows();
        byte[] buffer = new byte[bufferSize];
        matrix.get(0, 0, buffer); // 获取所有的像素点
        BufferedImage image = new BufferedImage(matrix.cols(), matrix.rows(), type);
        final byte[] targetPixels = ((DataBufferByte)image.getRaster().getDataBuffer()).getData();
        System.arraycopy(buffer, 0, targetPixels, 0, buffer.length);
        return image;
    }


    /**
     * 将BufferedImage内存图像保存为图像文件
     * @param image BufferedImage
     * @param filePath  文件名
     */
    private static void saveJpgImage(BufferedImage image, String filePath) {

        try {
            ImageIO.write(image, "jpg", new File(filePath));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

}
