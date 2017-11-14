package com.example.g.cardet;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.util.Log;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;


/**
 * Created by nohjunho on 2017-11-06.
 */

public class AlarmService_Service extends BroadcastReceiver {
    Context context;
    //위험, 일반 데이터 담을 버퍼
    ArrayList<double[]> buffer_risky = new ArrayList<>();
    ArrayList<double[]> buffer_normal = new ArrayList<>();
    public static final int labelIndex = 7;
    public DataNormalization normalizer = new NormalizerStandardize();
    public static final int numClasses = 2;
    public static final int batchSizeTraining = 500;
    public File locationToSave_dp; //DP 모델 저장 위치(핸드폰 기준)
    public static final String savedDpModel = "savedDpModel55.zip"; //DP 모델 저장 파일 이름
    public static final String savedCluModel = "savedCluModel55.ser"; // 클러스터링 모델 저장 파일 이름
    public static final String tempDpUpdateData = "dp_update_certain55.csv"; //dp 재학습할때 담는 변수
    public static final String RF_TrainData = "rf_training55.csv"; //핸드폰에 저장할 초기 훈련데이터 파일 이름
    public static final String DP_TrainData = "dp_training55.csv"; //핸드폰에 저장할 초기 훈련데이터 파일 이름
    public static final String CLU_TrainData = "clu_training55.arff"; //핸드폰에 저장할 초기 훈련데이터 파일 이름
    public static final String buffer_riskySave = "buffer_riskySave55.csv";
    public static final String buffer_normalSave = "buffer_normalSave55.csv";

    public int c = 0; //검지 횟수
    public int count = 0; //위험 또는 일반으로 검지된 데이터 수
    public int updated_count = 0; //갱신된 데이터 수
    public static final int ATTR_NUM = 7; //속성 수

    public RandomForest RaF;
    public MultiLayerNetwork model;
    public ClusteringDemo cd;
    public DataSet trainingData;

    public AlarmService_Service() {
        super();
    }

    @Override
    public void onReceive(Context context, Intent intent) {
        Log.i("onReceive", "onReceive실행");
        this.context = context;
        modelUpdating();

    }


    public void modelUpdating() {
        Log.i("modelUpdating", "호출");
        try {


            BufferedReader brrisky = new BufferedReader(new FileReader(new File(context.getApplicationContext().getFilesDir().getAbsolutePath() + "/" + buffer_riskySave)));
            BufferedReader brnormal = new BufferedReader(new FileReader(new File(context.getApplicationContext().getFilesDir().getAbsolutePath() + "/" + buffer_normalSave)));
            String s = "";
            while ((s = brrisky.readLine()) != null) {
                String[] temp = s.split(",");
                double[] d = new double[temp.length];
                for (int i = 0; i < temp.length; i++) {
                    d[i] = Double.parseDouble(temp[i]);
                }
                buffer_risky.add(d);
            }
            s = "";
            while ((s = brnormal.readLine()) != null) {
                String[] temp = s.split(",");
                double[] d = new double[temp.length];
                for (int i = 0; i < temp.length; i++) {
                    d[i] = Double.parseDouble(temp[i]);
                }
                buffer_normal.add(d);
            }
            brrisky.close();
            brnormal.close();

            Log.i("위험 버퍼 사이즈", buffer_risky.size() + " ");
            Log.i("일반 버퍼 사이즈", buffer_normal.size() + " ");

            double[][] updateDatas = new double[buffer_risky.size() + buffer_normal.size()][ATTR_NUM + 1];
            for (int i = 0; i < buffer_risky.size(); i++) {
                updateDatas[i] = buffer_risky.get(i);
            }
            for (int i = 0; i < buffer_normal.size(); i++) {
                updateDatas[i + buffer_risky.size()] = buffer_normal.get(i);
            }

            Log.i("modelUpdating22", Arrays.deepToString(updateDatas));
            updatingWithRaF(updateDatas);

            updatingWithDp(updateDatas);

            updatingWithClu(updateDatas);

            //갱신 데이터 버퍼 클리어
            updated_count = buffer_risky.size() + buffer_normal.size();
//                Log.i("buffer_rfbefore", buffer_rf.size() + ".");
            buffer_risky.clear();
            buffer_normal.clear();

//                Log.i("buffer_rfafter", buffer_rf.size() + ".");

//                handler.post(updateResults); //처리 완료후 Handler의 Post를 사용해서 이벤트 던짐
        } catch (Exception e) {
            Log.e("errorupdate", e.toString());
        }
    }

    public void updatingWithRaF(double[][] updateDatas) {
        //-----------------------------------랜포 재학습-------------------------------------------------
        Log.i("updatingWithRaF", "호출");
        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(new File(context.getApplicationContext().getFilesDir().getAbsolutePath() + "/" + RF_TrainData), true)); // 기존 파일에 붙여씀
            PrintWriter pw = new PrintWriter(bw);
            for (int i = 0; i < updateDatas.length; i++) {
                if (((int) updateDatas[i][7]) == 0) { //일반일 경우
                    pw.println((int) updateDatas[i][0] + "," + (int) updateDatas[i][1] + "," + (int) updateDatas[i][2] + "," + (int) updateDatas[i][3] + "," + (int) updateDatas[i][4] + "," + (int) updateDatas[i][5] + "," + (int) updateDatas[i][6] + "," + 2);
                    pw.flush();
                } else { //위험일 경우
                    pw.println((int) updateDatas[i][0] + "," + (int) updateDatas[i][1] + "," + (int) updateDatas[i][2] + "," + (int) updateDatas[i][3] + "," + (int) updateDatas[i][4] + "," + (int) updateDatas[i][5] + "," + (int) updateDatas[i][6] + "," + (int) updateDatas[i][7]);
                    pw.flush();
                }

//                            System.out.println("데이터" + (i + 1) + "번째 쓰기 완료");
            }
            BufferedReader br = new BufferedReader(new FileReader(new File(context.getApplicationContext().getFilesDir().getAbsolutePath() + "/" + RF_TrainData)));
            String s = "";
            while ((s = br.readLine()) != null) {
                Log.i("stringmessageRF", s);
            }
            RaF = initRF();

            pw.close();
            bw.close();
            br.close();
        } catch (Exception e) {
            Log.i("랜포 재학습 error", "오류");
            e.printStackTrace();
        }
    }

    public void updatingWithDp(double[][] updateDatas) {
        //----------------------------------신경망 재학습--------------------------------------------------//모델을 업데이트해야됨
        Log.i("updatingWithDp", "호출");
        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(new File(context.getApplicationContext().getFilesDir().getAbsolutePath() + "/" + tempDpUpdateData))); // 업데이트를 위한 데이터 파일 새로 생성
            PrintWriter pw = new PrintWriter(bw);
            for (int i = 0; i < updateDatas.length; i++) {
                pw.println(updateDatas[i][0] + "," + updateDatas[i][1] + "," + updateDatas[i][2] + "," + updateDatas[i][3] + "," + updateDatas[i][4] + "," + updateDatas[i][5] + "," + updateDatas[i][6] + "," + (int) updateDatas[i][7]);
                pw.flush();
            }
            pw.close();
            bw.close();

            //normalizer 생성
            trainingData = readCSVDataset(DP_TrainData, batchSizeTraining, labelIndex, numClasses);
            normalizer.fit(trainingData);

            //업데이트 데이터 -> DataSet 전환
            DataSet update_data = readCSVDataset(tempDpUpdateData, batchSizeTraining, labelIndex, numClasses);
            Log.i("update_date", update_data.getFeatureMatrix() + ";");

            //업데이트 데이터 정규화
            normalizer.transform(update_data);

            //모델 불러오기
            locationToSave_dp = new File(context.getApplicationContext().getFilesDir().getAbsolutePath() + savedDpModel);
            model = ModelSerializer.restoreMultiLayerNetwork(locationToSave_dp);
            //모델 갱신
            model.fit(update_data);

            Log.i("modelfit", "성공");
            File locationToSave = new File(context.getApplicationContext().getFilesDir().getAbsolutePath() + savedDpModel);      //Where to save the network. Note: the file is in .zip format - can be opened externally
            boolean saveUpdater = true;                                             //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
            ModelSerializer.writeModel(model, locationToSave, saveUpdater);
            Log.i("Network model", "Updated Network model");

        } catch (Exception e) {
            e.printStackTrace();
            Log.i("modelfit", "오류");
        }
    }

    public void updatingWithClu(double[][] updateDatas) {
        //----------------------------------클러스터링 재학습--------------------------------------------------
        Log.i("updatingWithClu", "호출");
        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(new File(context.getApplicationContext().getFilesDir().getAbsolutePath() + "/" + CLU_TrainData), true)); // 기존 파일에 붙여씀
            PrintWriter pw = new PrintWriter(bw);
            for (int i = 0; i < updateDatas.length; i++) {
                pw.println(updateDatas[i][0] + "," + updateDatas[i][1] + "," + updateDatas[i][2] + "," + updateDatas[i][3] + "," + updateDatas[i][4] + "," + updateDatas[i][5] + "," + updateDatas[i][6]);
                pw.flush();
//                System.out.println("데이터" + (i + 1) + "번째 쓰기 완료");
            }

//            BufferedReader br = new BufferedReader(new FileReader(new File(context.getApplicationContext().getFilesDir().getAbsolutePath() + "/" + CLU_TrainData)));
//            String s = "";
//            while ((s = br.readLine()) != null) {
//                Log.i("stringmessageCLU", s);
//            }

            cd = new ClusteringDemo(context.getApplicationContext(), new FileInputStream(new File(context.getApplicationContext().getFilesDir().getAbsolutePath() + "/" + CLU_TrainData)));
            Log.i("clusterer 업데이트 빌드", "ok");

            //클러스터링 모델 직렬화
            FileOutputStream fosSerial = new FileOutputStream(context.getApplicationContext().getFilesDir().getAbsolutePath() + savedCluModel);
            BufferedOutputStream bosSerial = new BufferedOutputStream(fosSerial);
            ObjectOutputStream oos = new ObjectOutputStream(bosSerial);
            oos.writeObject(cd);

            pw.close();
            bw.close();
//            br.close();
            oos.close();
            Log.i("cluster 업데이트 모델 저장", "ok");

        } catch (Exception e) {
            Log.i("클러스터링 재학습 error", "오류");
            e.printStackTrace();
        }

        //asset에 park.arff 을 getFilesDir().getAbsolutePath() 안에 넣고 추가로 버퍼에 있는 데이터를 넣는다
        //다시 읽어서 클러스터링 빌드를 한다.
    }

    private RandomForest initRF() throws Exception {
        Log.i("initRF", "호출");
        int numTrees = 50;// 트리 수
        String dirPath = context.getFilesDir().getAbsolutePath();
        File savefile = new File(dirPath + "/" + RF_TrainData);
        //일치하는 파일이 없으면 생성
        if (!savefile.exists()) {
            try {
                String sCurrentLine;
                InputStream is = context.getAssets().open("rf_training_certain.csv"); //asset에 있는걸 파일 내부로
                BufferedReader br = new BufferedReader(new InputStreamReader(is));
                FileOutputStream fos = new FileOutputStream(savefile);
                PrintWriter pw = new PrintWriter(fos);
                //---------------------
                while ((sCurrentLine = br.readLine()) != null) {
                    if (sCurrentLine != null) {
                        pw.println(sCurrentLine);
                        pw.flush();
                    }
                }
                pw.close();
                fos.close();
                br.close();
                is.close();
                Log.i("rf_training 파일", "내부 저장소에 생성");
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else {
            Log.i("rf_training 파일", "already 존재");
        }

        // 훈련데이터 읽기
        DescribeTrees DT = new DescribeTrees(RF_TrainData);
//        InputStream is = getAssets().open("rf_training_certain.csv");
        ArrayList<int[]> Input = DT.CreateInput(savefile);
//        for (int i = 0; i < Input.size(); i++) {
//            System.out.println(Arrays.toString(Input.get(i)));
//        }
        int categ = 2;

        RandomForest RaF = new RandomForest(numTrees, Input);

        // C : 범주 수 , M : 범주 속성
        RaF.C = categ;
        RaF.M = Input.get(0).length - 1;
        RaF.Ms = (int) Math.round(Math.log(RaF.M) / Math.log(2) + 1);
        RaF.createRF();
        return RaF;
    }

    private DataSet readCSVDataset(String csvFileClasspath, int batchSize, int labelIndex, int numClasses) throws IOException, InterruptedException {
        Log.i("readCSVDataset", "호출");
        String dirPath = context.getApplicationContext().getFilesDir().getAbsolutePath();
        File file = new File(dirPath);//폴더 생성
        File savefile = new File(dirPath + "/" + csvFileClasspath);
        if (!file.exists()) {
            file.mkdirs();
        }
        if (savefile.exists()) {
            Log.i("readCSVDataset(exists)", savefile.getName());
            RecordReader rr = new CSVRecordReader();
            rr.initialize(new FileSplit(savefile));
            DataSetIterator iterator = new RecordReaderDataSetIterator(rr, batchSize, labelIndex, numClasses);
            return iterator.next();
        } else {
            Log.i("readCSVDatasetNotexists", savefile.getName());
            Log.i("dp_training 파일", "내부 저장소에 생성");
            // 일치하는 파일이 없으면 생성
            try {
                String sCurrentLine;
                InputStream is = context.getApplicationContext().getAssets().open("dp_training_certain.csv"); //assets 폴더에 있는것 읽어오기
                BufferedReader br = new BufferedReader(new InputStreamReader(is));
                FileOutputStream fos = new FileOutputStream(savefile);
                PrintWriter pw = new PrintWriter(fos);
                //---------------------
                while ((sCurrentLine = br.readLine()) != null) {
                    if (sCurrentLine != null) {
                        pw.println(sCurrentLine);
                        pw.flush();
                    }
                }
                RecordReader rr = new CSVRecordReader();
                rr.initialize(new FileSplit(savefile));
                DataSetIterator iterator = new RecordReaderDataSetIterator(rr, batchSize, labelIndex, numClasses);
                pw.close();
                fos.close();
                br.close();
                is.close();
                return iterator.next();
            } catch (IOException e) {
                return null;
            }
        }
    }

}
