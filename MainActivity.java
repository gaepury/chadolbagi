package com.example.g.cardet;

import android.Manifest;
import android.app.Activity;
import android.app.AlarmManager;
import android.app.AlertDialog;
import android.app.Dialog;
import android.app.PendingIntent;
import android.app.ProgressDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.drawable.ColorDrawable;
import android.location.Location;
import android.os.Environment;
import android.os.Handler;
import android.os.Message;
import android.os.Vibrator;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.os.Bundle;
import android.telephony.TelephonyManager;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import com.google.android.gms.common.ConnectionResult;
import com.google.android.gms.common.GooglePlayServicesUtil;
import com.google.android.gms.common.api.GoogleApiClient;
import com.google.android.gms.location.LocationListener;
import com.google.android.gms.location.LocationRequest;
import com.google.android.gms.location.LocationServices;

import org.apache.commons.io.IOUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Timer;
import java.util.TimerTask;
import java.util.Vector;

public class MainActivity extends Activity
        implements CameraBridgeViewBase.CvCameraViewListener2,
        ActivityCompat.OnRequestPermissionsResultCallback,
        GoogleApiClient.ConnectionCallbacks,
        GoogleApiClient.OnConnectionFailedListener,
        LocationListener {

    public Logger log = LoggerFactory.getLogger(MainActivity.class);
    public Thread detectThread; //검지용 쓰레드
    public socketManager sm = new socketManager(); //서버 통신

    //신경망 모델링을 하고 난 후에는 모델링 된 객체를 불러오기만 함
    public File locationToSave_dp; //DP 모델 저장 위치(핸드폰 기준)
    public static final String savedDpModel = "savedDpModel57.zip"; //DP 모델 저장 파일 이름
    public static final String savedCluModel = "savedCluModel57.ser"; // 클러스터링 모델 저장 파일 이름
    public static final String tempDpUpdateData = "dp_update_certain57.csv"; //dp 재학습할때 담는 변수
    public static final String tempCluTestData = "clu_test57.csv"; //클러스터링 검지할때 테스트 데이터 담는 변수
    public static final String RF_TrainData = "rf_training57.csv"; //핸드폰에 저장할 초기 훈련데이터 파일 이름
    public static final String DP_TrainData = "dp_training57.csv"; //핸드폰에 저장할 초기 훈련데이터 파일 이름
    public static final String CLU_TrainData = "clu_training57.arff"; //핸드폰에 저장할 초기 훈련데이터 파일 이름
    public static final String preferenceInit = "init57";
    public static final String buffer_riskySave = "buffer_riskySave57.csv";
    public static final String buffer_normalSave = "buffer_normalSave57.csv";
    public static final int RISK_SITUATION = 1; //위험상황
    public static final int NORMAL_SITUATION = 2; //일반상황
    public static final int SUSPICION_SITUATION = 3; //의심상황

    public RandomForest RaF;
    public MultiLayerNetwork model;
    public ClusteringDemo cd;
    public int result_rf;
    public int result_dp;
    public int result_clu;

    public int c = 0; //검지 횟수
    public int count = 0; //위험 또는 일반으로 검지된 데이터 수
    public long decletime;
    public int updated_count = 0; //갱신된 데이터 수
    public static final int ATTR_NUM = 7; //속성 수

    //값 입력 유효성 검사 플래그
    public boolean errorFlag = false;

    //신경망 관련 속성
    public DataNormalization normalizer = new NormalizerStandardize();
    public static final int labelIndex = 7;
    public static final int numClasses = 2;
    public static final int batchSizeTraining = 500;
    //앱 생성후 나갔다가 들어왔을때 초기 세팅값
    public DataSet trainingData;
    private Map<Integer, String> classifiers;

    //위험, 일반 데이터 담을 버퍼
    ArrayList<double[]> buffer_risky = new ArrayList<>();
    ArrayList<double[]> buffer_normal = new ArrayList<>();

    //로딩바
    ProgressDialog progressDialog;
    private Handler handler = new Handler();
    final private int PROGRESS_DIALOG = 0;
    final private int CREATE_DIALOG = 1;
    final private int LOAD_DIALOG = 2;
    boolean complete_flag = false;

    Button detect;
    Button data_update;

    private CameraBridgeViewBase mOpenCvCameraView;

    private GoogleApiClient googleApiClient;
    private Location lct, oLct;

    Timer timer;
    Long stdTime = 0L;
    String fileName;
    FileWriter fw = null;

    double distance[], spd = 0;
    ArrayList<Double> speedList;

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String permissions[], @NonNull int[] grantResults) {
        switch (requestCode) {
            case 110:
                for (int res : grantResults) {
                    if (res != PackageManager.PERMISSION_GRANTED) {
                        Toast.makeText(this, "권한이 확보되지 않아 앱을 실행 할 수 없습니다.", Toast.LENGTH_SHORT).show();
                        finish();
                        break;
                    }
                }
                initProc();
                break;
            default:
                break;
        }
    }

    public void initProc() {
        if (GooglePlayServicesUtil.isGooglePlayServicesAvailable(this) == ConnectionResult.SUCCESS) {
            googleApiClient = new GoogleApiClient.Builder(this)
                    .addApi(LocationServices.API)
                    .addConnectionCallbacks(this)
                    .addOnConnectionFailedListener(this)
                    .build();

            if (!googleApiClient.isConnected() || !googleApiClient.isConnecting()) {
                googleApiClient.connect();
            }
        }

        distance = new double[2];
        distance[0] = 999;

        stdTime = System.currentTimeMillis();
        SimpleDateFormat dayTime = new SimpleDateFormat("yyyyMMdd_hhmmss", Locale.KOREAN);
        String tmpDir = Environment.getExternalStorageDirectory().getAbsolutePath() + "/CarDet";
        fileName = "/" + dayTime.format(new Date(System.currentTimeMillis())) + ".txt";
        try {
            if (!new File(tmpDir).exists())
                new File(tmpDir).mkdirs();

            fw = new FileWriter(tmpDir + fileName, true);
        } catch (Exception e) {
            e.printStackTrace();
        }

        speedList = new ArrayList<>();

        mOpenCvCameraView = (JavaCameraView) findViewById(R.id.camera_view);
        mOpenCvCameraView.setMaxFrameSize(1920, 1080);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        mLoaderCallback = new BaseLoaderCallback(this) {
            @Override
            public void onManagerConnected(int status) {
                switch (status) {
                    case LoaderCallbackInterface.SUCCESS:
                        mOpenCvCameraView.enableView();
                        break;
                    default:
                        super.onManagerConnected(status);
                        break;
                }
            }
        };
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);
        setContentView(R.layout.activity_main);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        Log.i("onCreate", "실행");

        Vector<String> permissions = new Vector<>();

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED)
            permissions.add(android.Manifest.permission.ACCESS_FINE_LOCATION);

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED)
            permissions.add(android.Manifest.permission.ACCESS_COARSE_LOCATION);

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED)
            permissions.add(android.Manifest.permission.CAMERA);

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED)
            permissions.add(android.Manifest.permission.WRITE_EXTERNAL_STORAGE);

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_PHONE_STATE) != PackageManager.PERMISSION_GRANTED)
            permissions.add(android.Manifest.permission.READ_PHONE_STATE);

        if (permissions.size() > 0)
            ActivityCompat.requestPermissions(this, permissions.toArray(new String[permissions.size()]), 110);
        else {
            initProc();
        }
        new Thread() { //모델링 생성 및 로드 담당 쓰레드
            @Override
            public void run() {
                init();
            }
        }.start();
    }

    public void init() {
        Log.i("init 함수", "호출");
        connectServer();
        //앱 생성시인지 아닌지
        SharedPreferences pref = getSharedPreferences("isCreated", Activity.MODE_PRIVATE);
        Log.i("pref", pref.getBoolean(preferenceInit, false) + ".");

        //앱 생성시에만 모델링
        if (pref.getBoolean(preferenceInit, false) == false) {
//            handler2.sendEmptyMessage(CREATE_DIALOG);
            SharedPreferences pref2 = getSharedPreferences("isCreated", Activity.MODE_PRIVATE);
            SharedPreferences.Editor editor = pref2.edit();

            classifiers = readEnumCSV("classifiers3.csv");
            initModeling();

            editor.putBoolean(preferenceInit, true);
            editor.commit();
            Log.i("All Model", "Create All model");

            handler.post(createAllModel);
            registerAlarm();

        } else {//다시 들어왔을때
            locationToSave_dp = new File(getFilesDir().getAbsolutePath() + savedDpModel);
            try {
                RaF = initRF();

                dpSetting();
                model = ModelSerializer.restoreMultiLayerNetwork(locationToSave_dp);
                Log.i("DeepLearning Model", "Load Dp Model");

                FileInputStream fis = new FileInputStream(getFilesDir().getAbsolutePath() + savedCluModel);
                BufferedInputStream bis = new BufferedInputStream(fis);
                ObjectInputStream ois = new ObjectInputStream(bis);
                cd = (ClusteringDemo) ois.readObject();
                ois.close();
                Log.i("Clustring Model", "Load Clustering Model");

                Log.i("All Model", "Load All model");
                handler.post(loadAllModel);
                registerAlarm();
            } catch (Exception e) {
                Log.i("restoredError", e.toString());
                Log.i("restored", "fail");
            }
        }
    }

    public void connectServer() {
        Log.i("server", "connectServer 호출");
        //서버 접속 쓰레드
        new Thread() {
            public void run() {
                try {
                    sm.connectServerSocket(7777);
                    if (sm.getServerSocket() != null && !sm.getServerSocket().isClosed()) {
                        TelephonyManager telManager = (TelephonyManager) getApplicationContext().getSystemService(getApplicationContext().TELEPHONY_SERVICE);
                        String phoneNum = telManager.getLine1Number();
                        Log.i("폰번호", phoneNum);
                        DataOutputStream dos = sm.getServer_dos();
                        dos.writeUTF(phoneNum); //폰번호 전달

                        dos.writeUTF("joincheck"); //정보확인
                        DataInputStream dis = sm.getServer_dis();
                        String joincheck = dis.readUTF();
                        Log.i("joinCheck", joincheck);
                        if (joincheck.equals("No")) {
                            dos.writeUTF("adduser"); //사용자 추가
                        }
                    }
                    while (sm.getServerSocket() != null && !sm.getServerSocket().isClosed()) {
                        DataInputStream dis = sm.getServer_dis();
                        if (dis.available() > 0) {
                            String Msg = dis.readUTF();
                            if (Msg.equals("alarm")) {
                                MainActivity.this.runOnUiThread(new Runnable() {
                                    public void run() {
                                        Vibrator vibe = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);
//                                        Toast toast = Toast.makeText(MainActivity.this, "반경 1km안에 차량에서 위험 상황이 발생했습니다.\n주행ㅇ에 주ㅡ이하세요.", Toast.LENGTH_LONG);
//                                        TextView v = (TextView) toast.getView().findViewById(android.R.id.message);
//                                        v.setTextColor(Color.WHITE);
//                                        v.setTextSize(100);
//                                        toast.getView().setBackgroundColor(Color.WHITE);
//                                        toast.getView().setPadding(50, 100, 50, 10);
//                                        toast.show();
                                        AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);     // 여기서 this는 Activity의 this

                                        // 여기서 부터는 알림창의 속성 설정
                                        builder.setTitle("위험 알림 경고")        // 제목 설정
                                                .setMessage("반경 1km안 차량에서 위험 상황이 발생했습니다.\n주행에 주의하세요!")        // 메세지 설정
                                                .setCancelable(false)        // 뒤로 버튼 클릭시 취소 가능 설정
                                                .setPositiveButton("확인", new DialogInterface.OnClickListener() {
                                                    // 확인 버튼 클릭시 설정
                                                    public void onClick(DialogInterface dialog, int whichButton) {
                                                        dialog.cancel();
                                                    }
                                                });

                                        AlertDialog dialog = builder.create();    // 알림창 객체 생성
                                        dialog.show();    // 알림창 띄우기

                                        vibe.vibrate(1000);
                                    }
                                });
                            }
                        }
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }.start();
    }

    public void initModeling() {
        try {
            RaF = initRF();
            model = initDP();
            cd = initClu();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void dpSetting() throws Exception { //앱을 다시 들어올때 초기 세팅
        Log.i("dpSetting 호출", "ok");
        classifiers = readEnumCSV("classifiers3.csv");
        trainingData = readCSVDataset(DP_TrainData, batchSizeTraining, labelIndex, numClasses);
        normalizer.fit(trainingData);
    }

    private RandomForest initRF() throws Exception {
        Log.i("initRF", "호출");
        int numTrees = 50;// 트리 수
        String dirPath = getFilesDir().getAbsolutePath();
        File savefile = new File(dirPath + "/" + RF_TrainData);
        //일치하는 파일이 없으면 생성
        if (!savefile.exists()) {
            try {
                String sCurrentLine;
                InputStream is = getAssets().open("rf_training_certain.csv"); //asset에 있는걸 파일 내부로
                BufferedReader br = new BufferedReader(new InputStreamReader(is));
                FileOutputStream fos = new FileOutputStream(savefile);
                PrintWriter pw = new PrintWriter(fos);
                //---------------------
                while ((sCurrentLine = br.readLine()) != null) {
                    if (sCurrentLine != null) {
//                        Log.i("sCurrentLintRF",sCurrentLine);
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

    private MultiLayerNetwork initDP() throws IOException, InterruptedException {
        Log.i("initDP", "호출");
        trainingData = readCSVDataset(DP_TrainData, batchSizeTraining, labelIndex, numClasses);
//        System.out.println(trainingData.getFeatureMatrix());

        //훈련 데이터 정규화
        normalizer.fit(trainingData);
        normalizer.transform(trainingData);

        int outputNum = 2; //범주수
        int iterations = 1000; //최적화 반복
        long seed = 9; //랜덤값

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed) //Random number generator seed. Used for reproducability between runs
                .iterations(iterations) // 최적화 반복
                .activation(Activation.LEAKYRELU) // 활성화 함수
                .weightInit(WeightInit.XAVIER) // 초기가중치
                .learningRate(0.1) // 학습률,Learning rate. Defaults to 1e-1
                .list() // Create a ListBuilder (for creating a MultiLayerConfiguration)
                .layer(0, new DenseLayer.Builder().nIn(ATTR_NUM).nOut(2).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .layer(2, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .layer(3,
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .activation(Activation.SOFTMAX).nIn(2).nOut(outputNum).build())
                .backprop(true) // 역전파
                .pretrain(false) // 사전학습
                .build();

        // run the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init(); // Initialize the MultiLayerNetwork. This should be called once before the network is used.
        model.setListeners(new ScoreIterationListener(100)); // 설정된 iteration 마다  원하는 행동들을 해서  트레이닝 과정을 볼수  있다.
        model.fit(trainingData); //모델 훈련

        Log.i("Network Model", "Finished Network model construction");

        //모델 저장.
        File locationToSave = new File(getFilesDir().getAbsolutePath() + savedDpModel);      //Where to save the network. Note: the file is in .zip format - can be opened externally
        boolean saveUpdater = true;                                             //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
        if (!locationToSave.exists()) {
            Log.i("Network Model", "Saved Model 없음 생성해야됨");
            ModelSerializer.writeModel(model, locationToSave, saveUpdater);
        }
        Log.i("Network Model", "Saved Network Model");
        return model;
    }

    private ClusteringDemo initClu() {
        Log.i("initClu", "호출");
        try {
            InputStream isTrain = getAssets().open("clu_training.arff");
            BufferedReader br = new BufferedReader(new InputStreamReader(isTrain));
            FileOutputStream fos = new FileOutputStream(new File(getFilesDir().getAbsolutePath() + "/" + CLU_TrainData));
            PrintWriter pw = new PrintWriter(fos);
            //---------------------
            String sCurrentLine;
            while ((sCurrentLine = br.readLine()) != null) {
                if (sCurrentLine != null) {
//                    Log.i("sCurrentLintClu",sCurrentLine);
                    pw.println(sCurrentLine);
                    pw.flush();
                }
            }

            InputStream isTrain2 = new FileInputStream(new File(getFilesDir().getAbsolutePath() + "/" + CLU_TrainData));
            cd = new ClusteringDemo(getApplicationContext(), isTrain2);
            Log.i("clusterer 빌드", "ok");

            //클러스터링 모델 직렬화
            FileOutputStream fosSerial = new FileOutputStream(getFilesDir().getAbsolutePath() + savedCluModel);
            BufferedOutputStream bosSerial = new BufferedOutputStream(fosSerial);
            ObjectOutputStream oos = new ObjectOutputStream(bosSerial);
            oos.writeObject(cd);

            pw.close();
            fos.close();
            br.close();
            isTrain.close();
            isTrain2.close();
            oos.close();
            bosSerial.close();
            fosSerial.close();

//            //일치하는 파일이 없으면 생성
//            if (!savefile.exists()) {
//                try {
//                    String sCurrentLine;
//                    BufferedReader br = new BufferedReader(new InputStreamReader(isTrain));
//                    FileOutputStream fos = new FileOutputStream(savefile);
//                    PrintWriter pw = new PrintWriter(fos);
//                    //---------------------
//                    while ((sCurrentLine = br.readLine()) != null) {
//                        if (sCurrentLine != null) {
//                            Log.i("sCurrentLintClu",sCurrentLine);
//                            pw.println(sCurrentLine);
//                            pw.flush();
//                        }
//                    }
//                    pw.close();
//                    fos.close();
//                    br.close();
//                    isTrain.close();
//                } catch (IOException e) {
//                    Log.i("error",e.toString());
//                    e.printStackTrace();
//                }
//            } else {
//                Log.i("파일ㅁ","있음");
//            }
//            //클러스터링 arff 파일 저장
//            BufferedReader br = new BufferedReader(new InputStreamReader(isTrain));
//
//            FileOutputStream fosSave = new FileOutputStream(savefile);
//            PrintWriter pw = new PrintWriter(fosSave);
//            //---------------------
//            while ((sCurrentLine = br.readLine()) != null) {
//                    Log.i("sCurrentLineCLU",sCurrentLine);
//                    pw.println(sCurrentLine);
//                    pw.flush();
//            }
//
//            pw.close();
//            fosSave.close();
//            br.close();
//            isTrain.close();
            Log.i("cluster 모델,훈련파일 저장", "ok");
        } catch (Exception e) {
            Log.i("clustering", "빌드or모델 저장 실패");
            e.printStackTrace();
        }
        return cd;
    }

    public void modelDetecting(double first, double last, double avgD, double maxD, double dist, double second, double clearance) throws InterruptedException, IOException, Exception {
        Log.i("modelDetecting", "호출");
        Log.i("detecting", (++c) + "회 실행");

        //검지 시작 시간
        long start = System.currentTimeMillis();

        int[] attributes_rf = new int[ATTR_NUM];
        double[] attributes_dp = new double[ATTR_NUM];
        double[] attributes_clu = new double[ATTR_NUM];

        //인공 신경망 테스트 데이터
        attributes_dp[0] = first; //처음속도
        attributes_dp[1] = last; //나중속도
        attributes_dp[2] = avgD; //평균감속도
        attributes_dp[3] = maxD; //최대감속도
        attributes_dp[4] = dist; //이동거리
        attributes_dp[5] = second; //이동시간
        attributes_dp[6] = clearance; //차간거리

        //랜포,클러스터링 테스트 데이터
        for (int i = 0; i < ATTR_NUM; i++) {
            attributes_rf[i] = (int) attributes_dp[i];
            attributes_clu[i] = attributes_dp[i];
        }

        //테스트데이터 Log
        Log.i("attribute_rf", "result:" + Arrays.toString(attributes_rf));
        Log.i("attribute_dp", "result:" + Arrays.toString(attributes_dp));
        Log.i("attribute_clu", "result:" + Arrays.toString(attributes_clu));

        // 랜포 검지
        result_rf = detectingWithRaF(attributes_rf);

        // 인공신경망 검지
        result_dp = detectingWithDp(attributes_dp);

        // 클러스터링 검지
        result_clu = detectingWithClu(attributes_clu);

        Log.i("처음 ~ detecting 걸리는 시간", (System.currentTimeMillis() - decletime) / 1000.0 + "s");

        //최종 검지
        if (result_rf == RISK_SITUATION && result_dp == RISK_SITUATION && result_clu == RISK_SITUATION) {
//            Toast.makeText(this.getApplicationContext(), "최종 검지 : 위험", Toast.LENGTH_SHORT).show();

            double[] temp_dp = new double[]{attributes_dp[0], attributes_dp[1], attributes_dp[2], attributes_dp[3], attributes_dp[4], attributes_dp[5], attributes_dp[6], 1};
            buffer_risky.add(temp_dp);

            DataOutputStream dos = sm.getServer_dos();
            dos.writeUTF("alarm");
            dos.writeUTF("metadata"); //gps 정보 전달
            dos.writeDouble(lct.getLatitude()); //위험 발생한 위치 위도
            dos.writeDouble(lct.getLongitude()); //위험 발생한 위치 경도
            dos.writeDouble(spd); //위험 발생했을 때 속도
            dos.writeUTF(stdTime+""); //위험 발생했을 때 시간

            Log.i("최종 검지 ", "위험");
            Log.i("최종 detecting 및 서버 전송시간", (System.currentTimeMillis() - decletime) / 1000.0 + "s");

        } else if (result_rf == NORMAL_SITUATION && result_dp == NORMAL_SITUATION && result_clu == NORMAL_SITUATION) {
//            Toast.makeText(this.getApplicationContext(), "최종 검지 : 일반", Toast.LENGTH_SHORT).show();
            double[] temp_dp = new double[]{attributes_dp[0], attributes_dp[1], attributes_dp[2], attributes_dp[3], attributes_dp[4], attributes_dp[5], attributes_dp[6], 0};
            buffer_normal.add(temp_dp);

            Log.i("최종 검지 ", "일반");
            Log.i("최종 detecting 및 서버 전송시간", (System.currentTimeMillis() - decletime) / 1000.0 + "s");
        } else {
//            Toast.makeText(this.getApplicationContext(), "최종 검지 : 의심", Toast.LENGTH_SHORT).show();
//            DataOutputStream dos = sm.getServer_dos();
//            dos.writeUTF("alarm");
//            dos.writeUTF("metadata"); //gps 정보 전달
//            dos.writeDouble(lct.getLatitude()); //위도
//            dos.writeDouble(lct.getLongitude()); //경도
//            dos.writeDouble(spd); //속도
//            dos.writeUTF(stdTime+""); //시간

            Log.i("최종 검지 ", "의심");
            Log.i("최종 detecting 및 서버 전송시간", (System.currentTimeMillis() - decletime) / 1000.0 + "s");
        }

//        detectThread= new Thread(null, detecting); //스레드 생성후 스레드에서 작업할 함수 지정(detecting)
//        detectThread.start();
    }

    public int detectingWithRaF(int[] attributes_rf) {
        int tempResult = RaF.realtime_Start(attributes_rf);
        return tempResult;
    }

    public int detectingWithDp(double[] attributes_dp) {
        INDArray testData = Nd4j.create(attributes_dp);//배열 -> INDArray 변환
        Map<String, Object> formattedTestData = realtime_changeDataForamt(testData);
        normalizer.transform(testData);//테스트 데이터 정규화
        INDArray output = model.output(testData);//테스트
        int tempResult = setFittedClassifiers(output, formattedTestData);
        logRisky(formattedTestData);
//        Log.i("output",output+".");
        return tempResult;
    }

    public int detectingWithClu(double[] attributes_clu) {
        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(new File(getFilesDir().getAbsolutePath() + "/" + tempCluTestData), false));
            PrintWriter pw = new PrintWriter(bw);
            pw.println("@relation test");
            pw.println("@attribute before numeric");
            pw.println("@attribute after numeric");
            pw.println("@attribute avg numeric");
            pw.println("@attribute max numeric");
            pw.println("@attribute dis numeric");
            pw.println("@attribute sec numeric");
            pw.println("@attribute cle numeric");
            pw.println("@data");
            pw.println(attributes_clu[0] + "," + attributes_clu[1] + "," + attributes_clu[2] + "," + attributes_clu[3] + "," + attributes_clu[4] + "," + attributes_clu[5] + "," + attributes_clu[6]);
            pw.flush();

            InputStream is = new FileInputStream(new File(getFilesDir().getAbsolutePath() + "/" + tempCluTestData));
            double[] result = cd.test(getApplicationContext(), is);
//            Log.i("clu_result",Arrays.toString(result));
            int tempResult;
            int risky_index = 1;

            if (result[risky_index] == 1.0) {
                tempResult = RISK_SITUATION;
                Log.i("클러스터링 검지 데이터", "[before:" + attributes_clu[0] + ", after:" + attributes_clu[1] + ", avg_decel:"
                        + attributes_clu[2] + ", max_decel:" + attributes_clu[3] + ", distance:"
                        + attributes_clu[4] + ", second:" + attributes_clu[5] + "clearance:" + attributes_clu[6] + "]");
                Log.i("클러스터링 검지 결과", " -위험");
            } else {
                tempResult = NORMAL_SITUATION;
                Log.i("클러스터링 검지 데이터", "[before:" + attributes_clu[0] + ", after:" + attributes_clu[1] + ", avg_decel:"
                        + attributes_clu[2] + ", max_decel:" + attributes_clu[3] + ", distance:"
                        + attributes_clu[4] + ", second:" + attributes_clu[5] + "clearance:" + attributes_clu[6] + "]");
                Log.i("클러스터링 검지 결과", " -일반");
            }

//            Log.i("result",Arrays.toString(result) );
            return tempResult;
        } catch (Exception e) {
            e.printStackTrace();
            return 0; //오류 상황
        }
    }

    public void logRisky(Map<String, Object> risky) {

        Log.i("인공신경망 검지 데이터", "[before:" + risky.get("before") + ", after:" + risky.get("after") + ", avg_decel:"
                + risky.get("avg_decel") + ", max_decel:" + risky.get("max_decel") + ", distance:"
                + risky.get("distance") + ", second:" + risky.get("second") + "clearance:" + risky.get("clearance") + "]");
        if (risky.get("classifier").equals("risky")) {
            Log.i("인공신경망 검지 결과", " -위험");
        } else {
            Log.i("인공신경망 검지 결과", " -일반");
        }

    }

    public int setFittedClassifiers(INDArray output, Map<String, Object> risky) {
        String temp = classifiers.get(maxIndex(getFloatArrayFromSlice(output)));
        risky.put("classifier", temp);
        if (temp.equals("risky")) {
            return RISK_SITUATION;
        } else {
            return NORMAL_SITUATION;
        }
    }

    public float[] getFloatArrayFromSlice(INDArray rowSlice) {
        float[] result = new float[rowSlice.columns()];//rowSlice.columns() 범주 수
        for (int i = 0; i < rowSlice.columns(); i++) {
            result[i] = rowSlice.getFloat(i);
        }
        return result;
    }

    public int maxIndex(float[] vals) { //vals는 범주수
        int maxIndex = 0;
        for (int i = 1; i < vals.length; i++) {
            float newnumber = vals[i];
            if ((newnumber > vals[maxIndex])) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public Map<String, Object> realtime_changeDataForamt(INDArray testData) {
        Map<String, Object> driving_information = new HashMap<>();
        driving_information.put("before", testData.getDouble(0));
        driving_information.put("after", testData.getDouble(1));
        driving_information.put("avg_decel", testData.getDouble(2));
        driving_information.put("max_decel", testData.getDouble(3));
        driving_information.put("distance", testData.getDouble(4));
        driving_information.put("second", testData.getDouble(5));
        driving_information.put("clearance", testData.getDouble(6));
        return driving_information;
    }

    public Map<Integer, String> readEnumCSV(String csvFileClasspath) {
        try {
            InputStream is = getAssets().open(csvFileClasspath);
            List<String> lines = IOUtils.readLines(is);
            Map<Integer, String> enums = new HashMap<>();
            for (String line : lines) {
                String[] parts = line.split(",");
                enums.put(Integer.parseInt(parts[0]), parts[1]);
            }
            return enums;
        } catch (Exception e) {
            return null;
        }
    }

    private DataSet readCSVDataset(String csvFileClasspath, int batchSize, int labelIndex, int numClasses) throws IOException, InterruptedException {
        Log.i("readCSVDataset", "호출");
        String dirPath = getFilesDir().getAbsolutePath();
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
                InputStream is = getAssets().open("dp_training_certain.csv"); //assets 폴더에 있는것 읽어오기
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

    Handler handler2 = new Handler() {
        public void handleMessage(Message msg) {
            switch (msg.what) {
                case (PROGRESS_DIALOG):
                    progressDialog = new ProgressDialog(getApplicationContext());
                    progressDialog.setProgressStyle(ProgressDialog.STYLE_SPINNER);
                    progressDialog.setMessage("model relearning..");
                case (CREATE_DIALOG):
                    progressDialog = new ProgressDialog(getApplicationContext());
                    progressDialog.setProgressStyle(ProgressDialog.STYLE_SPINNER);
                    progressDialog.setMessage("create model..");
                case (LOAD_DIALOG):
                    progressDialog = new ProgressDialog(getApplicationContext());
                    progressDialog.setProgressStyle(ProgressDialog.STYLE_SPINNER);
                    progressDialog.setMessage("load model..");
            }
        }
    };

    //Toast 메시지
    Runnable updateResults = new Runnable() {
        public void run() {
            progressDialog.dismiss();
            removeDialog(PROGRESS_DIALOG);
            Toast.makeText(getApplicationContext(), "데이터 " + updated_count + "개 재학습 완료", Toast.LENGTH_LONG).show();
        }
    };
    //Toast 메시지
    Runnable createAllModel = new Runnable() {
        public void run() {
            Toast.makeText(getApplicationContext(), "Create All Model", Toast.LENGTH_LONG).show();
            Toast.makeText(getApplicationContext(), "Detecting Start", Toast.LENGTH_LONG).show();
        }
    };
    //Toast 메시지
    Runnable loadAllModel = new Runnable() {
        public void run() {
            Toast.makeText(getApplicationContext(), "Load All Model", Toast.LENGTH_LONG).show();
            Toast.makeText(getApplicationContext(), "Detecting Start", Toast.LENGTH_LONG).show();
        }
    };

    public void registerAlarm() {
        Log.i("alarm", "registerAlram 호출");

        Intent intent = new Intent(this, AlarmService_Service.class);
        PendingIntent sender = PendingIntent.getBroadcast(this, 0, intent, 0);

        try {
            // 내일 아침 8시 10분에 처음 시작해서, 24시간 마다 실행되게
            Date tomorrow = new SimpleDateFormat("yyyy-MM-dd hh:mm:ss").parse("2017-11-12 00:00:01");
            AlarmManager am = (AlarmManager) getSystemService(ALARM_SERVICE);
            am.setInexactRepeating(AlarmManager.RTC, tomorrow.getTime(), 7 * 24 * 60 * 60 * 1000, sender);
        } catch (Exception e) {
            Log.i("asd", "aaa");
            e.printStackTrace();
        }
    }

    protected Dialog onCreateDialog(int id) {
        switch (id) {
            case (PROGRESS_DIALOG):
                progressDialog = new ProgressDialog(this);
                progressDialog.setProgressStyle(ProgressDialog.STYLE_SPINNER);
                progressDialog.setMessage("model relearning..");
                return progressDialog;
            case (CREATE_DIALOG):
                progressDialog = new ProgressDialog(this);
                progressDialog.setProgressStyle(ProgressDialog.STYLE_SPINNER);
                progressDialog.setMessage("create model..");
                return progressDialog;
            case (LOAD_DIALOG):
                progressDialog = new ProgressDialog(this);
                progressDialog.setProgressStyle(ProgressDialog.STYLE_SPINNER);
                progressDialog.setMessage("load model..");
                return progressDialog;
        }
        return null;
    }

    @Override
    public void onLocationChanged(Location location) {
        if (location != null)
            lct = location;
    }

    private void stopLocationUpdates() {
        if (googleApiClient != null && googleApiClient.isConnected())
            googleApiClient.disconnect();
    }

    @Override
    public void onConnected(Bundle bundle) {
        if (ActivityCompat.checkSelfPermission(this, android.Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(this, android.Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED)
            return;

        LocationRequest locationRequest;
        locationRequest = LocationRequest.create();
        locationRequest.setInterval(1000);
        locationRequest.setFastestInterval(500);
        locationRequest.setPriority(LocationRequest.PRIORITY_HIGH_ACCURACY);

        LocationServices.FusedLocationApi.requestLocationUpdates(googleApiClient, locationRequest, this);
    }

    @Override
    public void onConnectionFailed(@NonNull ConnectionResult connectionResult) {
        stopLocationUpdates();
    }

    @Override
    public void onConnectionSuspended(int i) {
    }

    private BaseLoaderCallback mLoaderCallback;

    protected void onStop() {
        super.onStop();
        Log.i("onStop", "호출");
        if (detectThread != null) {
            detectThread.interrupt();
        }

        //buffer에 있는거 파일에 쓰기
        if (buffer_risky.size() > 0 && buffer_normal.size() > 0) {
            Log.i("buffer", "파일쓰기시작");
            try {
                BufferedWriter bwrisky = new BufferedWriter(new FileWriter(new File(getFilesDir().getAbsolutePath() + "/" + buffer_riskySave), false));
                PrintWriter pwrisky = new PrintWriter(bwrisky);
                BufferedWriter bwnormal = new BufferedWriter(new FileWriter(new File(getFilesDir().getAbsolutePath() + "/" + buffer_normalSave), false));
                PrintWriter pwnormal = new PrintWriter(bwnormal);

                for (int i = 0; i < buffer_risky.size(); i++) {
                    double[] temp = buffer_risky.get(i);
                    String s = "";
                    for (int j = 0; j < temp.length; j++) {
                        if (j != temp.length - 1) {
                            s += temp[j] + ",";
                        } else {
                            s += temp[j];
                        }
                    }
                    pwrisky.println(s);
                    pwrisky.flush();
                }

                for (int i = 0; i < buffer_normal.size(); i++) {
                    double[] temp = buffer_normal.get(i);
                    String s = "";
                    for (int j = 0; j < temp.length; j++) {
                        if (j != temp.length - 1) {
                            s += temp[j] + ",";
                        } else {
                            s += temp[j];
                        }
                    }
                    pwnormal.println(s);
                    pwnormal.flush();
                }

                pwrisky.close();
                bwrisky.close();
                pwnormal.close();
                bwnormal.close();
                Log.i("buffer", "파일쓰기완료");
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    public void onPause() {
        super.onPause();
        Log.i("onPause", "호출");
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        Log.i("onResume", "호출");
        if (!OpenCVLoader.initDebug())
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_2_0, this, mLoaderCallback);
        else if (mLoaderCallback != null)
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
    }

    public void onDestroy() {
        super.onDestroy();
        Log.i("onDestroy", "호출");
        if (fw != null)
            try {
                fw.close();
            } catch (Exception e) {
                e.printStackTrace();
            }

        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();

    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        TimerTask timerTask = new TimerTask() {
            @Override
            public void run() {
                realProc();
            }
        };
        timer = new Timer();
        timer.schedule(timerTask, 3000, 5000);
    }

    private void realProc() {
        Log.i("lct",lct+"");
        if (lct == null)
            return;

        if (oLct == null)
            spd = 0;
        else
            spd = lct.distanceTo(oLct) * 3.6;

        oLct = lct;

        stdTime = System.currentTimeMillis();

        if (fw != null)
            try {
                fw.write(stdTime + "," + spd + "," + distance[0] + "\n");
                fw.flush();
            } catch (Exception e) {
                e.printStackTrace();
            }

        if (speedList.size() > 1 && speedList.get(speedList.size() - 1) >= spd) {
            if(speedList.get(speedList.size()-1)==0 && speedList.get(speedList.size()-2)==0){
                Log.i("멈춰있음","ㅋ");
                speedList.clear();
                return;
            }
            decletime = System.currentTimeMillis();
            speedList.add(spd);
            aiProc();
        } else if (speedList.size() <= 1)
            speedList.add(spd);
        else
            speedList.clear();
    }

    public void aiProc() {
        double first, //: 처음 속도
                last, //: 나중 속도
                avgD, //: 평균 감속도
                maxD, //: 최대 감속도
                dist, //: 이동 거리
                second, //: 이동시간
                avg; //: 평균 속도
        //distance[0] : 현재 차간 거리

        first = speedList.get(0); //처음속도

        last = speedList.get(speedList.size() - 1); //나중속도

        avgD = -((first - last) / speedList.size()); //평균감속도
        maxD = 0;
        for (int i = 0; i < speedList.size() - 1; i++) {
            if (maxD < speedList.get(i) - speedList.get(i + 1)) {
                maxD = speedList.get(i) - speedList.get(i + 1);
            }
        }
        maxD = -maxD; //최대 감속도

        double sum = 0;
        for (double t : speedList) {
            sum += t;
        }
        avg = sum / speedList.size();
        dist = avg * speedList.size(); //이동거리
        second = speedList.size(); //이동시간

        //TO DO
        try {
            if (RaF != null && model != null && cd != null)
                modelDetecting(first, last, avgD, maxD, dist, second, distance[0]);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public void onCameraViewStopped() {
        timer.cancel();
        timer.purge();
        timer = null;
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat mRgba = inputFrame.rgba();

        proc(mRgba.getNativeObjAddr(), distance);

        return mRgba;
    }

    public native void proc(long matAddrRgba, double[] dis);

    static {
        System.loadLibrary("native-lib");
    }
}
